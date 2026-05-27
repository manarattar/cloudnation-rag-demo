"""
Fetch ~100 Hoge Raad belastingrecht decisions from rechtspraak.nl open data.

Run once:
    python -m demo.fetch_corpus

Output: demo/corpus_data.json  (list of document dicts matching sample_documents schema)

API used:
  Search:  https://data.rechtspraak.nl/uitspraken/zoeken  (Atom feed, XML)
  Content: https://data.rechtspraak.nl/uitspraken/content?id=ECLI:NL:HR:...  (XML)

Strategy:
  - Query 8 different tax keywords → collect HR-prefix ECLIs from each result page
  - Deduplicate, then fetch full document XML for each ECLI
  - Strip XML tags, extract metadata, save as JSON
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SEARCH_BASE = "https://data.rechtspraak.nl/uitspraken/zoeken"
CONTENT_BASE = "https://data.rechtspraak.nl/uitspraken/content"
OUT_PATH = Path(__file__).parent / "corpus_data.json"
MAX_DOCS = 120
RATE_LIMIT_SLEEP = 0.12  # stay well under 10 req/s

ATOM_NS = "http://www.w3.org/2005/Atom"

# Tax keywords — each gives a different result page, so different HR ECLIs
KEYWORDS = [
    "inkomstenbelasting",
    "omzetbelasting",
    "BTW",
    "vennootschapsbelasting",
    "box 3",
    "DGA gebruikelijk loon",
    "erfbelasting",
    "overdrachtsbelasting",
]


def _fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "CloudNation-RAG-Demo/1.0"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5)
            elif attempt == retries - 1:
                raise
            else:
                time.sleep(2)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {url}")


def search_hr_eclis(
    keyword: str, max_results: int = 1000, sort: str = "DESC"
) -> list[str]:
    """Return all ECLI:NL:HR:* identifiers found in the search results page."""
    params: dict[str, str] = {"q": keyword, "max": str(max_results), "sort": sort}
    url = f"{SEARCH_BASE}?{urllib.parse.urlencode(params)}"
    xml_str = _fetch(url).decode("utf-8").lstrip("﻿")
    root = ET.fromstring(xml_str)
    ns = {"a": ATOM_NS}
    return [
        e.find("a:id", ns).text.strip()
        for e in root.findall("a:entry", ns)
        if e.find("a:id", ns) is not None
        and (e.find("a:id", ns).text or "").startswith("ECLI:NL:HR:")
    ]


def _strip_xml_tags(xml_str: str) -> str:
    """Remove all XML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", xml_str)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_meta(root: ET.Element) -> dict:
    """Pull structured metadata out of the rdf:RDF block."""
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dcterms": "http://purl.org/dc/terms/",
        "psi": "http://psi.rechtspraak.nl/",
    }
    desc = root.find(".//rdf:Description", ns)
    if desc is None:
        return {}

    def _text(tag: str) -> str:
        el = desc.find(tag, ns)
        return (el.text or "").strip() if el is not None else ""

    ecli = _text("dcterms:identifier")
    date = _text("dcterms:date")
    zaaknr = _text("psi:zaaknummer")
    subject_el = desc.find("dcterms:subject", ns)
    subject = (
        subject_el.get("{http://www.w3.org/2000/01/rdf-schema#}label", "")
        if subject_el is not None
        else ""
    )
    return {"ecli": ecli, "date": date, "zaaknr": zaaknr, "subject": subject}


def fetch_document(ecli: str) -> dict | None:
    """Fetch and parse a single ECLI document. Returns None on failure."""
    url = f"{CONTENT_BASE}?id={urllib.parse.quote(ecli)}"
    try:
        data = _fetch(url)
    except Exception as e:
        print(f"  SKIP {ecli}: fetch error: {e}", file=sys.stderr)
        return None

    xml_str = data.decode("utf-8")
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        print(f"  SKIP {ecli}: XML parse error: {e}", file=sys.stderr)
        return None

    meta = _extract_meta(root)
    if not meta.get("ecli"):
        meta["ecli"] = ecli

    # Extract all text by stripping tags from the full XML
    text = _strip_xml_tags(xml_str)

    # Remove the metadata preamble (everything before the first paragraph of the decision)
    # The actual ruling text usually starts after the metadata block
    # Remove known metadata noise phrases
    for noise in [
        r"Rechtspraak\.nl",
        r"Copyright \d{4} Rechtspraak\.",
        r"text/xml",
        r"public",
    ]:
        text = re.sub(noise, "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Skip documents with very little actual text
    if len(text) < 300:
        print(f"  SKIP {ecli}: too short ({len(text)} chars)", file=sys.stderr)
        return None

    date = meta.get("date", "")
    zaaknr = meta.get("zaaknr", "")
    subject = meta.get("subject", "Belastingrecht")

    doc_title = f"Hoge Raad {date}, {ecli}"
    article = ecli
    citation_suffix = f"zaaknr. {zaaknr}" if zaaknr else ""
    if citation_suffix:
        doc_title += f" (zaaknr. {zaaknr})"

    return {
        "doc_id": ecli.replace(":", "-"),
        "doc_type": "case_law",
        "doc_title": doc_title,
        "article": article,
        "paragraph": subject or "Belastingrecht",
        "classification": "public",
        "access_roles": ["*"],
        "text": text[:6000],  # cap at 6K chars; parent-child chunking handles the rest
    }


def main():
    print("=== Rechtspraak.nl ECLI Harvester ===")
    print(f"Target: {MAX_DOCS} Hoge Raad belastingrecht decisions")
    print()

    # Step 1: collect unique HR ECLIs with two sweeps:
    #   DESC sort — the 44 most recently updated (2026) decisions
    #   ASC sort  — 28 oldest decisions (1998-1999 era)
    # Different sort orders surface different parts of the 3.7M result set.
    seen: set[str] = set()
    eclis: list[str] = []

    for sort in ("DESC", "ASC"):
        print(f"Searching belasting sort={sort} ...", end=" ", flush=True)
        try:
            found = search_hr_eclis("belasting", sort=sort)
            new = [e for e in found if e not in seen]
            seen.update(new)
            eclis.extend(new)
            print(f"{len(found)} HR hits, {len(new)} new (total {len(eclis)})")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(RATE_LIMIT_SLEEP)

    eclis = eclis[:MAX_DOCS]
    print(f"\nCollected {len(eclis)} unique HR ECLIs — fetching documents...\n")

    # Step 2: fetch each document
    documents: list[dict] = []
    for i, ecli in enumerate(eclis, 1):
        print(f"[{i:3d}/{len(eclis)}] {ecli} ...", end=" ", flush=True)
        doc = fetch_document(ecli)
        if doc:
            documents.append(doc)
            print(f"OK ({len(doc['text'])} chars)")
        else:
            print("SKIP")
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"\nFetched {len(documents)} documents successfully.")

    # Step 3: save
    OUT_PATH.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
