"""
Module 1 — Ingestion & Knowledge Structuring
Hierarchical chunking for legal documents with full metadata preservation.

Design principle: every chunk must carry its full legislative address so the LLM
can cite it precisely. "Article 3.114, Paragraph 2" is metadata, not text — it
belongs in the payload, not embedded into the chunk body.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import TextNode

# ---------------------------------------------------------------------------
# 1. Document metadata schema
# ---------------------------------------------------------------------------


@dataclass
class LegalMetadata:
    """Structured address for every chunk in the corpus."""

    doc_id: str  # e.g. "AWR-2024"
    doc_type: str  # "legislation" | "case_law" | "policy" | "elearning"
    doc_title: str  # "Algemene Wet Rijksbelastingen 2024"
    chapter: Optional[str] = None  # "Chapter 3 — Income from Work"
    article: Optional[str] = None  # "Article 3.114"
    paragraph: Optional[str] = None  # "Paragraph 2"
    sub_paragraph: Optional[str] = None  # "Sub-paragraph a"
    ecli: Optional[str] = None  # "ECLI:NL:HR:2023:123"  (case law only)
    effective_date: Optional[str] = None
    classification: str = "public"  # "public" | "internal" | "restricted" | "fiod"
    access_roles: list = field(default_factory=lambda: ["*"])
    # ["*"] = everyone, ["inspector", "legal"] = restricted, ["fiod"] = FIOD-only

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def citation(self) -> str:
        """Human-readable citation string injected into every retrieved chunk."""
        parts = [self.doc_title]
        if self.chapter:
            parts.append(self.chapter)
        if self.article:
            parts.append(self.article)
        if self.paragraph:
            parts.append(self.paragraph)
        if self.ecli:
            parts.append(f"[{self.ecli}]")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# 2. Hierarchy-aware text splitter for legal codes
# ---------------------------------------------------------------------------

LEGAL_HIERARCHY = [
    # (level_name, regex_pattern)
    ("chapter", r"^(Hoofdstuk|Chapter)\s+[\dIVX]+[^\n]*"),
    ("article", r"^(Artikel|Article)\s+[\d\.]+[^\n]*"),
    ("paragraph", r"^\d+\.\s+"),  # "1. " numbered paragraphs
    ("sub_paragraph", r"^[a-z]\.\s+"),  # "a. " lettered items
]


def extract_legal_address(text_block: str) -> dict:
    """
    Parse a text block and extract its legislative position.
    Returns a dict of {level: heading_text}.
    """
    address = {}
    for level, pattern in LEGAL_HIERARCHY:
        match = re.match(pattern, text_block.strip(), re.MULTILINE)
        if match:
            address[level] = match.group(0).strip()
    return address


def build_legal_chunks(
    raw_text: str,
    base_metadata: LegalMetadata,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextNode]:
    """
    Produces chunks where every node carries:
      - Its text content (the actual legal prose)
      - Its legislative address in metadata (chapter/article/paragraph)
      - A pre-built citation string for zero-effort retrieval attribution
    """
    doc = Document(text=raw_text, metadata=base_metadata.to_dict())

    # HierarchicalNodeParser creates parent → child node trees.
    # Leaf nodes are the actual chunks; parents are stored for context window
    # expansion (small-to-big retrieval).
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],  # large=section, medium=paragraph, small=sentence
        chunk_overlap=chunk_overlap,
    )
    nodes = parser.get_nodes_from_documents([doc])
    leaf_nodes = get_leaf_nodes(nodes)

    enriched = []
    for node in leaf_nodes:
        # Extract legislative address from the text itself
        address = extract_legal_address(node.text)

        # Merge extracted address into metadata
        node.metadata.update(address)

        # Inject citation into metadata for retrieval-time attribution
        node.metadata["citation"] = base_metadata.citation
        node.metadata["chunk_size"] = len(node.text)

        enriched.append(node)

    return enriched


# ---------------------------------------------------------------------------
# 3. Case-law specific chunking (ECLI documents)
# ---------------------------------------------------------------------------


def build_case_law_chunks(
    verdict_text: str,
    ecli: str,
    court: str,
    date: str,
    classification: str = "public",
    access_roles: list = None,
) -> list[TextNode]:
    """
    Court verdicts have a fixed structure:
      HEADER → FACTS → LEGAL_CONSIDERATION → DECISION
    We preserve this structure as metadata so the LLM knows whether a chunk
    is factual background or the binding legal ruling.
    """
    sections = {
        "facts": r"(?i)(feiten|facts|background)",
        "legal_consideration": r"(?i)(overwegingen|legal consideration|beoordeling)",
        "decision": r"(?i)(beslissing|decision|uitspraak)",
    }

    base_meta = LegalMetadata(
        doc_id=ecli,
        doc_type="case_law",
        doc_title=f"Verdict {ecli} — {court} ({date})",
        ecli=ecli,
        effective_date=date,
        classification=classification,
        access_roles=access_roles or ["*"],
    )

    # Split into structural sections first
    chunks = []
    current_section = "preamble"
    current_text = []

    for line in verdict_text.splitlines():
        detected = None
        for section_name, pattern in sections.items():
            if re.search(pattern, line):
                detected = section_name
                break

        if detected and current_text:
            meta = LegalMetadata(**base_meta.__dict__)
            meta.chapter = current_section
            chunks.extend(
                build_legal_chunks("\n".join(current_text), meta, chunk_size=384)
            )
            current_section = detected
            current_text = []
        else:
            current_text.append(line)

    # Final section
    if current_text:
        meta = LegalMetadata(**base_meta.__dict__)
        meta.chapter = current_section
        chunks.extend(build_legal_chunks("\n".join(current_text), meta, chunk_size=384))

    return chunks


# ---------------------------------------------------------------------------
# 4. Ingestion pipeline entry point
# ---------------------------------------------------------------------------


def ingest_document(
    raw_text: str,
    metadata: LegalMetadata,
) -> list[TextNode]:
    """
    Routes documents to the correct chunker based on doc_type.
    All output nodes are ready for embedding and upsert into Qdrant.
    """
    if metadata.doc_type == "case_law":
        return build_case_law_chunks(
            raw_text,
            ecli=metadata.ecli or metadata.doc_id,
            court=metadata.doc_title,
            date=metadata.effective_date or "unknown",
            classification=metadata.classification,
            access_roles=metadata.access_roles,
        )
    else:
        return build_legal_chunks(raw_text, metadata)


# ---------------------------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_legislation = """
    Hoofdstuk 3 — Belastbaar inkomen uit werk en woning

    Artikel 3.114 Persoonsgebonden aftrek
    1. De persoonsgebonden aftrek wordt in aanmerking genomen bij het bepalen van
       het belastbare inkomen uit werk en woning.
    2. De aftrek bedraagt het bedrag van de in het kalenderjaar op de belastingplichtige
       drukkende persoonsgebonden aftrekposten.
       a. Aftrekposten als bedoeld in afdeling 6.2 worden in aanmerking genomen voor
          zover zij betrekking hebben op het kalenderjaar.
    """

    meta = LegalMetadata(
        doc_id="IB-2024-3114",
        doc_type="legislation",
        doc_title="Wet Inkomstenbelasting 2024",
        chapter="Hoofdstuk 3",
        article="Artikel 3.114",
        effective_date="2024-01-01",
        classification="public",
        access_roles=["*"],
    )

    nodes = ingest_document(sample_legislation, meta)
    for n in nodes:
        print(f"[{n.metadata['citation']}]")
        print(
            f"  article={n.metadata.get('article')} | paragraph={n.metadata.get('paragraph')}"
        )
        print(f"  text preview: {n.text[:80]}...")
        print()
