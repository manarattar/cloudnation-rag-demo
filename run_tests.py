"""
End-to-end quality + speed test for all key demo scenarios.
Run from project root:  python -X utf8 run_tests.py
"""

import os
import sys
import time

# Load key from .streamlit/secrets.toml (gitignored) if env var not already set.
# On CI/cloud, set GROQ_API_KEY in the environment directly.
if not os.environ.get("GROQ_API_KEY"):
    import pathlib
    import re as _re

    _secrets = pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml"
    if _secrets.exists():
        _m = _re.search(r'GROQ_API_KEY\s*=\s*"([^"]+)"', _secrets.read_text())
        if _m:
            os.environ["GROQ_API_KEY"] = _m.group(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.pipeline import (ingest_documents, is_collection_ready,  # noqa: E402
                           query)
from demo.sample_documents import DOCUMENTS  # noqa: E402

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _ok(text):
    print(f"  {GREEN}✓{RESET} {text}")


def _warn(text):
    print(f"  {YELLOW}⚠{RESET} {text}")


def _fail(text):
    print(f"  {RED}✗{RESET} {text}")


def _dim(text):
    print(f"  {DIM}{text}{RESET}")


# ── Test cases ────────────────────────────────────────────────────────────────
# (id, query, role, expected_keywords, should_have_answer, label)
# For blocking tests: keywords = list of words that must NOT appear in answer.
# For answer tests:   keywords = list of words that must appear.
CASES = [
    # RBAC — helpdesk sees no FIOD content
    (
        "rbac_block",
        "Tell me about FIOD fraud indicators and VAT carousel schemes.",
        "helpdesk",
        ["carrousel", "fraude", "50.000", "rode vlag"],  # must NOT appear
        False,
        "RBAC: helpdesk blocked from FIOD classified doc",
    ),
    # RBAC — fiod gets the full classified document
    (
        "rbac_allow",
        "Tell me about FIOD fraud indicators and VAT carousel schemes.",
        "fiod",
        ["btw", "carrousel", "fraude", "50.000", "rode vlag"],  # must appear
        True,
        "RBAC: fiod sees FIOD classified doc",
    ),
    # Box 1 tax rates — LLM answers in English with period decimal notation
    (
        "box1_rates",
        "What is the Box 1 income tax rate for 2024?",
        "helpdesk",
        ["36.97", "49.50", "75"],  # 36.97%, 49.50%, threshold €75k
        True,
        "Box 1 tax rates (36.97% / 49.50%)",
    ),
    # Cache hit — exact repeat of Box 1
    (
        "box1_cache",
        "What is the Box 1 income tax rate for 2024?",
        "helpdesk",
        ["36.97", "49.50"],
        True,
        "Box 1 cache hit (same query, same role)",
    ),
    # VAT rates — requires BTW e-learning doc (access_roles fixed to ["internal"])
    (
        "vat_rates",
        "What are the VAT (BTW) rates in the Netherlands?",
        "helpdesk",
        ["21", "9", "export"],
        True,
        "VAT rates (21%, 9%, 0% export)",
    ),
    # DGA salary — helpdesk must NOT see it (access_roles fixed to ["restricted"])
    (
        "dga_helpdesk",
        "What is the minimum salary for a DGA in 2024?",
        "helpdesk",
        ["56.000", "56,000"],  # must NOT appear
        False,
        "RBAC: helpdesk cannot see DGA policy (restricted)",
    ),
    # DGA salary — inspector has access
    (
        "dga_inspector",
        "What is the minimum salary for a DGA in 2024?",
        "inspector",
        ["56.000"],
        True,
        "DGA minimum salary 56000 (inspector role)",
    ),
    # Double deduction prohibited
    (
        "double_deduction",
        "Can I combine a home office deduction and childcare costs for the same expenses?",
        "helpdesk",
        ["niet", "dezelfde"],
        True,
        "Double deduction prohibited (home office + childcare)",
    ),
    # Kerstarrest Box 3
    (
        "kerstarrest",
        "What did the Hoge Raad rule in the Kerstarrest about Box 3?",
        "helpdesk",
        ["2021", "werkelijk", "rechtsherstel"],
        True,
        "Kerstarrest Box 3 ruling (ECLI:NL:HR:2021:1963)",
    ),
    # HR 2023 — home office DGA
    (
        "hr_2023_homeoffice",
        "What does ECLI:NL:HR:2023:123 say about home office deductions?",
        "helpdesk",
        ["zelfstandigheid", "geweigerd"],
        True,
        "HR 2023:123 — DGA home office deduction refused",
    ),
    # Box 3
    (
        "box3_rate",
        "What is the Box 3 tax rate and wealth threshold for 2024?",
        "helpdesk",
        ["57.000", "36"],
        True,
        "Box 3 rate 36%, threshold EUR 57000",
    ),
]

# Groq free tier: 6000 TPM. With 11 queries we can hit the ceiling.
# Wait 2 s between non-cached LLM calls to stay under the limit.
INTER_CALL_DELAY = 4.0

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print(f"\n{BOLD}+{'='*68}+{RESET}")
print(
    f"{BOLD}|   CloudNation RAG Demo  --  Full Quality & Speed Test            |{RESET}"
)
print(f"{BOLD}+{'='*68}+{RESET}")

print(f"\n{BOLD}Loading knowledge base...{RESET}")
t_ingest = time.time()
if not is_collection_ready():
    n = ingest_documents(DOCUMENTS)
    print(f"  Ingested {n} documents in {round((time.time()-t_ingest)*1000)} ms")
else:
    print("  Already loaded.")

# ── Run tests ──────────────────────────────────────────────────────────────────
results = []
last_was_llm_call = False

for i, (tid, q, role, keywords, should_answer, label) in enumerate(CASES, 1):
    print(f"\n{BOLD}[{i:02d}/{len(CASES)}] {label}{RESET}")
    print(f"  Role: {CYAN}{role}{RESET}  |  Query: {DIM}{q}{RESET}")

    if last_was_llm_call:
        time.sleep(INTER_CALL_DELAY)

    t0 = time.time()
    res = query(q, role=role)
    elapsed = res.get("latency_ms", round((time.time() - t0) * 1000))

    answer = res.get("answer", "").lower()
    grade = res.get("grade", "?")
    cached = res.get("cache_hit", False)
    cites = res.get("citations", [])
    last_was_llm_call = not cached

    # Speed verdict
    if cached:
        speed_tag = f"{GREEN}CACHE HIT -- {elapsed} ms{RESET}"
    elif elapsed < 1500:
        speed_tag = f"{GREEN}FAST -- {elapsed} ms{RESET}"
    elif elapsed < 4000:
        speed_tag = f"{YELLOW}OK -- {elapsed} ms{RESET}"
    else:
        speed_tag = f"{RED}SLOW -- {elapsed} ms{RESET}"

    # ── Blocking test: keywords must NOT appear ───────────────────────────
    if not should_answer:
        leaked = [kw for kw in keywords if kw.lower() in answer]
        blocked_ok = len(leaked) == 0
        verdict = "PASS" if blocked_ok else "FAIL"
        colour = GREEN if blocked_ok else RED
        print(f"  {colour}{verdict}{RESET} — grade={grade}  {speed_tag}")
        if not blocked_ok:
            _fail(f"Data leaked to {role}: {leaked}")
            _fail(f"Answer: {answer[:200]}")
        else:
            _dim(f"No sensitive keywords in answer. Grade: {grade}")
        results.append((tid, label, blocked_ok, elapsed, cached))
        continue

    # ── Answer test: keywords must appear ────────────────────────────────
    # Check for LLM error
    if answer.startswith("llm error"):
        print(f"  {RED}FAIL{RESET} — LLM error  {speed_tag}")
        _fail(answer[:200])
        results.append((tid, label, False, elapsed, cached))
        continue

    found = [kw for kw in keywords if kw.lower() in answer]
    missing = [kw for kw in keywords if kw.lower() not in answer]
    accuracy = len(found) / len(keywords) if keywords else 1.0

    if accuracy == 1.0:
        acc_tag = f"{GREEN}100%{RESET}"
        verdict = "PASS"
    elif accuracy >= 0.6:
        acc_tag = f"{YELLOW}{round(accuracy*100)}%{RESET}"
        verdict = "WARN"
    else:
        acc_tag = f"{RED}{round(accuracy*100)}%{RESET}"
        verdict = "FAIL"

    colour = GREEN if verdict == "PASS" else (YELLOW if verdict == "WARN" else RED)
    cache_label = "yes" if cached else "no"
    print(
        f"  {colour}{verdict}{RESET} -- accuracy={acc_tag}  grade={grade}"
        f"  cache={cache_label}  {speed_tag}"
    )
    print(f"  Citations: {cites}")

    if missing:
        _warn(f"Missing keywords: {missing}")
    if found:
        _dim(f"Found: {found}")

    raw = res.get("answer", "")
    excerpt = raw[:300] + ("..." if len(raw) > 300 else "")
    print(f"  {DIM}Answer ({len(raw)} chars): {excerpt}{RESET}")

    results.append((tid, label, verdict in ("PASS", "WARN"), elapsed, cached))


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*70}{RESET}")
print(f"{BOLD}SUMMARY{RESET}")
print(f"{'='*70}")

passed = sum(1 for *_, ok_flag, _, _ in results if ok_flag)
cache_hits = sum(1 for *_, _, c in results if c)
lats = [ms for _, _, _, ms, c in results if not c]
avg_lat = round(sum(lats) / len(lats)) if lats else 0

for tid, label, ok_flag, ms, c in results:
    tag = f"{GREEN}PASS{RESET}" if ok_flag else f"{RED}FAIL{RESET}"
    lat_str = "cache" if c else f"{ms} ms"
    print(f"  {tag}  [{lat_str:>9}]  {label}")

print(f"\n  Passed:      {passed}/{len(results)}")
print(f"  Cache hits:  {cache_hits}/{len(results)}")
print(f"  Avg latency (non-cached): {avg_lat} ms")
print(f"{'='*70}\n")
