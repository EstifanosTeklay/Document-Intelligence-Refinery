"""
Phase 0 — Triage Agent Standalone Runner
=========================================
Runs the full triage pipeline using only built-in libraries + pdfplumber.
No pydantic required. Uses dataclasses as a drop-in.

Usage:
    python3 run_phase0.py <path_to_pdf>
    python3 run_phase0.py                   # runs on any PDFs in ./data/
"""

import sys
import os
import json
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import pdfplumber

# ── Load config ───────────────────────────────────────────────────────
RULES_PATH = Path(__file__).parent / "extraction_rules.yaml"

def load_config():
    with open(RULES_PATH) as f:
        return yaml.safe_load(f)

cfg = load_config()
ORIGIN = cfg["origin_detection"]
LAYOUT = cfg["layout_detection"]
DOMAIN = cfg["domain_hints"]
ROUTING= cfg["strategy_routing"]

MIN_CHARS      = ORIGIN["min_chars_per_page"]
MAX_IMG_RATIO  = ORIGIN["max_image_area_ratio"]
SCAN_THRESHOLD = ORIGIN["scanned_page_threshold"]
MULTI_COL_THR  = LAYOUT["multi_column_threshold"]
TABLE_RATIO    = LAYOUT["table_heavy_page_ratio"]
FIG_RATIO      = LAYOUT["figure_heavy_page_ratio"]

# ── Colour output ─────────────────────────────────────────────────────
class C:
    NAVY   = "\033[34m"
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    PURPLE = "\033[35m"
    CYAN   = "\033[36m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    GREY   = "\033[90m"

def header(text):
    print(f"\n{C.BOLD}{C.NAVY}{'='*60}{C.RESET}")
    print(f"{C.BOLD}{C.NAVY}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.NAVY}{'='*60}{C.RESET}")

def section(text):
    print(f"\n{C.BOLD}{C.CYAN}  ── {text}{C.RESET}")

def ok(label, value):
    print(f"    {C.GREEN}✓{C.RESET}  {label:<35} {C.BOLD}{value}{C.RESET}")

def warn(label, value):
    print(f"    {C.YELLOW}⚠{C.RESET}  {label:<35} {C.YELLOW}{value}{C.RESET}")

def info(label, value):
    print(f"    {C.GREY}·{C.RESET}  {label:<35} {value}")

def decision(label, value, color=C.BOLD):
    print(f"    {C.PURPLE}▶{C.RESET}  {label:<35} {color}{C.BOLD}{value}{C.RESET}")

# ── Step 1: Extract page-level measurements ───────────────────────────
def extract_pages(pdf):
    pages = []
    for page in pdf.pages:
        try:
            w = page.width or 1
            h = page.height or 1
            area = w * h

            text  = page.extract_text() or ""
            chars = len(text.strip())

            images = page.images or []
            img_area = sum(img.get("width",0)*img.get("height",0) for img in images)
            img_ratio = min(img_area / area, 1.0)

            tables = page.extract_tables() or []
            words  = page.extract_words() or []

            pages.append({
                "page_num"       : page.page_number,
                "char_count"     : chars,
                "image_area_ratio": round(img_ratio, 4),
                "has_tables"     : len(tables) > 0,
                "table_count"    : len(tables),
                "has_figures"    : len(images) > 0,
                "figure_count"   : len(images),
                "words"          : words,
                "text"           : text,
                "error"          : None,
            })
        except Exception as e:
            pages.append({
                "page_num": page.page_number,
                "char_count": 0, "image_area_ratio": 0.0,
                "has_tables": False, "table_count": 0,
                "has_figures": False, "figure_count": 0,
                "words": [], "text": "", "error": str(e),
            })
    return pages

# ── Step 2: Detect origin type ────────────────────────────────────────
def detect_origin(pages):
    digital = scanned = 0
    page_verdicts = []
    for p in pages:
        is_scanned = (
            (p["char_count"] < MIN_CHARS and p["image_area_ratio"] > MAX_IMG_RATIO)
            or (p["char_count"] == 0 and p["image_area_ratio"] == 0)
        )
        if is_scanned:
            scanned += 1
            page_verdicts.append("SCAN")
        else:
            digital += 1
            page_verdicts.append("DIGI")

    total = max(len(pages), 1)
    ratio = scanned / total

    if ratio >= SCAN_THRESHOLD and digital == 0:
        origin = "scanned_image"
    elif ratio >= SCAN_THRESHOLD:
        origin = "mixed"
    else:
        origin = "native_digital"

    return origin, digital, scanned, ratio, page_verdicts

# ── Step 3: Detect layout complexity ─────────────────────────────────
def estimate_columns(words):
    if not words:
        return 1
    xs = [(w["x0"] + w["x1"]) / 2 for w in words]
    if len(set(xs)) <= 1:
        return 1
    xmin, xmax = min(xs), max(xs)
    if xmax - xmin < 10:
        return 1
    mid = (xmin + xmax) / 2
    left  = sum(1 for x in xs if x < mid)
    right = sum(1 for x in xs if x >= mid)
    total = len(xs)
    if 0.25 <= left/total <= 0.75 and 0.25 <= right/total <= 0.75:
        return 2
    return 1

def detect_layout(pages):
    content_pages = [p for p in pages if p["char_count"] > 0 or p["has_tables"]]
    if not content_pages:
        return "single_column", 0, 0, 0.0

    total_all = max(len(pages), 1)
    table_pgs = sum(1 for p in pages if p["has_tables"])
    fig_pgs   = sum(1 for p in pages if p["has_figures"])
    col_counts = [estimate_columns(p["words"]) for p in content_pages]
    avg_cols  = sum(col_counts) / max(len(col_counts), 1)

    t_ratio = table_pgs / total_all
    f_ratio = fig_pgs   / total_all

    if t_ratio >= TABLE_RATIO and f_ratio >= FIG_RATIO:
        layout = "mixed"
    elif t_ratio >= TABLE_RATIO:
        layout = "table_heavy"
    elif f_ratio >= FIG_RATIO:
        layout = "figure_heavy"
    elif avg_cols >= MULTI_COL_THR:
        layout = "multi_column"
    else:
        layout = "single_column"

    return layout, table_pgs, fig_pgs, avg_cols

# ── Step 4: Detect domain hint ────────────────────────────────────────
def detect_domain(full_text):
    if not full_text.strip():
        return "general", {}
    text_lower = full_text.lower()
    scores = {}
    for domain, keywords in DOMAIN.items():
        scores[domain] = sum(1 for kw in keywords if kw.lower() in text_lower)
    best = max(scores, key=lambda d: scores[d])
    return ("general" if scores[best] == 0 else best), scores

# ── Step 5: Estimate extraction cost ─────────────────────────────────
def estimate_cost(origin, layout):
    if origin == "scanned_image":
        return "needs_vision_model", "C"
    if origin == "form_fillable":
        return "needs_layout_model", "B"
    if layout in ("multi_column", "table_heavy", "figure_heavy", "mixed"):
        return "needs_layout_model", "B"
    if origin == "mixed":
        return "needs_layout_model", "B"
    return "fast_text_sufficient", "A"

# ── Main triage function ──────────────────────────────────────────────
def run_triage(pdf_path):
    pdf_path = Path(pdf_path)

    header(f"PHASE 0 — TRIAGE AGENT")
    print(f"  {C.GREY}Document: {pdf_path.name}{C.RESET}")

    t_start = time.time()

    # Validate
    if not pdf_path.exists():
        print(f"  {C.RED}ERROR: File not found: {pdf_path}{C.RESET}")
        return None

    section("Opening document with pdfplumber")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                print(f"  {C.RED}ERROR: Document has 0 pages{C.RESET}")
                return None
            info("Total pages", total_pages)
            pages = extract_pages(pdf)
    except Exception as e:
        err = str(e).lower()
        if "password" in err or "encrypted" in err:
            warn("Status", "PASSWORD PROTECTED — cannot open")
        else:
            print(f"  {C.RED}ERROR: {e}{C.RESET}")
        return None

    corrupted = [p for p in pages if p["error"]]
    if corrupted:
        warn("Corrupted pages", f"{len(corrupted)} pages had errors")

    # Measurements
    section("Step 1 — Page Measurements")
    avg_chars = sum(p["char_count"] for p in pages) / total_pages
    avg_img   = sum(p["image_area_ratio"] for p in pages) / total_pages
    total_tbl = sum(p["table_count"] for p in pages)
    total_fig = sum(p["figure_count"] for p in pages)
    info("Avg chars/page",    f"{avg_chars:.0f}")
    info("Avg image ratio",   f"{avg_img:.3f}")
    info("Total tables found",total_tbl)
    info("Total figures found",total_fig)

    # Show sample of pages
    print(f"\n    {C.GREY}Page sample (first 10):{C.RESET}")
    print(f"    {'Page':>5} {'Chars':>7} {'ImgRatio':>10} {'Tables':>7} {'Verdict':>8}")
    print(f"    {'-'*45}")
    for p in pages[:10]:
        verdict_col = C.RED if (p["char_count"] < MIN_CHARS and p["image_area_ratio"] > MAX_IMG_RATIO) else C.GREEN
        print(f"    {p['page_num']:>5} {p['char_count']:>7} {p['image_area_ratio']:>10.3f} "
              f"{'YES' if p['has_tables'] else 'no':>7} "
              f"{verdict_col}{'SCAN' if (p['char_count'] < MIN_CHARS and p['image_area_ratio'] > MAX_IMG_RATIO) else 'DIGI':>8}{C.RESET}")
    if total_pages > 10:
        print(f"    {C.GREY}... {total_pages - 10} more pages{C.RESET}")

    # Origin detection
    section("Step 2 — Origin Type Detection")
    origin, digital, scanned, scan_ratio, _ = detect_origin(pages)
    info("Digital pages",   digital)
    info("Scanned pages",   scanned)
    info("Scan ratio",      f"{scan_ratio:.1%}  (threshold: {SCAN_THRESHOLD:.0%})")
    decision("origin_type", origin.upper(),
             C.RED if origin == "scanned_image" else C.GREEN if origin == "native_digital" else C.YELLOW)

    # Layout detection
    section("Step 3 — Layout Complexity Detection")
    layout, tbl_pgs, fig_pgs, avg_cols = detect_layout(pages)
    info("Pages with tables",  f"{tbl_pgs}  ({tbl_pgs/total_pages:.1%}  threshold: {TABLE_RATIO:.0%})")
    info("Pages with figures", f"{fig_pgs}  ({fig_pgs/total_pages:.1%}  threshold: {FIG_RATIO:.0%})")
    info("Avg column count",   f"{avg_cols:.2f}  (threshold: {MULTI_COL_THR})")
    decision("layout_complexity", layout.upper())

    # Domain detection
    section("Step 4 — Domain Hint Detection")
    full_text = " ".join(p["text"] for p in pages)
    domain, scores = detect_domain(full_text)
    for d, s in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * min(s, 30)
        marker = f" {C.PURPLE}◀ best match{C.RESET}" if d == domain else ""
        print(f"    {d:<12} {C.CYAN}{bar}{C.RESET} ({s}){marker}")
    decision("domain_hint", domain.upper(), C.PURPLE)

    # Extraction cost
    section("Step 5 — Extraction Cost Estimation")
    cost, strategy = estimate_cost(origin, layout)
    cost_color = C.GREEN if strategy == "A" else C.CYAN if strategy == "B" else C.YELLOW
    decision("estimated_cost",  cost.upper())
    decision("initial_strategy", f"Strategy {strategy}", cost_color)

    # Build profile dict
    elapsed = time.time() - t_start
    profile = {
        "doc_id"                  : pdf_path.stem.lower().replace(" ", "_"),
        "filename"                : pdf_path.name,
        "total_pages"             : total_pages,
        "origin_type"             : origin,
        "layout_complexity"       : layout,
        "domain_hint"             : domain,
        "estimated_extraction_cost": cost,
        "initial_strategy"        : f"Strategy {strategy}",
        "avg_chars_per_page"      : round(avg_chars, 2),
        "avg_image_area_ratio"    : round(avg_img, 4),
        "digital_page_count"      : digital,
        "scanned_page_count"      : scanned,
        "table_page_count"        : tbl_pgs,
        "figure_page_count"       : fig_pgs,
        "domain_scores"           : scores,
        "triage_time_seconds"     : round(elapsed, 3),
    }

    # Save profile
    profiles_dir = Path(__file__).parent / ".refinery" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / f"{profile['doc_id']}.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    # Summary
    header("TRIAGE COMPLETE")
    ok("Document",          pdf_path.name)
    ok("Origin Type",       origin.upper())
    ok("Layout",            layout.upper())
    ok("Domain",            domain.upper())
    ok("Strategy Selected", f"Strategy {strategy}  ({cost})")
    ok("Profile saved to",  str(profile_path))
    ok("Triage time",       f"{elapsed:.2f}s")
    print()

    return profile


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run on a specific file
        result = run_triage(sys.argv[1])
    else:
        # Look for PDFs in data/corpus/
        corpus_dir = Path(__file__).parent / "data"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        pdfs = list(corpus_dir.glob("*.pdf"))

        if not pdfs:
            print(f"""
{C.BOLD}{C.YELLOW}No PDFs found.{C.RESET}

To test Phase 0, either:

  Option 1 — Pass a PDF directly:
    {C.CYAN}python3 run_phase0.py /path/to/your/document.pdf{C.RESET}

  Option 2 — Drop PDFs into the corpus folder and run again:
    {C.CYAN}mkdir -p data/corpus/
    cp your_document.pdf data/corpus/
    python3 run_phase0.py{C.RESET}

The triage agent will classify:
  - Origin type    (native_digital / scanned_image / mixed)
  - Layout         (single_column / table_heavy / multi_column / etc)
  - Domain         (financial / legal / technical / general)
  - Strategy       (A / B / C)
""")
        else:
            print(f"\n{C.BOLD}Found {len(pdfs)} PDF(s) in data/corpus/{C.RESET}")
            profiles = []
            for pdf in pdfs:
                result = run_triage(pdf)
                if result:
                    profiles.append(result)

            if len(profiles) > 1:
                print(f"\n{C.BOLD}{C.NAVY}{'='*60}")
                print(f"  CORPUS SUMMARY — {len(profiles)} documents{C.RESET}")
                print(f"{C.BOLD}{C.NAVY}{'='*60}{C.RESET}")
                print(f"  {'Document':<35} {'Origin':<16} {'Layout':<16} {'Strategy'}")
                print(f"  {'-'*80}")
                for p in profiles:
                    print(f"  {p['filename']:<35} {p['origin_type']:<16} "
                          f"{p['layout_complexity']:<16} {p['initial_strategy']}")
