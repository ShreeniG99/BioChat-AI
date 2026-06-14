"""
BioChatAI – Query Enhancement Pipeline Infographic
Clinical Cartography · Precision visual documentation
"""
from PIL import Image, ImageDraw, ImageFont
import os

# ── PATHS ────────────────────────────────────────────────────────────────────
FONTS = r"C:\Users\shree\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\f745a90e-e461-43ac-91fe-1a92c06ef178\a7144673-07b7-4651-a8ba-912ef5d6e75d\skills\canvas-design\canvas-fonts"
OUT   = r"C:\Dev\BioChatAI_final\assets\query_enhancement_pipeline.png"

def F(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)

# ── PALETTE ──────────────────────────────────────────────────────────────────
BG        = "#07101C"
PANEL     = "#0C1828"
HEADER_BG = "#04090F"
GRID      = "#0A1520"
TEXT      = "#C5D8E8"
MUTED     = "#3D5568"
CODE_CLR  = "#52D8CC"
FORMULA   = "#7EFFD4"
ARROW_CLR = "#2A7090"
ARROW_TIP = "#3ABED0"

# Group accent → (accent_hex, panel_hex)
GRP = {
    "io":       ("#4A9EFF", "#081830"),
    "textproc": ("#AA72FF", "#100828"),
    "analysis": ("#FF7A50", "#1C0A06"),
    "search":   ("#3EC88A", "#061A10"),
    "output":   ("#FFD050", "#1A1200"),
}

W       = 2400
MARGIN  = 90
INNER_W = W - 2 * MARGIN
PAD     = 30
ARROW_H = 56

# ── STAGE DATA ───────────────────────────────────────────────────────────────
STAGES = [
    dict(
        num="01", group="io",
        title="RAW QUERY INPUT",
        desc='User submits a natural language biomedical question via the frontend chat interface.',
        details=[
            'Example:  "What is the mechanism of metformin in treating type 2 diabetes?"',
            'Validated:  max 1,000 characters · stripped of leading/trailing whitespace',
            'Transmitted:  POST /api/query  →  { query: str, max_results: int, user_id: int? }',
        ],
        formula=None,
        tags=["USER INPUT", "FRONTEND → FASTAPI", "HTTP POST"],
    ),
    dict(
        num="02", group="textproc",
        title="UNICODE & ENCODING NORMALIZATION",
        desc='BiomedicalTextProcessor.normalize_biomedical_text() applies 4 sequential cleanup passes.',
        details=[
            'unicodedata.normalize("NFKC", text)       — compatibility decomposition + composition',
            'Encoding fixes:  Î± → α    Î² → β    Î³ → γ    Â± → ±    Âµ → µ',
            're.sub(r"[—–−]", "-", text)              — en/em-dash unification',
            're.sub(r"\\s+", " ", text).strip()        — whitespace collapse',
        ],
        formula='normalize_biomedical_text(query)   →   clean_query: str',
        tags=["UNICODE NFKC", "OCR ARTIFACT FIXES", "HELPERS.PY · BiomedicalTextProcessor"],
    ),
    dict(
        num="03", group="textproc",
        title="MEDICAL ABBREVIATION EXPANSION",
        desc='100+ medical abbreviations and 8 drug-class names expanded using word-boundary regex substitution.',
        details=[
            'HTN    → hypertension                    MI     → myocardial infarction',
            'T2DM   → type 2 diabetes mellitus        COPD   → chronic obstructive pulmonary disease',
            'CAR-T  → chimeric antigen receptor T cell',
            'CRISPR → clustered regularly interspaced short palindromic repeats',
            'SSRI   → selective serotonin reuptake inhibitor    ARB → angiotensin receptor blocker',
            'Pattern applied:  re.sub(r"\\b{abbrev}\\b", expansion, text, re.IGNORECASE)',
        ],
        formula='expand_abbreviations(clean_query)   →   expanded_query: str',
        tags=["100+ ABBREVIATIONS", "8 DRUG CLASS TERMS", "WORD BOUNDARY REGEX"],
    ),
    dict(
        num="04", group="analysis",
        title="INTENT DETECTION",
        desc='Query classified into 1 of 8 intent types. Confidence = fraction of the type\'s regex patterns matched.',
        details=[
            '① mechanism      — "how does"  "pathway"  "process"  "function"  "works"',
            '② treatment      — "treatment"  "therapy"  "drug"  "medication"  "cure"  "therapeutic"',
            '③ diagnosis      — "diagnose"  "symptoms"  "signs"  "detect"  "screening"',
            '④ research       — "study"  "research"  "trial"  "evidence"  "findings"  "clinical trial"',
            '⑤ comparison     — "versus"  "vs"  "compare"  "difference"  "better"  "alternative"',
            '⑥ definition     — "what is"  "define"  "meaning"  "explain"  "description"',
            '⑦ epidemiology   — "prevalence"  "incidence"  "epidemiology"  "statistics"  "mortality"',
            '⑧ adverse_effects— "side effects"  "adverse"  "toxicity"  "contraindication"  "safety"',
        ],
        formula='intent_confidence = Σ(regex_matches_for_type) / len(patterns_for_type)   ∈ [0, 1]',
        tags=["8 INTENT TYPES", "MULTI-PATTERN REGEX", "CONFIDENCE SCORED", "RAG_SERVICE.PY"],
    ),
    dict(
        num="05", group="analysis",
        title="BIOMEDICAL ENTITY EXTRACTION",
        desc='5 entity-type categories identified via keyword-pattern lists. Each yields a normalized score.',
        details=[
            'disease    — cancer  diabetes  hypertension  alzheimer  covid  syndrome  disorder  carcinoma',
            'drug       — drug  medication  compound  molecule  inhibitor  agonist  antagonist  therapy',
            'gene       — gene  protein  enzyme  receptor  antibody  biomarker  mutation',
            'procedure  — surgery  procedure  imaging  test  assay  biopsy  endoscopy',
            'anatomy    — heart  brain  liver  kidney  lung  blood  cell  tissue',
        ],
        formula='entity_score[type] = |matched_patterns[type]| / |all_patterns[type]|   ∈ [0, 1]',
        tags=["5 ENTITY TYPES", "KEYWORD PATTERN LISTS", "SCORED 0 → 1"],
    ),
    dict(
        num="06", group="analysis",
        title="COMPLEXITY SCORING",
        desc='4-factor weighted formula; result normalized to [0.0, 1.0] and attached to the query analysis object.',
        details=[
            'factor₁  =  len(query.split()) / 25.0                      — word-count contribution',
            'factor₂  =  len(entities_found) / 10.0                     — entity density contribution',
            'factor₃  =  intent_confidence                               — intent-clarity contribution',
            'factor₄  =  1.0  if {"clinical","trial","study","research"} ∩ words  else  0.0',
        ],
        formula='complexity_score = min(1.0,  (factor₁ + factor₂ + factor₃ + factor₄) / 4)   ∈ [0.0, 1.0]',
        tags=["4 FACTORS", "NORMALIZED RANGE [0, 1]", "ARITHMETIC MEAN"],
    ),
    dict(
        num="07", group="analysis",
        title="QUESTION TYPE CLASSIFICATION",
        desc='Rhetorical form of the query mapped to 1 of 4 types; governs the intent-specific prompt template selected for generation.',
        details=[
            '"how" / "why" / "mechanism"          → explanatory      (step-by-step causal explanation)',
            '"compare" / "versus" / "difference"  → comparative      (structured A-vs-B analysis)',
            '"latest" / "recent" / "new"          → current_research  (prioritise 2020–2024 literature)',
            'default                               → factual           (direct definition or fact retrieval)',
        ],
        formula='question_type  ∈  { factual,  explanatory,  comparative,  current_research }',
        tags=["4 QUESTION TYPES", "PROMPT TEMPLATE SELECTOR", "DOWNSTREAM: BIOGPT PROMPT"],
    ),
    dict(
        num="08", group="search",
        title="PUBMED QUERY ENHANCEMENT",
        desc='Cleaned, intent-tagged query augmented with MeSH date filters and study-type qualifiers before submission to NCBI.',
        details=[
            'Date filter:    AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
            'treatment/therapy  →  AND (clinical trial[pt] OR randomized controlled trial[pt])',
            'mechanism/pathway  →  AND (review[pt] OR research support[pt])',
            'Fallback (no results):  strip stop-words → keep top-5 content words → retry',
        ],
        formula='_enhance_pubmed_query(expanded_query)   →   pubmed_query: str',
        tags=["MESH TERMS", "DATE FILTER 2020–2024", "STUDY-TYPE FILTER", "STOP-WORD FALLBACK"],
    ),
    dict(
        num="09", group="output",
        title="STRUCTURED QUERY ANALYSIS OUTPUT",
        desc='The full pipeline produces a typed dict consumed by retrieval, generation, and confidence-scoring stages downstream.',
        details=[
            'intent:              "mechanism"',
            'intent_confidence:    0.67',
            'entities:            ["disease", "drug"]',
            'entity_scores:       { "disease": 0.40,  "drug": 0.33 }',
            'complexity_score:     0.72',
            'question_type:       "explanatory"',
            'biomedical_terms:     4',
            'enhanced_query:      "metformin mechanism AND review[pt] AND (2020:2024[pdat])"',
        ],
        formula=None,
        tags=["DICT OUTPUT", "→ HYBRID VECTOR SEARCH", "→ BIOGPT PROMPT", "→ CONFIDENCE SCORER"],
    ),
]

# ── FONT REGISTRY ─────────────────────────────────────────────────────────────
fonts = {
    "title_big":  F("Jura-Medium.ttf",          80),
    "subtitle":   F("Jura-Light.ttf",            24),
    "num":        F("Outfit-Bold.ttf",            50),
    "stage_name": F("Outfit-Bold.ttf",            36),
    "desc":       F("InstrumentSans-Regular.ttf", 24),
    "detail":     F("JetBrainsMono-Regular.ttf",  20),
    "formula":    F("JetBrainsMono-Bold.ttf",     20),
    "tag":        F("Jura-Medium.ttf",            16),
    "footer":     F("JetBrainsMono-Regular.ttf",  16),
    "legend":     F("Jura-Light.ttf",             15),
    "grp_label":  F("Outfit-Bold.ttf",            12),
}

# ── MEASURE PASS ─────────────────────────────────────────────────────────────
def stage_h(stage):
    h  = PAD            # top
    h += 62             # number circle / title row
    h += 8              # gap
    h += 32             # desc line
    h += 14             # gap
    h += len(stage["details"]) * 32
    if stage["formula"]:
        h += 12 + 38    # formula box
    h += 18 + 32        # tag row
    h += PAD            # bottom
    return h

# ── RENDER ────────────────────────────────────────────────────────────────────
def render():
    HEADER_H = 160
    FOOTER_H = 90

    s_heights = [stage_h(s) for s in STAGES]
    total_h   = HEADER_H + sum(s_heights) + ARROW_H * (len(STAGES)-1) + FOOTER_H + 50

    img  = Image.new("RGB", (W, total_h), BG)
    draw = ImageDraw.Draw(img)

    # background dot grid (more subtle than lines)
    for gx in range(80, W, 80):
        for gy in range(80, total_h, 80):
            draw.ellipse([gx-1, gy-1, gx+1, gy+1], fill="#0C1826")

    # ── HEADER ───────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (W, HEADER_H)], fill=HEADER_BG)
    draw.rectangle([(0, HEADER_H-3), (W, HEADER_H)], fill="#142035")

    draw.text((MARGIN, 22),  "QUERY ENHANCEMENT PIPELINE", font=fonts["title_big"],  fill="#44DDD0")
    # underline accent
    title_bb = draw.textbbox((0,0), "QUERY ENHANCEMENT PIPELINE", font=fonts["title_big"])
    draw.rectangle([(MARGIN, 112), (MARGIN + title_bb[2] + 2, 115)], fill="#44DDD0"+"55")
    draw.text((MARGIN, 122), "BioChatAI  ·  rag_service.py  +  utils/helpers.py  ·  BiomedicalTextProcessor",
              font=fonts["subtitle"], fill="#2A5068")

    # Legend (top-right)
    legend = [
        ("INPUT / OUTPUT",      GRP["io"][0]),
        ("TEXT PROCESSING",     GRP["textproc"][0]),
        ("QUERY ANALYSIS",      GRP["analysis"][0]),
        ("SEARCH ENHANCEMENT",  GRP["search"][0]),
        ("PIPELINE OUTPUT",     GRP["output"][0]),
    ]
    lx = W - MARGIN
    for lname, lcol in reversed(legend):
        lbb = draw.textbbox((0,0), lname, font=fonts["legend"])
        lw  = lbb[2] - lbb[0] + 20
        lx -= lw + 8
        draw.rounded_rectangle([lx, 38, lx+lw, 60], radius=4,
                                fill=lcol+"1A", outline=lcol+"80", width=1)
        draw.text((lx+8, 41), lname, font=fonts["legend"], fill=lcol)

    # ── STAGES ───────────────────────────────────────────────────────────────
    y = HEADER_H + 22

    for idx, stage in enumerate(STAGES):
        accent, s_bg = GRP[stage["group"]]
        sh = s_heights[idx]
        bx0, bx1 = MARGIN, W - MARGIN

        # drop shadow
        draw.rounded_rectangle([bx0+5, y+5, bx1+5, y+sh+5], radius=10, fill="#020508")

        # panel
        draw.rounded_rectangle([bx0, y, bx1, y+sh], radius=10, fill=s_bg,
                                outline=accent+"55", width=2)

        # left accent bar
        draw.rounded_rectangle([bx0, y, bx0+10, y+sh], radius=10, fill=accent)

        # ── number circle ──
        cx, cy = bx0 + 64, y + 44
        draw.ellipse([cx-26, cy-26, cx+26, cy+26],
                     fill=accent+"1E", outline=accent+"70", width=2)
        nbb = draw.textbbox((0,0), stage["num"], font=fonts["num"])
        nw, nh = nbb[2]-nbb[0], nbb[3]-nbb[1]
        draw.text((cx - nw//2, cy - nh//2 - 1), stage["num"], font=fonts["num"], fill=accent)

        # ── title ──
        tx = bx0 + 112
        draw.text((tx, y + PAD),      stage["title"], font=fonts["stage_name"], fill="#E8F4FF")
        draw.text((tx, y + PAD + 46), stage["desc"],  font=fonts["desc"],       fill="#5A7A90")

        # ── separator line under desc ──
        sep_y = y + PAD + 46 + 32 + 6
        draw.line([(tx, sep_y), (bx1 - 24, sep_y)], fill=accent+"22", width=1)

        # ── details ──
        dy = sep_y + 10
        for line in stage["details"]:
            draw.ellipse([tx, dy+9, tx+8, dy+17], fill=accent+"70")
            draw.text((tx+16, dy), line, font=fonts["detail"], fill=CODE_CLR)
            dy += 32

        # ── formula ──
        if stage["formula"]:
            fy = dy + 10
            draw.rounded_rectangle([tx-6, fy, bx1-24, fy+34],
                                   radius=5, fill="#030B14", outline=accent+"33", width=1)
            # left accent stripe on formula box
            draw.rounded_rectangle([tx-6, fy, tx-2, fy+34], radius=5, fill=accent+"60")
            draw.text((tx+8, fy+7), stage["formula"], font=fonts["formula"], fill=FORMULA)
            dy = fy + 44

        # ── tags ──
        tag_y = dy + 14
        tx2   = tx
        for tag in stage["tags"]:
            tbb = draw.textbbox((0,0), tag, font=fonts["tag"])
            tw  = tbb[2] - tbb[0] + 20
            draw.rounded_rectangle([tx2, tag_y, tx2+tw, tag_y+26],
                                   radius=5, fill=accent+"18", outline=accent+"55", width=1)
            draw.text((tx2+10, tag_y+5), tag, font=fonts["tag"], fill=accent)
            tx2 += tw + 10

        y += sh

        # ── arrow connector ──
        if idx < len(STAGES)-1:
            ax      = W // 2
            ay0     = y + 8
            ay1     = y + ARROW_H - 16
            draw.line([(ax, ay0), (ax, ay1)], fill=ARROW_TIP, width=3)
            draw.polygon([(ax-10, ay1), (ax+10, ay1), (ax, ay1+14)], fill=ARROW_TIP)
            y += ARROW_H

    # ── FOOTER ────────────────────────────────────────────────────────────────
    footer_y = y + 30
    draw.rectangle([(0, footer_y), (W, footer_y+FOOTER_H)], fill=HEADER_BG)
    draw.rectangle([(0, footer_y), (W, footer_y+2)], fill="#1A3850")

    math_line = (
        "complexity = min(1.0, (|words|/25 + |entities|/10 + intent_conf + clinical_flag) / 4)"
        "   |   "
        "intent_confidence = matched / total_patterns"
        "   |   "
        "entity_score[t] = |matched[t]| / |patterns[t]|"
    )
    draw.text((MARGIN, footer_y + 16), math_line, font=fonts["footer"], fill="#28607A")
    # right-side tag
    sig = "BioChatAI  Query Enhancement Pipeline  v1.0"
    sbb = draw.textbbox((0,0), sig, font=fonts["footer"])
    draw.text((W - MARGIN - (sbb[2]-sbb[0]), footer_y + 52), sig, font=fonts["footer"], fill="#1A4060")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, dpi=(144, 144))
    print(f"Saved: {OUT}  [{W} x {total_h} px]")

render()
