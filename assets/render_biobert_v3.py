"""
BioChatAI – BioBERT Embedding Pipeline  v3
Split-panel design:
  LEFT  (70%)  — pipeline stage cards (2-column inner layout)
  RIGHT (30%)  — live data-shape tracker
Dark navy palette, distinct from previous renders.
"""
from PIL import Image, ImageDraw, ImageFont
import os, math

FONTS = r"C:\Users\shree\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\f745a90e-e461-43ac-91fe-1a92c06ef178\a7144673-07b7-4651-a8ba-912ef5d6e75d\skills\canvas-design\canvas-fonts"
OUT   = r"C:\Dev\BioChatAI_final\assets\biobert_embedding_pipeline_v3.png"

def F(n, s): return ImageFont.truetype(os.path.join(FONTS, n), s)

# ── CANVAS ──────────────────────────────────────────────────────────────────
W        = 2800
L_W      = 1860   # left pipeline panel width
R_W      = W - L_W - 4   # right shape-tracker panel
MARGIN   = 60
PAD      = 26
ARROW_H  = 50

# ── PALETTE ─────────────────────────────────────────────────────────────────
BG         = "#06101C"
HDR        = "#030A14"
PANEL      = "#0B1928"
DIVIDER    = "#0F2038"
WHITE      = "#D0E8F4"
MUTED      = "#304858"
DIM_MUTED  = "#1E3448"
CODE       = "#4ED8C8"
FORM       = "#7EFFD4"
ARROW_C    = "#3AC0D8"
TRACKER_BG = "#080F1A"
TRACKER_BD = "#0F2038"

# Stage group colours  (accent, dark-bg)
GRP = {
    "assemble": ("#00C9A7", "#051812"),
    "preproc":  ("#9B72FF", "#0D0820"),
    "chunk":    ("#3EC88A", "#051A10"),
    "cache":    ("#FF9F43", "#1A0E05"),
    "tokenize": ("#FF6B9D", "#1A0610"),
    "model":    ("#FF6060", "#1A0505"),
    "math":     ("#F7DC6F", "#1A1805"),
    "cwrite":   ("#FF9F43", "#1A0E05"),
    "store":    ("#5DADE2", "#05101A"),
}

# ── FONTS ───────────────────────────────────────────────────────────────────
fnt = {
    "title":   F("Jura-Medium.ttf",           72),
    "sub":     F("Jura-Light.ttf",             20),
    "num":     F("Outfit-Bold.ttf",            44),
    "sname":   F("Outfit-Bold.ttf",            30),
    "desc":    F("InstrumentSans-Regular.ttf", 20),
    "code":    F("JetBrainsMono-Regular.ttf",  17),
    "bold_c":  F("JetBrainsMono-Bold.ttf",     17),
    "tag":     F("Jura-Medium.ttf",            13),
    "tracker": F("Outfit-Bold.ttf",            17),
    "shape":   F("JetBrainsMono-Bold.ttf",     18),
    "dtype":   F("JetBrainsMono-Regular.ttf",  14),
    "footer":  F("JetBrainsMono-Regular.ttf",  14),
    "legend":  F("Jura-Light.ttf",             13),
    "math_lg": F("JetBrainsMono-Bold.ttf",     20),
}

# ── STAGE DATA ───────────────────────────────────────────────────────────────
# Each stage:  num, group, title, left_lines (description), right_lines (code/math),
#              formula, tags, shape_out (for tracker), dtype_out, shape_vis (short)

STAGES = [
    dict(
        num="01", group="assemble", title="ARTICLE TEXT ASSEMBLY",
        left=[
            "Three document sections pulled from",
            "the article dict and joined into one",
            "content string before processing.",
            "",
            "title     — article headline",
            "abstract  — primary content source",
            "full_text — PMC body (if available)",
        ],
        right=[
            'title    = doc.get("title", "")',
            'abstract = doc.get("abstract", "")',
            'full_text= doc.get("full_text", "")',
            "",
            'content  =',
            '  "Title: {t}',
            '   Abstract: {a}',
            '   Full Text: {f}"',
        ],
        formula=None,
        tags=["embed_document()","DICT INPUT","3-SECTION JOIN"],
        shape_out="str",   dtype_out="Python str", shape_vis="TEXT",
    ),
    dict(
        num="02", group="preproc", title="BIOMEDICAL ABBREVIATION EXPANSION",
        left=[
            "_preprocess_biomedical_text() scans",
            "every word, strips punctuation,",
            "and replaces 13 medical abbreviations",
            "before tokenization.",
            "",
            "Why: BioBERT vocab contains both",
            "abbreviated and expanded forms —",
            "expanding ensures richer token overlap.",
        ],
        right=[
            "words = text.split()",
            "for i, word in enumerate(words):",
            "  w = word.strip('.,;:()[]')",
            "  if w in abbreviations:",
            "    words[i] = word.replace(",
            "      w, abbreviations[w])",
            "",
            "HTN→hypertension  MI→myocard.inf.",
            "COVID-19→coronavirus disease 2019",
            "mRNA→messenger RNA  DNA→deoxyribo..",
        ],
        formula="_preprocess_biomedical_text(text)  →  expanded_text: str",
        tags=["13 ABBREVIATIONS","WORD-LEVEL SCAN","PUNCT-AWARE"],
        shape_out="str", dtype_out="Python str", shape_vis="TEXT+",
    ),
    dict(
        num="03", group="cache", title="TWO-LEVEL CACHE LOOKUP",
        left=[
            "Before any model compute, both cache",
            "levels are checked in order.",
            "A HIT at either level skips ALL",
            "stages 04–07 (no inference needed).",
            "",
            "L1 — memory_cache dict (thread-safe)",
            "L2 — disk  .pkl  (30-day TTL)",
            "",
            "CACHE HIT  → return v̂ immediately",
            "CACHE MISS → continue to chunking",
        ],
        right=[
            "# Cache key",
            "key = hashlib.md5(",
            "  text.encode('utf-8')",
            ").hexdigest()       # 32-char hex",
            "",
            "# L1 check",
            "with self.lock:",
            "  if key in memory_cache:",
            "    return memory_cache[key]",
            "",
            "# L2 check",
            "if os.path.exists(cache_path):",
            "  mod = os.path.getmtime(path)",
            "  if now - mod < 30*24*3600:",
            "    return pickle.load(f)",
        ],
        formula="cache_key = MD5(cleaned_text).hexdigest()   hit_rate = hits / (hits + misses)",
        tags=["L1 MEMORY 10K MAX","L2 DISK 30d TTL","THREAD-SAFE LOCK","HIT→SKIP MODEL"],
        shape_out="cache_key",dtype_out="str (32-char hex)", shape_vis="KEY",
    ),
    dict(
        num="04", group="chunk", title="OVERLAPPING DOCUMENT CHUNKING",
        left=[
            "Long documents split into overlapping",
            "400-word windows so each chunk fits",
            "within BioBERT's 512-token limit.",
            "80-word overlap preserves sentence",
            "context at chunk boundaries.",
            "",
            "chunk[i] = words[i·320 : i·320+400]",
            "stride   = chunk_size − overlap = 320",
        ],
        right=[
            "chunk_size = 400  # words",
            "overlap    =  80  # words",
            "stride     = 320  # words",
            "",
            "# num chunks formula",
            "if |words| <= 400:",
            "  chunks = [full_text]",
            "else:",
            "  N = ceil((|w|-400)/320)+1",
            "",
            "# batch processing",
            "embeddings = await embed_batch(",
            "  chunks, batch_size=8)",
        ],
        formula="chunk[i] = words[i×320 : i×320+400]    N = ⌈(|words|−400)/320⌉ + 1",
        tags=["chunk_size=400","overlap=80","stride=320","batch_size=8"],
        shape_out="List[str]", dtype_out="List[str]  len=N", shape_vis="[str×N]",
    ),
    dict(
        num="05", group="tokenize", title="BERT WORDPIECE TOKENIZATION",
        left=[
            "AutoTokenizer converts each chunk",
            "into sub-word token IDs using the",
            "BioBERT WordPiece vocabulary",
            "(28,996 tokens).",
            "",
            "[CLS] prepended  (id = 101)",
            "[SEP] appended   (id = 102)",
            "Texts > 512 tokens truncated RIGHT.",
            "Shorter texts padded to batch length.",
        ],
        right=[
            'tok = AutoTokenizer.from_pretrained(',
            '  "dmis-lab/biobert-v1.1")',
            "",
            "inputs = tok(",
            "  text,",
            "  return_tensors = 'pt',",
            "  truncation     = True,",
            "  padding        = True,",
            "  max_length     = 512",
            ")",
            "# outputs three tensors [1, T]:",
            "# input_ids      — token integer IDs",
            "# attention_mask — 1=real  0=pad",
            "# token_type_ids — 0 (single sent.)",
        ],
        formula="inputs → { input_ids [1,T],  attention_mask [1,T],  token_type_ids [1,T] }   T ≤ 512",
        tags=["vocab=28,996","WordPiece","MAX 512 TOKENS","[CLS]…[SEP]"],
        shape_out="[1, T≤512]", dtype_out="torch.LongTensor ×3", shape_vis="[1,T,3]",
    ),
    dict(
        num="06", group="model", title="BIOBERT TRANSFORMER FORWARD PASS",
        left=[
            "Tokenized input fed through BioBERT:",
            "BERT-base pretrained on PubMed",
            "abstracts + PMC full-text articles.",
            "",
            "Architecture:",
            "  12 transformer encoder layers",
            "  12 self-attention heads per layer",
            "  hidden dimension = 768",
            "",
            "torch.no_grad() disables autograd —",
            "~40% less memory, faster inference.",
        ],
        right=[
            'model = AutoModel.from_pretrained(',
            '  "dmis-lab/biobert-v1.1")',
            "model.to(device)   # cuda / cpu",
            "model.eval()       # dropout OFF",
            "",
            "with torch.no_grad():",
            "  outputs = model(**inputs)",
            "",
            "# outputs.last_hidden_state",
            "#   shape: [1, T, 768]",
            "# All T token representations",
            "# from the final encoder layer",
        ],
        formula="model(**inputs) → last_hidden_state  shape: [batch=1, seq_len=T, hidden=768]",
        tags=["12 LAYERS","12 HEADS","hidden=768","torch.no_grad()","BERT-BASE"],
        shape_out="[1, T, 768]", dtype_out="torch.FloatTensor", shape_vis="[1,T,768]",
    ),
    dict(
        num="07", group="model", title="[CLS] TOKEN EXTRACTION",
        left=[
            "Only the [CLS] token (position 0)",
            "is used as the document embedding.",
            "",
            "The [CLS] token attends to ALL other",
            "tokens across all 12 layers, making",
            "it the sequence-level summary vector.",
            "",
            ".cpu()   — move off GPU to system RAM",
            ".numpy() — convert Tensor → ndarray",
            ".squeeze()— drop batch dim [1,768]→[768]",
        ],
        right=[
            "# Extract CLS token (pos 0)",
            "embedding = (",
            "  outputs",
            "  .last_hidden_state",
            "  [:, 0, :]       # [1, 768]",
            "  .cpu()          # off GPU",
            "  .numpy()        # → ndarray",
            "  .squeeze()      # [768,]",
            ")",
            "",
            "# Shape progression:",
            "# [1, T, 768]  — full hidden state",
            "# [1, 768]     — after [:, 0, :]",
            "# [768,]       — after .squeeze()",
        ],
        formula="last_hidden_state[:, 0, :]  →  [1, 768]  →  .squeeze()  →  ndarray [768,]",
        tags=["CLS = token 0","SEQUENCE POOLING","[1,T,768]→[768,]","float32"],
        shape_out="[768,]", dtype_out="np.ndarray float32", shape_vis="[768,]",
    ),
    dict(
        num="08", group="math", title="L2 NORMALIZATION TO UNIT HYPERSPHERE",
        left=[
            "The raw 768-dim CLS vector is",
            "L2-normalized so its magnitude = 1.",
            "This projects all embeddings onto",
            "the surface of a unit hypersphere.",
            "",
            "KEY PROPERTY:",
            "For unit vectors, cosine similarity",
            "equals the dot product:",
            "  cos(v̂, ŵ) = v̂ · ŵ  ∈ [-1, 1]",
            "FAISS IndexFlatIP exploits this.",
        ],
        right=[
            "# Step 1: compute L2 norm",
            "# ||v||₂ = sqrt(Σᵢ vᵢ²)",
            "",
            "# Step 2: normalize",
            "embedding = embedding /",
            "  np.linalg.norm(embedding)",
            "",
            "# Verification: ||v̂||₂ = 1.0",
            "",
            "# FAISS re-normalizes as batch:",
            "faiss.normalize_L2(",
            "  embeddings_array)  # float32",
            "",
            "# cos(v̂,ŵ) = v̂·ŵ (dot product)",
        ],
        formula="v̂ = v / ||v||₂    ||v||₂ = sqrt(Σᵢ₌₁⁷⁶⁸ vᵢ²)    cos(v̂,ŵ) = v̂·ŵ ∈ [-1, 1]",
        tags=["||v̂||₂=1.0","UNIT HYPERSPHERE","cos=dot(unit vecs)","np.linalg.norm()"],
        shape_out="[768,] ||v̂||=1", dtype_out="np.ndarray float32", shape_vis="v̂[768]",
    ),
    dict(
        num="09", group="cwrite", title="CACHE WRITE-BACK",
        left=[
            "Freshly computed embedding stored",
            "in both cache layers immediately.",
            "",
            "L1 FIFO eviction policy:",
            "  if len > 10,000 entries",
            "    → delete oldest 30% of keys",
            "    → retain 7,000 entries (70%)",
            "",
            "L2 disk: ~3 KB per .pkl file",
            "(np.ndarray float32 768 dims)",
        ],
        right=[
            "# L1 write",
            "with self.lock:",
            "  memory_cache[key] = embedding",
            "  if len(memory_cache) > 10000:",
            "    # FIFO evict oldest 30%",
            "    to_del = keys[:3000]",
            "    for k in to_del: del cache[k]",
            "",
            "# L2 write",
            "with open(cache_path, 'wb') as f:",
            "  pickle.dump(embedding, f)",
            "",
            "self.cache_misses += 1",
        ],
        formula="~3 KB / embedding    FIFO evict at 10K → retain 7K    miss_count += 1",
        tags=["L1 FIFO EVICT","L2 PICKLE ~3KB","miss_count++","THREAD-SAFE"],
        shape_out="cached v̂", dtype_out="stored: dict + .pkl", shape_vis="CACHED",
    ),
    dict(
        num="10", group="store", title="DUAL INDEX STORAGE",
        left=[
            "Normalized embedding written to",
            "BOTH search indexes with full metadata.",
            "",
            "FAISS (dense / semantic search):",
            "  IndexFlatIP — inner product index",
            "  cos_sim(q̂,d̂) = q̂ · d̂  (IP of units)",
            "  Persisted: biomedical_faiss.index",
            "",
            "Elasticsearch (sparse / BM25):",
            "  dense_vector field dims=768",
            "  BM25: text^2 title^1.5 abstract^1.2",
        ],
        right=[
            "# FAISS",
            "arr = np.array(embeds, dtype=float32)",
            "faiss.normalize_L2(arr)",
            "faiss_index.add(arr)",
            "faiss.write_index(idx, '*.index')",
            "pickle.dump(docs, meta_file)",
            "",
            "# Elasticsearch (if available)",
            "# ES mapping: dense_vector dims=768",
            "# fields: text, title, abstract,",
            "#  journal, year, pmid, doi,",
            "#  authors, chunk_id, timestamp",
            "",
            "chunk_id = f'{pmid}_{chunk_index}'",
        ],
        formula="FAISS: cos(q̂,d̂)=q̂·d̂   ES: dense_vector[768]   chunk_id={pmid}_{i}",
        tags=["IndexFlatIP","ES dense_vector 768","PERSIST .index","chunk_id=pmid_i"],
        shape_out="indexed v̂", dtype_out="FAISS ntotal+1  /  ES doc", shape_vis="STORED",
    ),
]

# ── SHAPE TRACKER DATA ───────────────────────────────────────────────────────
# Shown in right panel, per stage
TRACKER = [
    ("TEXT CONTENT",   "str",                    "#00C9A7", 0.2),
    ("EXPANDED TEXT",  "str  (abbrevs expanded)", "#9B72FF", 0.2),
    ("CACHE KEY",      "str  MD5 hex [32]",       "#FF9F43", 0.1),
    ("CHUNKS",         "List[str]  N×400w",        "#3EC88A", 0.35),
    ("TOKEN TENSORS",  "[1, T≤512] ×3",           "#FF6B9D", 0.5),
    ("HIDDEN STATE",   "[1,  T,  768]",            "#FF6060", 0.9),
    ("CLS VECTOR",     "[768,]  float32",          "#FF6060", 0.7),
    ("UNIT VECTOR",    "[768,]  ||v||=1",          "#F7DC6F", 0.7),
    ("CACHED v̂",      "dict + .pkl  ~3KB",        "#FF9F43", 0.7),
    ("INDEXED v̂",     "FAISS + ES  dims=768",     "#5DADE2", 0.7),
]

# ── HELPERS ─────────────────────────────────────────────────────────────────
def draw_rr(d, xy, r, fill=None, outline=None, w=1):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=w)

def tw(d, text, font):
    b = d.textbbox((0,0), text, font=font)
    return b[2]-b[0]

def th(d, text, font):
    b = d.textbbox((0,0), text, font=font)
    return b[3]-b[1]

# ── MEASURE STAGE HEIGHT ─────────────────────────────────────────────────────
def stage_h(s):
    n_left  = len(s["left"])
    n_right = len(s["right"])
    n_lines = max(n_left, n_right)
    h  = PAD
    h += 56         # num + title row
    h += 8
    h += 24         # desc / separator
    h += 14
    h += n_lines * 22
    if s["formula"]:
        h += 12 + 32
    h += 16 + 26    # tags
    h += PAD
    return h

# ── MAIN RENDER ─────────────────────────────────────────────────────────────
def render():
    HEADER_H = 148
    FOOTER_H = 72

    sh_list  = [stage_h(s) for s in STAGES]
    total_h  = (HEADER_H + sum(sh_list)
                + ARROW_H * (len(STAGES)-1)
                + FOOTER_H + 40)

    img  = Image.new("RGB", (W, total_h), BG)
    draw = ImageDraw.Draw(img)

    # dot grid (very subtle)
    for gx in range(80, W, 80):
        for gy in range(80, total_h, 80):
            draw.ellipse([gx-1,gy-1,gx+1,gy+1], fill="#091520")

    # vertical panel divider
    draw.line([(L_W+2, HEADER_H), (L_W+2, total_h-FOOTER_H)],
              fill=DIVIDER, width=2)

    # ── HEADER ───────────────────────────────────────────────────────────
    draw.rectangle([(0,0),(W,HEADER_H)], fill=HDR)
    draw.rectangle([(0,HEADER_H-3),(W,HEADER_H)], fill="#102030")

    draw.text((MARGIN, 20), "BIOBERT EMBEDDING CREATION PIPELINE",
              font=fnt["title"], fill="#42D8C4")
    title_w = tw(draw, "BIOBERT EMBEDDING CREATION PIPELINE", fnt["title"])
    draw.rectangle([(MARGIN, 104),(MARGIN+title_w, 107)], fill="#42D8C4"+"44")
    draw.text((MARGIN, 114),
              "BioChatAI  ·  embedding_service.py  ·  dmis-lab/biobert-v1.1  ·  768-dim vectors  ·  L2-normalized  ·  FAISS IndexFlatIP",
              font=fnt["sub"], fill="#1E3A50")

    # RIGHT PANEL HEADER
    draw.rectangle([(L_W+4, 0),(W, HEADER_H)], fill="#040C16")
    rhx = L_W + 4 + (R_W)//2
    rh_lbl = "DATA SHAPE TRACKER"
    rlw = tw(draw, rh_lbl, fnt["tracker"])
    draw.text((rhx - rlw//2, 50), rh_lbl, font=fnt["tracker"], fill="#2A6080")
    draw.text((rhx - 80, 76), "output type per stage", font=fnt["legend"], fill="#1A3848")

    # Legend
    leg = [("DOCUMENT","assemble"),("PREPROCESSING","preproc"),("CHUNKING","chunk"),
           ("CACHE","cache"),("TOKENIZER","tokenize"),("MODEL","model"),
           ("MATH","math"),("STORAGE","store")]
    lx = L_W - MARGIN
    for lname, lkey in reversed(leg):
        lcol = GRP[lkey][0]
        lbb  = draw.textbbox((0,0), lname, font=fnt["legend"])
        lw2  = lbb[2]-lbb[0]+16
        lx  -= lw2+6
        draw_rr(draw, [lx,24,lx+lw2,44], 4, fill=lcol+"18", outline=lcol+"70", w=1)
        draw.text((lx+7,27), lname, font=fnt["legend"], fill=lcol)

    # ── STAGES ───────────────────────────────────────────────────────────
    y = HEADER_H + 18

    for idx, (stage, tracker) in enumerate(zip(STAGES, TRACKER)):
        accent, s_bg = GRP[stage["group"]]
        sh = sh_list[idx]
        bx0, bx1 = MARGIN, L_W - MARGIN

        # shadow
        draw_rr(draw, [bx0+4,y+4,bx1+4,y+sh+4], 10, fill="#020508")
        # panel
        draw_rr(draw, [bx0,y,bx1,y+sh], 10, fill=s_bg, outline=accent+"50", w=2)
        # left accent bar
        draw_rr(draw, [bx0,y,bx0+10,y+sh], 10, fill=accent)

        # ── number circle
        cx, cy = bx0+58, y+44
        draw.ellipse([cx-26,cy-26,cx+26,cy+26], fill=accent+"1A", outline=accent+"65", width=2)
        nbb = draw.textbbox((0,0), stage["num"], font=fnt["num"])
        draw.text((cx-(nbb[2]-nbb[0])//2, cy-(nbb[3]-nbb[1])//2-2),
                  stage["num"], font=fnt["num"], fill=accent)

        # ── title
        tx = bx0 + 104
        draw.text((tx, y+PAD),     stage["title"], font=fnt["sname"], fill="#E0F0FF")

        # ── thin separator
        sep_y = y+PAD+38
        draw.line([(tx, sep_y),(bx1-20, sep_y)], fill=accent+"25", width=1)

        # ── two-column inner layout
        col_split = tx + (bx1 - tx - 30) // 2 + 10   # midpoint for right col
        LEFT_X  = tx
        RIGHT_X = col_split + 14

        # vertical divider between columns
        draw.line([(col_split, sep_y+8),(col_split, y+sh-PAD-36)],
                  fill=accent+"18", width=1)

        dy = sep_y + 12
        n_lines = max(len(stage["left"]), len(stage["right"]))

        for li in range(n_lines):
            if li < len(stage["left"]):
                line = stage["left"][li]
                col  = CODE if line.startswith(("#","→","#")) else \
                       (accent if line and line[0] in "0123456789" else "#546A78")
                draw.text((LEFT_X, dy), line, font=fnt["code"], fill=col)
            if li < len(stage["right"]):
                rline = stage["right"][li]
                rcol  = CODE if rline.strip().startswith(("#","#")) else \
                        (FORM if "=" in rline and not rline.startswith(" ") else CODE)
                draw.text((RIGHT_X, dy), rline, font=fnt["code"], fill=CODE)
            dy += 22

        # ── formula bar
        if stage["formula"]:
            fy = dy+8
            draw_rr(draw, [tx-4,fy,bx1-18,fy+30], 5, fill="#030C18", outline=accent+"28", w=1)
            draw_rr(draw, [tx-4,fy,tx, fy+30], 5, fill=accent+"50")
            draw.text((tx+8,fy+6), stage["formula"], font=fnt["bold_c"], fill=FORM)
            dy = fy+38

        # ── tags
        tag_y = dy+12; tx2 = tx
        for tag in stage["tags"]:
            tbb = draw.textbbox((0,0), tag, font=fnt["tag"])
            tw2 = tbb[2]-tbb[0]+18
            draw_rr(draw, [tx2,tag_y,tx2+tw2,tag_y+24], 5,
                    fill=accent+"16", outline=accent+"50", w=1)
            draw.text((tx2+8,tag_y+5), tag, font=fnt["tag"], fill=accent)
            tx2 += tw2+8

        # ── RIGHT PANEL: shape tracker cell
        rx0 = L_W + 8
        rx1 = W - 14
        rcy = y + sh//2

        t_label, t_dtype, t_col, t_fill = tracker

        # cell background
        draw_rr(draw, [rx0+8, y+10, rx1-8, y+sh-10], 8,
                fill=TRACKER_BG, outline=t_col+"30", w=1)

        # stage label (small, top of cell)
        slbl = f"STAGE {stage['num']}"
        draw.text((rx0+20, y+18), slbl, font=fnt["legend"], fill=t_col+"80")

        # shape name (large, centred)
        sname_lbl = t_label
        snw = tw(draw, sname_lbl, fnt["shape"])
        draw.text((rx0+8+(rx1-rx0-snw)//2, rcy-28), sname_lbl,
                  font=fnt["shape"], fill=t_col)

        # dtype label
        dtw = tw(draw, t_dtype, fnt["dtype"])
        draw.text((rx0+8+(rx1-rx0-dtw)//2, rcy+4), t_dtype,
                  font=fnt["dtype"], fill=t_col+"90")

        # visual bar (proportional to 768)
        bar_w = int((rx1-rx0-40) * t_fill)
        bar_x0 = rx0+20
        bar_y  = rcy+30
        draw_rr(draw, [bar_x0, bar_y, bar_x0+rx1-rx0-40, bar_y+10], 4,
                fill=t_col+"15", outline=t_col+"20", w=1)
        if bar_w > 4:
            draw_rr(draw, [bar_x0, bar_y, bar_x0+bar_w, bar_y+10], 4, fill=t_col+"70")

        # connector dot (left of right panel)
        draw.ellipse([rx0,rcy-5, rx0+10, rcy+5], fill=t_col)
        draw.line([(bx1+4, rcy),(rx0, rcy)], fill=t_col+"40", width=1)

        y += sh

        # ── ARROW between stages
        if idx < len(STAGES)-1:
            ax  = (MARGIN + L_W - MARGIN) // 2
            ay0 = y+8; ay1 = y+ARROW_H-16

            # cache-hit bypass label after stage 03
            if idx == 2:
                draw.line([(ax,ay0),(ax,ay1)], fill=ARROW_C, width=3)
                draw.polygon([(ax-9,ay1),(ax+9,ay1),(ax,ay1+13)], fill=ARROW_C)
                draw.text((ax+14,ay0+4), "CACHE MISS → continue",
                          font=fnt["legend"], fill=ARROW_C)
                # bypass annotation right side
                byp_x = bx1 - 60
                draw.text((byp_x, ay0+4), "CACHE HIT → skip to stage 09",
                          font=fnt["legend"], fill=GRP["cache"][0])
            else:
                draw.line([(ax,ay0),(ax,ay1)], fill=ARROW_C, width=3)
                draw.polygon([(ax-9,ay1),(ax+9,ay1),(ax,ay1+13)], fill=ARROW_C)
            y += ARROW_H

    # ── FOOTER ────────────────────────────────────────────────────────────
    fy = y+20
    draw.rectangle([(0,fy),(W,fy+FOOTER_H)], fill=HDR)
    draw.rectangle([(0,fy),(W,fy+2)], fill="#102030")

    math_footer = (
        "v̂ = v/||v||₂   |   cos(v̂,ŵ)=v̂·ŵ   |   chunk[i]=words[i*320:i*320+400]   |   "
        "N=ceil((|w|-400)/320)+1   |   key=MD5(text)   |   hit_rate=hits/(hits+misses)"
    )
    draw.text((MARGIN, fy+16), math_footer, font=fnt["footer"], fill="#1E4060")

    sig = "BioChatAI  ·  BioBERT Embedding Pipeline  ·  dmis-lab/biobert-v1.1  dim=768"
    sigw = tw(draw, sig, fnt["footer"])
    draw.text((W-MARGIN-sigw, fy+42), sig, font=fnt["footer"], fill="#162838")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, dpi=(144,144))
    print(f"Saved: {OUT}  [{W} x {total_h} px]")

render()
