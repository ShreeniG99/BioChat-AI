"""
BioChatAI – BioBERT Embedding Creation Pipeline Infographic
Clinical Cartography · Precision visual documentation
"""
from PIL import Image, ImageDraw, ImageFont
import os, math

FONTS = r"C:\Users\shree\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\f745a90e-e461-43ac-91fe-1a92c06ef178\a7144673-07b7-4651-a8ba-912ef5d6e75d\skills\canvas-design\canvas-fonts"
OUT   = r"C:\Dev\BioChatAI_final\assets\biobert_embedding_pipeline.png"

def F(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)

# ── PALETTE ──────────────────────────────────────────────────────────────────
BG        = "#06101C"
HEADER_BG = "#030C16"
MUTED     = "#3B5468"
CODE_CLR  = "#52D8CC"
FORMULA   = "#7EFFD4"
ARROW_TIP = "#3ABED0"
BRANCH_CLR= "#FF9F43"   # cache-hit branch arrow

GRP = {
    "input":   ("#4A9EFF", "#071830"),  # blue  — input/output
    "preproc": ("#AA72FF", "#100828"),  # purple — preprocessing
    "cache":   ("#FF9F43", "#1C1006"),  # amber  — cache system
    "chunk":   ("#3EC88A", "#061A10"),  # green  — chunking
    "model":   ("#FF6B6B", "#1C0606"),  # coral  — model/inference
    "math":    ("#F7DC6F", "#1A1800"),  # gold   — math/normalization
    "store":   ("#5DADE2", "#060E1A"),  # sky    — storage
}

W       = 2400
MARGIN  = 90
PAD     = 30
ARROW_H = 54
INNER_W = W - 2 * MARGIN

# ── STAGE DATA ────────────────────────────────────────────────────────────────
STAGES = [
    dict(
        num="01", group="input",
        title="ARTICLE TEXT ASSEMBLY",
        desc="Three document sections are concatenated into a single content string before any processing begins.",
        details=[
            'title    = document.get("title", "")         — e.g. "CRISPR-Cas9 mechanisms in gene editing"',
            'abstract = document.get("abstract", "")      — full abstract text (primary content source)',
            'full_text= document.get("full_text", "")     — extended body text (if available)',
            'content  = "Title: {title}  Abstract: {abstract}  Full Text: {full_text}"',
        ],
        formula=None,
        tags=["DOCUMENT DICT IN", "3-SECTION CONCAT", "EMBEDDING_SERVICE.PY · embed_document()"],
    ),
    dict(
        num="02", group="preproc",
        title="BIOMEDICAL TEXT PREPROCESSING",
        desc="_preprocess_biomedical_text() expands 13 critical medical abbreviations before tokenization.",
        details=[
            'HTN  → hypertension             MI      → myocardial infarction',
            'COPD → chronic obstructive pulmonary disease    CHF → congestive heart failure',
            'DM   → diabetes mellitus        CKD     → chronic kidney disease',
            'mRNA → messenger RNA            HIV     → human immunodeficiency virus',
            'COVID-19 → coronavirus disease 2019    DNA → deoxyribonucleic acid',
            'Method: split text → strip punctuation from each word → replace if in dict → rejoin',
        ],
        formula='_preprocess_biomedical_text(text)   →   expanded_text: str',
        tags=["13 ABBREVIATIONS", "WORD SPLIT + REPLACE", "PUNCTUATION-AWARE"],
    ),
    dict(
        num="03", group="chunk",
        title="OVERLAPPING DOCUMENT CHUNKING",
        desc="Long documents are split into overlapping fixed-size word windows to preserve local context across chunk boundaries.",
        details=[
            'chunk_size  = 400 words  (per chunk)',
            'overlap     =  80 words  (shared tokens between adjacent chunks)',
            'stride      = 320 words  (= chunk_size - overlap)',
            'num_chunks  = ceil((|words| - 400) / 320) + 1   (for |words| > 400)',
            'chunk[i]    = words[ i·320 : i·320 + 400 ]',
            'batch_size  =   8 chunks processed in parallel per embed_document() call',
        ],
        formula='chunk_document(text, chunk_size=400, overlap=80)   →   List[str]',
        tags=["400-WORD CHUNKS", "80-WORD OVERLAP", "STRIDE = 320", "BATCH SIZE = 8"],
    ),
    dict(
        num="04", group="cache",
        title="TWO-LEVEL CACHE LOOKUP",
        desc="Before any model inference, both cache layers are checked. A hit at either level skips all computation.",
        details=[
            'Cache key:  hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()   → 32-char hex string',
            'L1 — Memory cache:  self.memory_cache[key]  (Python dict, max 10,000 entries)',
            '                    Thread-safe via threading.Lock()',
            'L2 — Disk cache:    embeddings_cache/chunks/{cache_key}.pkl',
            '                    pickle.load(f) → np.ndarray [768,]',
            '                    TTL = 30 days  (auto-deleted if older than 30 × 24 × 3600 s)',
            'CACHE HIT  → return embedding immediately (skip stages 05–07)',
            'CACHE MISS → continue to BioBERT tokenization',
        ],
        formula='cache_key = MD5(text)   |   hit_rate = cache_hits / (cache_hits + cache_misses)',
        tags=["L1 MEMORY (10K MAX)", "L2 DISK .pkl (30-DAY TTL)", "CACHE HIT → SKIP MODEL", "THREAD-SAFE LOCK"],
    ),
    dict(
        num="05", group="model",
        title="BERT WORDPIECE TOKENIZATION",
        desc="AutoTokenizer from dmis-lab/biobert-v1.1 converts text to subword token IDs ready for the transformer.",
        details=[
            'tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")',
            'inputs    = tokenizer(text, return_tensors="pt",',
            '                      truncation=True, padding=True, max_length=512)',
            'Outputs three tensors — all shape [1, seq_len]:',
            '  input_ids      — integer token IDs (vocab size: 28,996)',
            '  attention_mask — 1 for real tokens, 0 for padding',
            '  token_type_ids — 0 for single-sentence input (no NSP)',
            'Special tokens: [CLS] prepended (id=101)  [SEP] appended (id=102)',
            'Max tokens: 512  (texts truncated beyond this; ~380 avg words)',
        ],
        formula='inputs → { input_ids: [1, T],  attention_mask: [1, T],  token_type_ids: [1, T] }   T ≤ 512',
        tags=["WORDPIECE SUBWORD", "MAX 512 TOKENS", "[CLS] + [SEP]", "VOCAB = 28,996"],
    ),
    dict(
        num="06", group="model",
        title="BIOBERT TRANSFORMER FORWARD PASS",
        desc="The tokenized input is passed through BioBERT's 12-layer transformer. The [CLS] token embedding is extracted as the document representation.",
        details=[
            'model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")',
            'model.eval()  →  inference mode (dropout disabled, BatchNorm in eval)',
            'device: CUDA if torch.cuda.is_available() else CPU',
            'with torch.no_grad():  →  no gradient computation (faster, less memory)',
            '  outputs = model(**inputs)',
            '  last_hidden_state: shape [1, T, 768]   (T = sequence length)',
            '  CLS extraction:    outputs.last_hidden_state[:, 0, :]   → [1, 768]',
            '  .cpu().numpy().squeeze()   →   np.ndarray  shape [768,]',
        ],
        formula='model(**inputs) → last_hidden_state[:, 0, :]   shape: [1, T, 768] → [768,]',
        tags=["12 LAYERS · 12 HEADS", "HIDDEN DIM = 768", "CLS TOKEN POOLING", "torch.no_grad()"],
    ),
    dict(
        num="07", group="math",
        title="L2 NORMALIZATION TO UNIT SPHERE",
        desc="The raw 768-dim vector is L2-normalized so its magnitude equals 1. This maps all embeddings onto a unit hypersphere, enabling cosine similarity via dot product.",
        details=[
            'raw_embedding:  v  ∈  R^768   (raw CLS output, arbitrary magnitude)',
            'L2 norm:        ||v||₂  =  sqrt( v₁² + v₂² + ... + v₇₆₈² )',
            'normalized:     v̂  =  v / ||v||₂          so  ||v̂||₂ = 1.0',
            'Property:  cos_sim(v̂, ŵ) = v̂ · ŵ  (dot product of unit vectors = cosine)',
            'Code:   embedding = embedding / np.linalg.norm(embedding)',
            'FAISS:  faiss.normalize_L2(embeddings_array)  applied again before index.add()',
        ],
        formula='v̂ = v / ||v||₂   where  ||v||₂ = sqrt(Σᵢ vᵢ²)     cos(v̂, ŵ) = v̂ · ŵ  ∈ [-1, 1]',
        tags=["||v̂||₂ = 1.0", "HYPERSPHERE PROJECTION", "DOT PRODUCT = COSINE SIM", "np.linalg.norm()"],
    ),
    dict(
        num="08", group="cache",
        title="CACHE WRITE-BACK",
        desc="The freshly computed normalized embedding is stored in both cache layers before being forwarded to storage.",
        details=[
            'L1 write:  self.memory_cache[cache_key] = embedding',
            '           Eviction policy: FIFO — if len > 10,000 → remove oldest 30% of keys',
            '           Target after eviction: 70% × 10,000 = 7,000 entries',
            'L2 write:  pickle.dump(embedding, open(cache_path, "wb"))',
            '           Path: embeddings_cache/chunks/{md5_hash}.pkl',
            '           File stores:  np.ndarray  dtype=float32  shape=[768,]',
        ],
        formula='self.memory_cache[key] = v̂    +    pickle.dump(v̂, path)    cache_misses += 1',
        tags=["FIFO EVICTION AT 10K", "WRITE TO MEMORY + DISK", "FLOAT32 PICKLE"],
    ),
    dict(
        num="09", group="store",
        title="DUAL INDEX STORAGE",
        desc="The normalized embedding and its full metadata are written into both FAISS (dense search) and Elasticsearch (sparse search).",
        details=[
            'FAISS (IndexFlatIP):',
            '  embeddings_array = np.array(embeddings, dtype=np.float32)',
            '  faiss.normalize_L2(embeddings_array)   →   re-normalize as float32 batch',
            '  self.faiss_index.add(embeddings_array) →   append to flat IP index',
            '  faiss.write_index(index, "biomedical_faiss.index")  +  pickle metadata',
            'Elasticsearch (if available):',
            '  Field: "embedding" → dense_vector, dims=768',
            '  Fields: text, title, abstract, journal, year, pmid, doi, chunk_id, timestamp',
            'Chunk metadata attached: pmid, title, journal, year, doi, chunk_index, total_chunks',
        ],
        formula='chunk_id = f"{pmid}_{chunk_index}"    stored in: FAISS IndexFlatIP  +  ES dense_vector[768]',
        tags=["FAISS IndexFlatIP", "ES dense_vector dims=768", "METADATA PICKLE", "biomedical_faiss.index"],
    ),
]

def F_load(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)

fonts = {
    "title":    F_load("Jura-Medium.ttf",          80),
    "subtitle": F_load("Jura-Light.ttf",            23),
    "num":      F_load("Outfit-Bold.ttf",           48),
    "sname":    F_load("Outfit-Bold.ttf",           34),
    "desc":     F_load("InstrumentSans-Regular.ttf",23),
    "detail":   F_load("JetBrainsMono-Regular.ttf", 19),
    "formula":  F_load("JetBrainsMono-Bold.ttf",    19),
    "tag":      F_load("Jura-Medium.ttf",           15),
    "footer":   F_load("JetBrainsMono-Regular.ttf", 15),
    "legend":   F_load("Jura-Light.ttf",            14),
    "branch":   F_load("Outfit-Bold.ttf",           18),
}

def stage_h(s):
    h  = PAD
    h += 60             # num / title
    h += 8
    h += 30             # desc
    h += 14
    h += len(s["details"]) * 29
    if s["formula"]:
        h += 12 + 36
    h += 18 + 28        # tags
    h += PAD
    return h

def render():
    HEADER_H = 162
    FOOTER_H = 88

    sh_list  = [stage_h(s) for s in STAGES]
    total_h  = HEADER_H + sum(sh_list) + ARROW_H * (len(STAGES) - 1) + FOOTER_H + 50

    img  = Image.new("RGB", (W, total_h), BG)
    draw = ImageDraw.Draw(img)

    # dot grid
    for gx in range(80, W, 80):
        for gy in range(80, total_h, 80):
            draw.ellipse([gx-1, gy-1, gx+1, gy+1], fill="#0A1A28")

    # ── HEADER ───────────────────────────────────────────────────────────────
    draw.rectangle([(0,0),(W, HEADER_H)], fill=HEADER_BG)
    draw.rectangle([(0, HEADER_H-3),(W, HEADER_H)], fill="#142438")
    draw.text((MARGIN, 20), "BIOBERT EMBEDDING CREATION PIPELINE",
              font=fonts["title"], fill="#52D8CC")
    tbb = draw.textbbox((0,0), "BIOBERT EMBEDDING CREATION PIPELINE", font=fonts["title"])
    draw.rectangle([(MARGIN, 108),(MARGIN + tbb[2] + 2, 111)], fill="#52D8CC"+"44")
    draw.text((MARGIN, 122),
              "BioChatAI  ·  embedding_service.py  ·  vector_search_service.py  ·  dmis-lab/biobert-v1.1",
              font=fonts["subtitle"], fill="#254860")

    # Legend
    legend = [
        ("DOCUMENT I/O",        GRP["input"][0]),
        ("TEXT PREPROCESSING",  GRP["preproc"][0]),
        ("CHUNKING",            GRP["chunk"][0]),
        ("CACHE SYSTEM",        GRP["cache"][0]),
        ("MODEL INFERENCE",     GRP["model"][0]),
        ("MATH / NORMALIZATION",GRP["math"][0]),
        ("INDEX STORAGE",       GRP["store"][0]),
    ]
    lx = W - MARGIN
    for lname, lcol in reversed(legend):
        lbb = draw.textbbox((0,0), lname, font=fonts["legend"])
        lw  = lbb[2] - lbb[0] + 20
        lx -= lw + 8
        draw.rounded_rectangle([lx, 36, lx+lw, 58], radius=4,
                                fill=lcol+"1A", outline=lcol+"80", width=1)
        draw.text((lx+8, 39), lname, font=fonts["legend"], fill=lcol)

    # ── STAGES ───────────────────────────────────────────────────────────────
    y = HEADER_H + 22

    for idx, stage in enumerate(STAGES):
        accent, s_bg = GRP[stage["group"]]
        sh = sh_list[idx]
        bx0, bx1 = MARGIN, W - MARGIN

        # shadow
        draw.rounded_rectangle([bx0+5, y+5, bx1+5, y+sh+5], radius=10, fill="#020508")
        # panel
        draw.rounded_rectangle([bx0, y, bx1, y+sh], radius=10,
                                fill=s_bg, outline=accent+"55", width=2)
        # left bar
        draw.rounded_rectangle([bx0, y, bx0+10, y+sh], radius=10, fill=accent)

        # number circle
        cx, cy = bx0+64, y+46
        draw.ellipse([cx-28, cy-28, cx+28, cy+28],
                     fill=accent+"1E", outline=accent+"70", width=2)
        nbb = draw.textbbox((0,0), stage["num"], font=fonts["num"])
        nw, nh = nbb[2]-nbb[0], nbb[3]-nbb[1]
        draw.text((cx-nw//2, cy-nh//2-2), stage["num"], font=fonts["num"], fill=accent)

        tx = bx0 + 112

        # title + desc
        draw.text((tx, y+PAD),      stage["title"], font=fonts["sname"], fill="#E8F4FF")
        draw.text((tx, y+PAD+46),   stage["desc"],  font=fonts["desc"],  fill="#4A6A80")

        # thin separator
        sep_y = y + PAD + 46 + 32 + 6
        draw.line([(tx, sep_y),(bx1-24, sep_y)], fill=accent+"20", width=1)

        # details
        dy = sep_y + 10
        for line in stage["details"]:
            draw.ellipse([tx, dy+9, tx+8, dy+17], fill=accent+"65")
            draw.text((tx+16, dy), line, font=fonts["detail"], fill=CODE_CLR)
            dy += 29

        # formula box
        if stage["formula"]:
            fy = dy + 10
            draw.rounded_rectangle([tx-6, fy, bx1-24, fy+32],
                                   radius=5, fill="#030C18", outline=accent+"30", width=1)
            draw.rounded_rectangle([tx-6, fy, tx-2, fy+32], radius=5, fill=accent+"55")
            draw.text((tx+8, fy+7), stage["formula"], font=fonts["formula"], fill=FORMULA)
            dy = fy + 42

        # tags
        tag_y = dy + 14
        tx2 = tx
        for tag in stage["tags"]:
            tbb = draw.textbbox((0,0), tag, font=fonts["tag"])
            tw  = tbb[2]-tbb[0]+20
            draw.rounded_rectangle([tx2, tag_y, tx2+tw, tag_y+26],
                                   radius=5, fill=accent+"18", outline=accent+"55", width=1)
            draw.text((tx2+10, tag_y+5), tag, font=fonts["tag"], fill=accent)
            tx2 += tw+10

        y += sh

        # ── ARROW / BRANCH CONNECTOR ──────────────────────────────────────
        if idx < len(STAGES)-1:
            ax     = W//2
            ay0    = y + 8
            ay1    = y + ARROW_H - 16

            # Special branch indicator after CACHE (stage 04, idx=3)
            if idx == 3:
                # Main arrow (CACHE MISS path)
                draw.line([(ax, ay0),(ax, ay1)], fill=ARROW_TIP, width=3)
                draw.polygon([(ax-10,ay1),(ax+10,ay1),(ax,ay1+14)], fill=ARROW_TIP)

                # Branch label: CACHE MISS
                mid_y = (ay0+ay1)//2
                draw.text((ax+18, mid_y-10), "CACHE MISS", font=fonts["branch"], fill=ARROW_TIP)

                # CACHE HIT bypass arrow — curved right side
                bx_start = W - MARGIN - 24
                by_start = y - sh_list[3] + 60        # top of cache box
                by_end   = y + ARROW_H + sh_list[4] + sh_list[5] + sh_list[6] + ARROW_H*3 + 20

                # dashed horizontal line right
                dash_y = y - sh_list[3]//2
                for dx in range(bx_start, bx_start+90, 14):
                    draw.line([(dx, dash_y),(dx+8, dash_y)], fill=BRANCH_CLR, width=2)

                # vertical line on far right
                right_x = bx_start + 96
                draw.line([(right_x, dash_y),(right_x, by_end)], fill=BRANCH_CLR, width=2)

                # horizontal back to pipeline at bypass end
                bypass_y = y + ARROW_H * 3 + sum(sh_list[4:7]) + 30
                for dx2 in range(bx_start, right_x, 14):
                    draw.line([(dx2, bypass_y),(dx2+8, bypass_y)], fill=BRANCH_CLR, width=2)

                # bypass label
                draw.text((right_x+6, dash_y + (bypass_y-dash_y)//2 - 10),
                           "CACHE\nHIT", font=fonts["branch"], fill=BRANCH_CLR)
            else:
                draw.line([(ax, ay0),(ax, ay1)], fill=ARROW_TIP, width=3)
                draw.polygon([(ax-10,ay1),(ax+10,ay1),(ax,ay1+14)], fill=ARROW_TIP)

            y += ARROW_H

    # ── FOOTER ───────────────────────────────────────────────────────────────
    footer_y = y + 28
    draw.rectangle([(0, footer_y),(W, footer_y+FOOTER_H)], fill=HEADER_BG)
    draw.rectangle([(0, footer_y),(W, footer_y+2)], fill="#142438")

    math = (
        "v̂ = v / ||v||₂    "
        "|   cos_sim(v̂, ŵ) = v̂ · ŵ    "
        "|   cache_key = MD5(text)    "
        "|   chunk[i] = words[i*320 : i*320+400]    "
        "|   num_chunks = ceil((|w|-400)/320)+1"
    )
    draw.text((MARGIN, footer_y+16), math, font=fonts["footer"], fill="#1E4A60")

    sig = "BioChatAI  BioBERT Embedding Pipeline  dmis-lab/biobert-v1.1  dim=768"
    sbb = draw.textbbox((0,0), sig, font=fonts["footer"])
    draw.text((W - MARGIN - (sbb[2]-sbb[0]), footer_y+50), sig,
              font=fonts["footer"], fill="#183C58")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, dpi=(144, 144))
    print(f"Saved: {OUT}  [{W} x {total_h} px]")

render()
