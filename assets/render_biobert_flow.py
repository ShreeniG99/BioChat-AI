"""
BioChatAI – BioBERT Embedding Pipeline
Horizontal data-transformation diagram with tensor shapes, math annotations,
and a two-level cache branch. Academic / research-paper aesthetic.
"""
from PIL import Image, ImageDraw, ImageFont
import os

FONTS = r"C:\Users\shree\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\f745a90e-e461-43ac-91fe-1a92c06ef178\a7144673-07b7-4651-a8ba-912ef5d6e75d\skills\canvas-design\canvas-fonts"
OUT   = r"C:\Dev\BioChatAI_final\assets\biobert_embedding_flow.png"

def F(name, sz):
    return ImageFont.truetype(os.path.join(FONTS, name), sz)

# ── CANVAS ─────────────────────────────────────────────────────────────────
W, H = 4200, 2050
BG   = "#07101C"

# ── PALETTE ────────────────────────────────────────────────────────────────
C = {
    "bg":      "#07101C",
    "panel":   "#0C1A28",
    "header":  "#040C18",
    "grid":    "#0A1824",
    "border":  "#162436",
    "white":   "#D8EAF4",
    "muted":   "#3A5668",
    "code":    "#4ED8CC",
    "form":    "#80FFD8",
    "arrow":   "#2A7090",
    "arrowt":  "#3AC0D8",
    # node group colours
    "doc":     "#4A9EFF",   # blue   – document
    "prep":    "#AA72FF",   # purple – preprocessing
    "chunk":   "#3EC88A",   # green  – chunking
    "cache":   "#FF9F43",   # amber  – cache
    "tok":     "#FF6B6B",   # coral  – tokenizer
    "model":   "#E8507A",   # rose   – model
    "norm":    "#F7DC6F",   # gold   – normalisation
    "store":   "#5DADE2",   # sky    – storage
    "bypass":  "#FF9F43",   # same as cache for bypass line
}

def Fp(name, sz): return F(name, sz)

fonts = {
    "title":   Fp("Jura-Medium.ttf",          62),
    "sub":     Fp("Jura-Light.ttf",            22),
    "node_h":  Fp("Outfit-Bold.ttf",           26),
    "node_b":  Fp("InstrumentSans-Regular.ttf",18),
    "code":    Fp("JetBrainsMono-Regular.ttf", 16),
    "shape":   Fp("JetBrainsMono-Bold.ttf",    17),
    "arrow_l": Fp("Jura-Light.ttf",            15),
    "tag":     Fp("Jura-Medium.ttf",           13),
    "math":    Fp("JetBrainsMono-Regular.ttf", 15),
    "footer":  Fp("JetBrainsMono-Regular.ttf", 14),
    "legend":  Fp("Jura-Light.ttf",            14),
    "branch":  Fp("Outfit-Bold.ttf",           16),
    "section": Fp("Outfit-Bold.ttf",           18),
}

# ── NODE DEFINITIONS ───────────────────────────────────────────────────────
# Each node: x-centre, body lines, colour key, shape label (above arrow)
# Layout: two rows
#   ROW A (top):    DOC → PREPROC → CHUNK → CACHE → TOKENIZER → MODEL → CLS
#   ROW B (bottom): CACHE HIT bypass + L2 NORM → CACHE WRITE → FAISS/ES

NODE_W  = 310
NODE_H  = 290
ROW_A_Y = 290        # centre y of row A nodes
ROW_B_Y = 1640       # centre y of row B nodes
MARGIN  = 60

# Row A x-centres  (7 nodes)
RA = [MARGIN + NODE_W//2 + i*(NODE_W + 90) for i in range(7)]
# Row B x-centres  (4 nodes — norm, cache-write, faiss, es) aligned under cols 3-6
RB_BASE = RA[3]
RB = [RB_BASE + i*(NODE_W + 90) for i in range(4)]

def draw_node(draw, cx, cy, w, h, title, lines, col, tags=(), radius=12):
    x0, y0 = cx - w//2, cy - h//2
    x1, y1 = cx + w//2, cy + h//2
    # shadow
    draw.rounded_rectangle([x0+4, y0+4, x1+4, y1+4], radius=radius, fill="#020508")
    # panel
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                            fill=C["panel"], outline=col+"55", width=2)
    # top accent stripe
    draw.rounded_rectangle([x0, y0, x1, y0+8], radius=radius, fill=col)

    # title
    tbb = draw.textbbox((0,0), title, font=fonts["node_h"])
    tw = tbb[2]-tbb[0]
    draw.text((cx - tw//2, y0+18), title, font=fonts["node_h"], fill=col)

    # separator
    draw.line([(x0+14, y0+50), (x1-14, y0+50)], fill=col+"30", width=1)

    # body lines
    ty = y0 + 60
    for line in lines:
        draw.text((x0+14, ty), line, font=fonts["code"], fill=C["code"])
        ty += 22

    # tags
    if tags:
        tgy = y1 - 34
        tgx = x0 + 10
        for tag in tags:
            tbb2 = draw.textbbox((0,0), tag, font=fonts["tag"])
            tw2 = tbb2[2]-tbb2[0]+12
            draw.rounded_rectangle([tgx, tgy, tgx+tw2, tgy+20],
                                   radius=4, fill=col+"18", outline=col+"50", width=1)
            draw.text((tgx+6, tgy+3), tag, font=fonts["tag"], fill=col)
            tgx += tw2 + 6

def draw_arrow(draw, x0, y, x1, col="#3AC0D8", label="", label_pos="above"):
    # horizontal arrow
    draw.line([(x0, y), (x1-14, y)], fill=col, width=3)
    draw.polygon([(x1-14, y-8), (x1-14, y+8), (x1, y)], fill=col)
    if label:
        lbb = draw.textbbox((0,0), label, font=fonts["arrow_l"])
        lw  = lbb[2]-lbb[0]
        lx  = (x0+x1)//2 - lw//2
        ly  = y - 22 if label_pos == "above" else y + 8
        draw.text((lx, ly), label, font=fonts["arrow_l"], fill=col)

def draw_shape_badge(draw, cx, y, text, col):
    bb = draw.textbbox((0,0), text, font=fonts["shape"])
    w  = bb[2]-bb[0]+16
    x0 = cx - w//2
    draw.rounded_rectangle([x0, y, x0+w, y+24], radius=4,
                            fill=col+"22", outline=col+"80", width=1)
    draw.text((x0+8, y+3), text, font=fonts["shape"], fill=col)

def draw_vertical_arrow(draw, x, y0, y1, col, dashed=False):
    if dashed:
        seg = 14
        yy = y0
        while yy < y1 - seg:
            draw.line([(x, yy), (x, yy+seg-4)], fill=col, width=2)
            yy += seg
    else:
        draw.line([(x, y0), (x, y1-14)], fill=col, width=3)
    draw.polygon([(x-8, y1-14), (x+8, y1-14), (x, y1)], fill=col)

def draw_formula_strip(draw, x0, y, x1, text, col):
    draw.rounded_rectangle([x0, y, x1, y+30], radius=4,
                            fill="#030C18", outline=col+"30", width=1)
    draw.rounded_rectangle([x0, y, x0+4, y+30], radius=4, fill=col+"60")
    draw.text((x0+12, y+6), text, font=fonts["math"], fill=C["form"])

# ── RENDER ─────────────────────────────────────────────────────────────────
def render():
    img  = Image.new("RGB", (W, H), C["bg"])
    draw = ImageDraw.Draw(img)

    # dot grid
    for gx in range(80, W, 80):
        for gy in range(80, H, 80):
            draw.ellipse([gx-1,gy-1,gx+1,gy+1], fill=C["grid"])

    # ── HEADER ─────────────────────────────────────────────────────────────
    draw.rectangle([(0,0),(W,110)], fill=C["header"])
    draw.rectangle([(0,107),(W,110)], fill="#142438")
    draw.text((MARGIN, 18), "BIOBERT EMBEDDING CREATION PIPELINE",
              font=fonts["title"], fill="#4ED8CC")
    draw.text((MARGIN, 84),
              "BioChatAI  ·  dmis-lab/biobert-v1.1  ·  768-dim L2-normalized vectors  ·  FAISS IndexFlatIP  +  Elasticsearch dense_vector",
              font=fonts["sub"], fill=C["muted"])

    # Legend (top right)
    leg = [("DOCUMENT","doc"),("PREPROCESSING","prep"),("CHUNKING","chunk"),
           ("CACHE","cache"),("TOKENIZER/MODEL","tok"),("MATH","norm"),("STORAGE","store")]
    lx = W - MARGIN
    for lname, lkey in reversed(leg):
        lcol = C[lkey]
        lbb  = draw.textbbox((0,0), lname, font=fonts["legend"])
        lw   = lbb[2]-lbb[0]+18
        lx  -= lw+8
        draw.rounded_rectangle([lx,18,lx+lw,40], radius=4, fill=lcol+"18", outline=lcol+"70", width=1)
        draw.text((lx+8,21), lname, font=fonts["legend"], fill=lcol)

    # ════════════════════════════════════════════════════════════════════════
    # ROW A — NODES
    # ════════════════════════════════════════════════════════════════════════

    # ── A0: ARTICLE TEXT ───────────────────────────────────────────────────
    draw_node(draw, RA[0], ROW_A_Y, NODE_W, NODE_H,
        "ARTICLE TEXT",
        ["title: str",
         "abstract: str",
         "full_text: str",
         "",
         '"Title: {t}',
         ' Abstract: {a}',
         ' Full Text: {f}"',
         "",
         "→ content: str"],
        C["doc"], tags=["DICT IN","embed_document()"])

    # ── A1: PREPROCESS ─────────────────────────────────────────────────────
    draw_node(draw, RA[1], ROW_A_Y, NODE_W, NODE_H,
        "PREPROCESS",
        ["13 abbreviations",
         "word.strip('.,;:()')",
         "if word in abbrevs:",
         "  replace(word)",
         "",
         "HTN→hypertension",
         "MI→myocard. infarct.",
         "COVID-19→coronavirus",
         "→ expanded_text: str"],
        C["prep"], tags=["13 TERMS","WORD-LEVEL"])

    # ── A2: CHUNK ──────────────────────────────────────────────────────────
    draw_node(draw, RA[2], ROW_A_Y, NODE_W, NODE_H,
        "CHUNK",
        ["chunk_size = 400 w",
         "overlap    =  80 w",
         "stride     = 320 w",
         "",
         "chunk[i]=",
         " words[i*320:",
         "        i*320+400]",
         "",
         "→ List[str]  len=N"],
        C["chunk"], tags=["400W / 80W OVERLAP","STRIDE=320"])

    # ── A3: CACHE CHECK ────────────────────────────────────────────────────
    draw_node(draw, RA[3], ROW_A_Y, NODE_W+20, NODE_H+20,
        "CACHE CHECK",
        ["key=MD5(text).hex()",
         "",
         "L1 memory_cache{}",
         "   max 10,000 entries",
         "   threading.Lock()",
         "",
         "L2 disk  .pkl  30d",
         "   embeddings_cache/",
         "   chunks/{key}.pkl"],
        C["cache"], tags=["L1 MEMORY","L2 DISK 30d TTL","THREAD-SAFE"])

    # ── A4: TOKENIZER ──────────────────────────────────────────────────────
    draw_node(draw, RA[4], ROW_A_Y, NODE_W, NODE_H,
        "TOKENIZER",
        ["AutoTokenizer",
         ".from_pretrained(",
         " biobert-v1.1)",
         "vocab = 28,996",
         "max_length = 512",
         "truncation = True",
         "padding    = True",
         "→[CLS]…tokens…[SEP]",
         "→ tensors (pt)"],
        C["tok"], tags=["WORDPIECE","MAX 512 TOKENS"])

    # ── A5: BIOBERT MODEL ──────────────────────────────────────────────────
    draw_node(draw, RA[5], ROW_A_Y, NODE_W, NODE_H,
        "BIOBERT",
        ["AutoModel",
         ".from_pretrained(",
         " biobert-v1.1)",
         "12 layers · 12 heads",
         "hidden dim = 768",
         "torch.no_grad()",
         "model(**inputs)",
         "last_hidden_state",
         "→ [1, T, 768]"],
        C["model"], tags=["BERT-BASE","768 DIM","NO GRAD"])

    # ── A6: CLS EXTRACTION ─────────────────────────────────────────────────
    draw_node(draw, RA[6], ROW_A_Y, NODE_W, NODE_H,
        "CLS EXTRACT",
        ["last_hidden_state",
         "  shape [1, T, 768]",
         "",
         "[:, 0, :]",
         "  → [1, 768]",
         "",
         ".cpu().numpy()",
         ".squeeze()",
         "→ ndarray [768,]"],
        C["model"], tags=["TOKEN 0","CLS POOLING","float32"])

    # ── SHAPE BADGES above row-A nodes ─────────────────────────────────────
    badge_y = ROW_A_Y - NODE_H//2 - 36
    shapes_a = [
        ("str", C["doc"]),
        ("str", C["prep"]),
        ("List[str]", C["chunk"]),
        ("cache_key: str", C["cache"]),
        ("{ids,mask} [1,T]", C["tok"]),
        ("[1, T, 768]", C["model"]),
        ("[768,] float32", C["model"]),
    ]
    for i,(sh,col) in enumerate(shapes_a):
        draw_shape_badge(draw, RA[i], badge_y, sh, col)

    # ── ROW A ARROWS ───────────────────────────────────────────────────────
    arrow_y = ROW_A_Y
    pairs   = [(0,1),(1,2),(2,3),(4,5),(5,6)]
    labels  = ["expand","chunk(400,80)","MD5 key","tokens [1,T]","[1,T,768]"]
    for (a,b),lbl in zip(pairs,labels):
        x0 = RA[a] + NODE_W//2 + (10 if a==3 else 0)
        x1 = RA[b] - NODE_W//2 - (10 if b==3 else 0)
        draw_arrow(draw, x0, arrow_y, x1, C["arrowt"], lbl, "above")

    # Arrow A3→A4 (CACHE MISS)
    x0m = RA[3] + (NODE_W+20)//2
    x1m = RA[4] - NODE_W//2
    draw_arrow(draw, x0m, arrow_y, x1m, C["arrowt"], "CACHE MISS", "above")

    # ════════════════════════════════════════════════════════════════════════
    # CACHE HIT BYPASS  (dashed arc from A3 above down to ROW B start)
    # ════════════════════════════════════════════════════════════════════════
    cache_top  = ROW_A_Y - (NODE_H+20)//2
    bypass_x   = RA[3] + (NODE_W+20)//2 + 20
    bypass_top = cache_top - 52
    norm_cx    = RB[0]   # L2 norm node centre

    # horizontal from cache top-right → bypass_x
    draw.line([(RA[3]+(NODE_W+20)//2, bypass_top+24),
               (bypass_x+60, bypass_top+24)], fill=C["bypass"], width=2)
    # draw dashes
    seg=14; xx=bypass_x+60
    while xx < norm_cx:
        draw.line([(xx,bypass_top+24),(xx+seg-4,bypass_top+24)], fill=C["bypass"], width=2)
        xx+=seg
    # vertical down to ROW B L2 node
    draw.line([(norm_cx, bypass_top+24),(norm_cx, ROW_B_Y - NODE_H//2 - 16)],
              fill=C["bypass"], width=2)
    draw.polygon([(norm_cx-8, ROW_B_Y-NODE_H//2-16),
                  (norm_cx+8, ROW_B_Y-NODE_H//2-16),
                  (norm_cx,   ROW_B_Y-NODE_H//2)], fill=C["bypass"])
    # label
    lbb = draw.textbbox((0,0), "CACHE HIT", font=fonts["branch"])
    lw  = lbb[2]-lbb[0]
    draw.text((norm_cx - lw//2, bypass_top-2), "CACHE HIT", font=fonts["branch"], fill=C["bypass"])
    draw.text((norm_cx - lw//2 + 4, bypass_top+16), "(skip model)", font=fonts["arrow_l"], fill=C["bypass"])

    # ── VERTICAL DROP  CLS → L2 NORM ───────────────────────────────────────
    drop_x  = RA[6]
    drop_y0 = ROW_A_Y + NODE_H//2
    drop_y1 = ROW_B_Y - NODE_H//2 - 16

    # elbow: down from A6 then left to RB[0]
    mid_y   = (drop_y0 + drop_y1)//2
    draw.line([(drop_x, drop_y0),(drop_x, mid_y)], fill=C["arrowt"], width=3)
    draw.line([(drop_x, mid_y),(norm_cx, mid_y)], fill=C["arrowt"], width=3)
    draw.line([(norm_cx, mid_y),(norm_cx, drop_y1)], fill=C["arrowt"], width=3)
    draw.polygon([(norm_cx-8,drop_y1),(norm_cx+8,drop_y1),(norm_cx,drop_y1+14)],
                 fill=C["arrowt"])

    # shape badge on elbow
    draw_shape_badge(draw, (drop_x+norm_cx)//2, mid_y - 28, "ndarray [768,]", C["model"])

    # ── label "CLS output"
    draw.text((drop_x+12, drop_y0+10), "CLS output\n[768,] float32",
              font=fonts["arrow_l"], fill=C["arrowt"])

    # ════════════════════════════════════════════════════════════════════════
    # ROW B — NODES
    # ════════════════════════════════════════════════════════════════════════

    # ── B0: L2 NORMALIZATION ───────────────────────────────────────────────
    draw_node(draw, RB[0], ROW_B_Y, NODE_W+20, NODE_H+20,
        "L2 NORMALIZE",
        ["v  ∈  R^768",
         "||v||₂ = sqrt(Σvᵢ²)",
         "",
         "v̂ = v / ||v||₂",
         "||v̂||₂ = 1.0",
         "",
         "code:",
         "embedding /=",
         " np.linalg.norm(emb)"],
        C["norm"], tags=["UNIT SPHERE","||v̂||=1","cos=dot product"])

    # ── B1: CACHE WRITE-BACK ───────────────────────────────────────────────
    draw_node(draw, RB[1], ROW_B_Y, NODE_W+20, NODE_H+20,
        "CACHE WRITE",
        ["L1: memory_cache",
         "  [key] = v̂",
         "  FIFO evict >10K",
         "  → keep 70% (7K)",
         "",
         "L2: pickle.dump(v̂,",
         "  open(path,'wb'))",
         "  ~3 KB per vector",
         "  cache_misses += 1"],
        C["cache"], tags=["FIFO EVICT","PICKLE float32","~3 KB/VECTOR"])

    # ── B2: FAISS INDEX ────────────────────────────────────────────────────
    draw_node(draw, RB[2], ROW_B_Y, NODE_W, NODE_H,
        "FAISS INDEX",
        ["IndexFlatIP(768)",
         "faiss.normalize_L2(",
         "  emb_array)",
         "index.add(array)",
         "",
         "cos_sim(q,d)=q·d",
         "(inner product of",
         " unit vectors)",
         "→ biomedical.index"],
        C["store"], tags=["IndexFlatIP","DOT=COSINE","PERSIST .index"])

    # ── B3: ELASTICSEARCH ──────────────────────────────────────────────────
    draw_node(draw, RB[3], ROW_B_Y, NODE_W, NODE_H,
        "ELASTICSEARCH",
        ["dense_vector(768)",
         "text, title, abstract",
         "journal, year, pmid",
         "doi, authors, chunk_id",
         "timestamp",
         "",
         "BM25 multi-field:",
         "text^2 title^1.5",
         "abstract^1.2"],
        C["store"], tags=["dense_vector 768","BM25 SPARSE","IF ES AVAILABLE"])

    # ── SHAPE BADGES below row-B nodes ─────────────────────────────────────
    badge_yb = ROW_B_Y + (NODE_H+20)//2 + 12
    shapes_b = [
        ("v̂ ∈ R^768  ||v̂||=1", C["norm"]),
        ("pickle [768,] float32", C["cache"]),
        ("IndexFlatIP ntotal+1", C["store"]),
        ("ES doc indexed", C["store"]),
    ]
    for i,(sh,col) in enumerate(shapes_b):
        draw_shape_badge(draw, RB[i], badge_yb, sh, col)

    # ── ROW B ARROWS ───────────────────────────────────────────────────────
    arrow_yb = ROW_B_Y
    rb_pairs = [(0,1),(1,2),(1,3)]
    rb_labels= ["v̂ normalized","v̂ to FAISS","v̂ + meta to ES"]
    for i,((a,b),lbl) in enumerate(zip(rb_pairs, rb_labels)):
        x0 = RB[a] + (NODE_W+20)//2
        x1 = RB[b] - (NODE_W//2 if b>=2 else (NODE_W+20)//2)
        if b == 3:
            # branch down from cache-write to ES
            fork_x = RB[1] + (NODE_W+20)//2
            fork_y = ROW_B_Y + 40
            draw.line([(fork_x, fork_y),(RB[3]-NODE_W//2, fork_y)], fill=C["arrowt"], width=3)
            draw.polygon([(RB[3]-NODE_W//2-14, fork_y-8),
                          (RB[3]-NODE_W//2-14, fork_y+8),
                          (RB[3]-NODE_W//2,    fork_y)], fill=C["arrowt"])
            draw.text(((fork_x+RB[3])//2-40, fork_y-20), lbl,
                      font=fonts["arrow_l"], fill=C["arrowt"])
        else:
            draw_arrow(draw, x0, arrow_yb, x1, C["arrowt"], lbl, "above")

    # ════════════════════════════════════════════════════════════════════════
    # FORMULA STRIPS (bottom annotation band)
    # ════════════════════════════════════════════════════════════════════════
    STRIP_Y = ROW_B_Y + (NODE_H+20)//2 + 60
    strips = [
        (MARGIN, "chunk[i] = words[ i*320 : i*320+400 ]   num_chunks = ceil((|w|-400)/320)+1",
         C["chunk"]),
        (MARGIN + 740,
         "v̂ = v/||v||₂    ||v̂||₂=1    cos(v̂,ŵ)=v̂·ŵ ∈[-1,1]   (FAISS IndexFlatIP computes this)",
         C["norm"]),
        (MARGIN + 1860,
         "hit_rate = cache_hits/(cache_hits+cache_misses)   evict if len>10K → keep 70%",
         C["cache"]),
    ]
    for sx, st, sc in strips:
        draw_formula_strip(draw, sx, STRIP_Y, sx+700, st, sc)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION LABELS
    # ════════════════════════════════════════════════════════════════════════
    sec_y = 128
    sections = [
        (RA[0], "① DOCUMENT", C["doc"]),
        (RA[1], "② PREPROCESS", C["prep"]),
        (RA[2], "③ CHUNK", C["chunk"]),
        (RA[3], "④ CACHE CHECK", C["cache"]),
        (RA[4], "⑤ TOKENIZE", C["tok"]),
        (RA[5], "⑥ INFER", C["model"]),
        (RA[6], "⑦ EXTRACT", C["model"]),
    ]
    for sx, slbl, scol in sections:
        sbb = draw.textbbox((0,0), slbl, font=fonts["section"])
        sw  = sbb[2]-sbb[0]
        draw.text((sx - sw//2, sec_y), slbl, font=fonts["section"], fill=scol)

    sec_b = [
        (RB[0], "⑧ L2 NORM", C["norm"]),
        (RB[1], "⑨ CACHE WRITE", C["cache"]),
        (RB[2], "⑩ FAISS", C["store"]),
        (RB[3], "⑪ ELASTICSEARCH", C["store"]),
    ]
    for sx, slbl, scol in sec_b:
        sbb = draw.textbbox((0,0), slbl, font=fonts["section"])
        sw  = sbb[2]-sbb[0]
        draw.text((sx - sw//2, ROW_B_Y - (NODE_H+20)//2 - 32), slbl,
                  font=fonts["section"], fill=scol)

    # ── ROW LABELS ────────────────────────────────────────────────────────
    draw.text((14, ROW_A_Y - 16), "PIPELINE\nROW A", font=fonts["legend"], fill=C["muted"])
    draw.text((14, ROW_B_Y - 16), "PIPELINE\nROW B", font=fonts["legend"], fill=C["muted"])

    # ── FOOTER ────────────────────────────────────────────────────────────
    FOOTER_Y = H - 52
    draw.rectangle([(0,FOOTER_Y),(W,H)], fill=C["header"])
    draw.rectangle([(0,FOOTER_Y),(W,FOOTER_Y+2)], fill="#142438")
    footer_txt = (
        "BERT-base  |  12 layers  |  12 attention heads  |  hidden_dim=768  |  "
        "vocab=28,996 WordPiece  |  max_seq=512  |  CLS pooling  |  "
        "L2-norm  |  IndexFlatIP (cos=IP of unit vecs)  |  "
        "cache: MD5 key · L1 dict 10K · L2 pkl 30d TTL"
    )
    draw.text((MARGIN, FOOTER_Y+16), footer_txt, font=fonts["footer"], fill=C["muted"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, dpi=(144, 144))
    print(f"Saved: {OUT}  [{W} x {H} px]")

render()
