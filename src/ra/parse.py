"""PDF -> testo diviso per sezioni + frasi su limiti/future work (F2)."""
import json, os, re, pymupdf
from .config import p

HEAD = [("abstract", r"abstract"), ("intro", r"(\d+\.?\s*|[IVX]+\.?\s*)?introduction"), ("related", r"(\d+\.?\s*|[IVX]+\.?\s*)?(related work|background|prior work|literature review)"),
        ("method", r"(\d+\.?\s*|[IVX]+\.?\s*)?(method\w*|approach|proposed\w*|system (design|model)|architecture|design|framework|problem (formulation|statement))"),
        ("experiments", r"(\d+\.?\s*|[IVX]+\.?\s*)?(experiment\w*|evaluation|empirical\w*|setup|implementation)"),
        ("results", r"(\d+\.?\s*|[IVX]+\.?\s*)?(results?|analysis|findings)( and discussion)?"),
        ("discussion", r"(\d+\.?\s*|[IVX]+\.?\s*)?(discussion|limitations?)"),
        ("conclusion", r"(\d+\.?\s*|[IVX]+\.?\s*)?(conclusions?|concluding remarks|future work|summary)( and future work)?"),
        ("references", r"(references|bibliography)")]
GAPRE = re.compile(r"[^.]*\b(limitation|future work|we leave|left for future|remains? (open|unexplored|an open)|does not address|not (yet )?(addressed|explored|considered)|open (problem|question|challenge)|beyond the scope)\b[^.]*\.", re.I)

def text_of(path):
    doc = pymupdf.open(path)
    pages = [pg.get_text("text") for pg in doc]
    lines_per_page = [set(l.strip() for l in t.splitlines()[:3] + t.splitlines()[-3:]) for t in pages]
    freq = {}
    for s in lines_per_page:
        for l in s: freq[l] = freq.get(l, 0) + 1
    rep = {l for l, c in freq.items() if c > max(3, len(pages) // 3)}
    out = "\n".join("\n".join(l for l in t.splitlines() if l.strip() not in rep) for t in pages)
    return out, len(pages)

def sections(text):
    marks = []
    for m in re.finditer(r"^[ \t]*(.{3,60}?)[ \t]*$", text, re.M):
        t = m.group(1).strip().rstrip(".:")
        for name, rx in HEAD:
            if re.fullmatch(rx, t, re.I): marks.append((m.start(), name)); break
    out, cur = {}, None
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        if name in out and name != "references": continue
        out[name] = text[pos:end].strip()
        if name == "references": break
    return out

def parse(paper):
    from .harvest import download_pdf
    cp = p("data/text", re.sub(r"[^\w.-]", "_", paper["id"]) + ".json")
    if os.path.exists(cp): return json.load(open(cp))
    path = download_pdf(paper)
    if not path: return None
    try: text, n = text_of(path)
    except Exception: return None
    body = text.split("\nReferences\n")[0] if "\nReferences\n" in text else text
    sec = sections(text)
    gaps = list(dict.fromkeys(" ".join(m.group(0).split()) for m in GAPRE.finditer(body)))[:15]
    d = {"n_pages": n, "sections": {k: v[:9000] for k, v in sec.items() if k != "references"}, "gap_sentences": gaps,
         "middle": body[len(body)//4: len(body)//4 + 7000],"head": text[:3000], "tail": body[-4000:], "ok_intro": "intro" in sec, "ok_concl": "conclusion" in sec}
    json.dump(d, open(cp, "w"), ensure_ascii=False)
    return d
