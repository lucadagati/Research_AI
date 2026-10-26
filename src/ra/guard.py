"""cite_guard (P2): ogni [Pk] deve esistere nel corpus del run; le citazioni testuali devono comparire nelle fonti."""
import re
from .parse import parse
KEY = re.compile(r"\[(P\d+(?:\s*[,;]\s*P\d+)*)\]")
def _norm(s): return re.sub(r"\W+", " ", (s or "").lower()).strip()

def walk(o, f):
    if isinstance(o, str): return f(o)
    if isinstance(o, list): return [walk(x, f) for x in o]
    if isinstance(o, dict): return {k: walk(v, f) for k, v in o.items()}
    return o

def clean(obj, valid, stats):
    """rimuove dalle stringhe le chiavi [Pk] non valide e conta"""
    def f(s):
        def rep(m):
            ks = [k.strip() for k in re.split(r"[,;]", m.group(1))]
            ok = [k for k in ks if k in valid]
            stats["total"] += len(ks); stats["invalid"] += len(ks) - len(ok)
            return f"[{', '.join(ok)}]" if ok else ""
        return KEY.sub(rep, s)
    return walk(obj, f)

def valid_ids(ids, valid, stats, label=""):
    ok = [i for i in ids if i in valid]
    stats["total"] += len(ids); stats["invalid"] += len(ids) - len(ok)
    return ok

def quote_ok(quote, paper):
    q = _norm(quote)
    if len(q) < 15: return False
    srcs = [paper.get("abstract", ""), paper.get("title", "")]
    d = parse(paper) if paper.get("pdf_url") else None
    if d: srcs += [d.get("head", ""), d.get("tail", ""), d.get("middle", "")] + list(d["sections"].values())
    return any(q in _norm(s) for s in srcs)
