"""Raccolta paper da fonti aperte senza key: arXiv, Semantic Scholar (anonimo), Crossref, Unpaywall.
Tutte le richieste sono in cache su disco (data/cache) e con backoff."""
import hashlib, json, os, re, time, urllib.parse, xml.etree.ElementTree as ET, requests
from .config import CFG, p

MAIL = "ldagati@unime.it"
UA = {"User-Agent": f"Research_AI (mailto:{MAIL})"}
NS = {"a": "http://www.w3.org/2005/Atom"}
_last = {}
GAP = {"arxiv.org": 3.1, "api.semanticscholar.org": 1.2, "api.crossref.org": 0.3, "api.unpaywall.org": 0.2}

def _get(url, data=None, raw=False, tries=6):
    key = hashlib.sha1((url + (json.dumps(data, sort_keys=True) if data else "")).encode()).hexdigest()
    cp = p("data/cache", key + (".bin" if raw else ".json"))
    if os.path.exists(cp):
        return open(cp, "rb").read() if raw else json.load(open(cp))
    host = urllib.parse.urlparse(url).netloc; err = "429/5xx"
    for i in range(tries):
        w = GAP.get(host, 0.5) - (time.time() - _last.get(host, 0))
        if w > 0: time.sleep(w)
        _last[host] = time.time()
        try:
            r = requests.post(url, json=data, headers=UA, timeout=90) if data else requests.get(url, headers=UA, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(8 * (i + 1)); continue
            if r.status_code in (400, 404): return None
            r.raise_for_status()
            if raw:
                open(cp, "wb").write(r.content); return r.content
            out = r.json() if "json" in r.headers.get("content-type", "") or not r.text.startswith("<") else r.text
            if isinstance(out, (dict, list)): json.dump(out, open(cp, "w"))
            return out if not isinstance(out, str) else out
        except Exception as e:
            err = e; time.sleep(4 * (i + 1))
    print(f"[harvest] fallito {url[:90]}: {err}")
    return None

def pid(pp):
    if pp.get("arxiv_id"): return "arxiv:" + pp["arxiv_id"]
    if pp.get("doi"): return "doi:" + pp["doi"].lower()
    return "s2:" + pp["s2_id"]

def norm_title(t): return re.sub(r"[^a-z0-9]", "", (t or "").lower())

def blank(**kw):
    d = {"title": "", "abstract": "", "year": None, "date": None, "authors": [], "venue": "", "citations": 0,
         "influential": 0, "arxiv_id": None, "doi": None, "s2_id": None, "pdf_url": None, "oa": False,
         "source": "", "topics": []}
    d.update(kw); return d

# ---------- arXiv ----------
def arxiv_search(query, n=25, sort="relevance"):
    out = _arxiv(query, n, sort)
    words = [w for w in re.findall(r"[\w-]+", query) if w.upper() not in ("AND", "OR", "NOT")]
    while len(out) < 5 and len(words) > 2:     # query troppo stretta: toglie l'ultimo termine
        words = words[:-1]; out = _arxiv(" ".join(words), n, sort)
    return out

def _arxiv(query, n, sort):
    ws = [w for w in re.findall(r"[\w-]+", query) if w.upper() not in ("AND", "OR", "NOT")]
    terms = " AND ".join(f'all:{w}' for w in ws)
    out = []
    for start in range(0, n, 200):
        q = urllib.parse.urlencode({"search_query": terms, "start": start, "max_results": min(200, n - start), "sortBy": sort})
        txt = _get("http://export.arxiv.org/api/query?" + q, raw=True)
        if not txt: break
        es = ET.fromstring(txt).findall("a:entry", NS)
        for e in es:
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
            d = blank(arxiv_id=aid, title=" ".join(e.find("a:title", NS).text.split()),
                      abstract=" ".join(e.find("a:summary", NS).text.split()),
                      date=e.find("a:published", NS).text[:10], year=int(e.find("a:published", NS).text[:4]),
                      authors=[a.find("a:name", NS).text for a in e.findall("a:author", NS)][:8],
                      pdf_url=f"https://arxiv.org/pdf/{aid}", oa=True, source="arxiv")
            d["id"] = pid(d); out.append(d)
        if len(es) < min(200, n - start): break
    return out

# ---------- Semantic Scholar ----------
S2F = "paperId,title,abstract,year,venue,citationCount,influentialCitationCount,externalIds,openAccessPdf,publicationDate,authors"
def _s2(r):
    ex = r.get("externalIds") or {}
    d = blank(s2_id=r.get("paperId"), title=r.get("title") or "", abstract=r.get("abstract") or "", year=r.get("year"),
              date=r.get("publicationDate"), venue=r.get("venue") or "", citations=r.get("citationCount") or 0,
              influential=r.get("influentialCitationCount") or 0, arxiv_id=ex.get("ArXiv"), doi=ex.get("DOI"),
              authors=[a["name"] for a in (r.get("authors") or [])][:8], source="s2")
    oa = (r.get("openAccessPdf") or {}).get("url")
    if d["arxiv_id"]: d["pdf_url"] = f"https://arxiv.org/pdf/{d['arxiv_id']}"; d["oa"] = True
    elif oa: d["pdf_url"] = oa; d["oa"] = True
    d["id"] = pid(d) if (d["arxiv_id"] or d["doi"] or d["s2_id"]) else None
    return d if d["id"] and d["title"] else None

_s2_off = 0.0
def s2_search(query, limit=30):
    """best effort: la ricerca anonima di S2 è spesso in 429, quindi pochi tentativi"""
    global _s2_off
    if time.time() < _s2_off: return []           # circuit breaker: dopo un 429 salta la ricerca S2 per 15 minuti
    q = urllib.parse.urlencode({"query": query, "limit": min(limit, 100), "fields": S2F})
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search?" + q, tries=1)
    if r is None: _s2_off = time.time() + 900
    return [x for x in (_s2(i) for i in ((r or {}).get("data") or [])) if x]

def s2_batch(ids):
    """ids: lista di 'ARXIV:..', 'DOI:..' o paperId -> lista di record (None se assente)"""
    out = []
    for i in range(0, len(ids), 400):
        r = _get("https://api.semanticscholar.org/graph/v1/paper/batch?fields=" + S2F, data={"ids": ids[i:i + 400]})
        out += [(_s2(x) if x else None) for x in (r or [None] * len(ids[i:i + 400]))]
    return out

def s2_neighbors(paper, kind="references", limit=60):
    key = f"ARXIV:{paper['arxiv_id']}" if paper.get("arxiv_id") else (f"DOI:{paper['doi']}" if paper.get("doi") else paper.get("s2_id"))
    if not key: return []
    r = _get(f"https://api.semanticscholar.org/graph/v1/paper/{key}/{kind}?fields={S2F}&limit={limit}")
    k = "citedPaper" if kind == "references" else "citingPaper"
    return [x for x in (_s2(i.get(k) or {}) for i in ((r or {}).get("data") or [])) if x]

# ---------- Crossref + Unpaywall (IEEE OA) ----------
def ieee_oa(query, pages=1, min_cit=100, from_year=2012, prefix="10.1109", rank="is-referenced-by-count", only_oa=True):
    """Crossref (+Unpaywall). prefix=None -> tutti gli editori; rank="relevance" per la ricerca per idea."""
    out = []
    for pg in range(pages):
        q = urllib.parse.urlencode({"query.bibliographic": query, "rows": 100, "offset": pg * 100,
              "filter": (f"prefix:{prefix}," if prefix else "") + f"from-pub-date:{from_year},type:journal-article", "sort": rank, "order": "desc",
              "select": "DOI,title,is-referenced-by-count,container-title,issued,abstract", "mailto": MAIL})
        r = _get("https://api.crossref.org/works?" + q)
        items = ((r or {}).get("message") or {}).get("items", [])
        for it in items:
            if it["is-referenced-by-count"] < min_cit or not it.get("title") or not it.get("DOI"): continue
            u = _get(f"https://api.unpaywall.org/v2/{it['DOI']}?email={MAIL}")
            loc = (u or {}).get("best_oa_location")
            if not (u or {}).get("is_oa") or not loc: continue
            d = blank(doi=it["DOI"].lower(), title=(it.get("title") or [""])[0], citations=it["is-referenced-by-count"],
                      venue=(it.get("container-title") or [""])[0], year=it["issued"]["date-parts"][0][0],
                      abstract=re.sub(r"<[^>]+>", "", it.get("abstract", "")), pdf_url=loc.get("url_for_pdf"), oa=True, source="ieee_oa")
            d["id"] = pid(d); out.append(d)
        if len(items) < 100 or (rank != "relevance" and items[-1]["is-referenced-by-count"] < min_cit): break
    return out

# ---------- normalizzazione, etichette, PDF ----------
def merge(pool):
    """Deduplica per id, DOI e titolo normalizzato; preferisce il record con più citazioni e arXiv."""
    by_t, out = {}, {}
    for d in pool:
        t = norm_title(d["title"])
        k = by_t.get(t) or d["id"]
        if k in out:
            o = out[k]
            for f in ("arxiv_id", "doi", "s2_id", "pdf_url", "venue", "abstract", "date"):
                if not o.get(f) and d.get(f): o[f] = d[f]
            o["citations"] = max(o["citations"], d["citations"]); o["oa"] = o["oa"] or d["oa"]
            if o.get("arxiv_id"): o["id"] = "arxiv:" + o["arxiv_id"]
        else:
            out[k] = dict(d); by_t[t] = k
    res = {}
    for d in out.values(): res[d["id"]] = d
    return list(res.values())

def label(d, niche=False):
    age = max(1, CFG["current_year"] - (d.get("year") or CFG["current_year"]) + 1)
    d["cit_rate"] = round((d.get("citations") or 0) / age, 2)
    th = CFG["niche_min_citations"] if niche else CFG["min_citations"]
    d["pillar"] = (d.get("citations") or 0) >= th
    d["frontier"] = bool((d.get("year") or 0) >= CFG["frontier_from_year"])
    return d

def frontier_rank(papers):
    """tra i recenti tiene i più forti per cit/anno (percentile >= 60 nel gruppo)"""
    rec = sorted([d for d in papers if d["frontier"]], key=lambda d: d["cit_rate"])
    if not rec: return papers
    cut = rec[int(len(rec) * 0.4)]["cit_rate"]
    for d in rec: d["frontier"] = d["cit_rate"] >= cut
    return papers

def download_pdf(d):
    if not d.get("pdf_url"): return None
    dst = p("data/pdf", re.sub(r"[^\w.-]", "_", d["id"]) + ".pdf")
    if os.path.exists(dst): return dst
    b = _get(d["pdf_url"], raw=True, tries=2)
    if b and b[:4] == b"%PDF":
        open(dst, "wb").write(b); return dst
    return None
