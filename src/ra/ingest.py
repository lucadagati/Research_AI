"""F1-F4: base locale offline nei campi del profilo (pilastri + frontiera), schede, indice, cluster, pratiche."""
import json
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
from . import db, harvest as h, llm, cards
from .config import CFG, p

def ingest(max_topics=0, with_cards=True, log=print):
    topics = CFG["topics"][:max_topics or None]
    for t in topics:
        q, niche = t["q"], t["niche"]
        pool = h.arxiv_search(q, CFG["max_per_topic"], sort="relevance")
        pool += h.arxiv_search(q, 200, sort="submittedDate")           # recenti, per la frontiera
        pool += h.ieee_oa(q, CFG["ieee_pages"], CFG["niche_min_citations"] if niche else CFG["min_citations"], CFG["from_year"])
        ids = list(dict.fromkeys("ARXIV:" + d["arxiv_id"] for d in pool if d["source"] == "arxiv"))
        info = dict(zip(ids, h.s2_batch(ids)))
        for d in pool:
            r = info.get("ARXIV:" + (d["arxiv_id"] or ""))
            if r: d["citations"], d["influential"], d["s2_id"], d["venue"] = r["citations"], r["influential"], r["s2_id"], d["venue"] or r["venue"]
        pool = [d for d in h.merge(pool) if d.get("abstract") and (d.get("year") or 0) >= CFG["from_year"]]
        for d in pool: h.label(d, niche); d["topic"] = q
        h.frontier_rank(pool)
        keep = [d for d in pool if d["pillar"] or d["frontier"]]
        for d in keep: db.put_paper(d)
        log(f"{q}: {len(pool)} candidati → {sum(d['pillar'] for d in keep)} pilastri, {sum(d['frontier'] for d in keep)} frontiera")
    papers = db.all_papers(); log(f"base: {len(papers)} paper")
    # indice vettoriale (F4)
    have = set(db.get_vecs("doc")[0]); todo = [d for d in papers if d["id"] not in have]
    for i in range(0, len(todo), 64):
        b = todo[i:i + 64]; V = llm.embed([d["title"] + ". " + d["abstract"] for d in b])
        for d, v in zip(b, V): db.put_vec(d["id"], "doc", v)
    if with_cards:   # schede da abstract per tutti, full text per i pilastri più citati
        def job(d): return cards.ensure_card(d, full=False)
        with ThreadPoolExecutor(CFG["llm"]["parallel"]) as ex: list(ex.map(job, [d for d in papers if not db.get_card(d["id"])]))
    cluster(log)

def cluster(log=print):
    ids, M = db.get_vecs("doc")
    if len(ids) < 20: return
    k = max(4, min(30, len(ids) // 25)); lab = KMeans(k, n_init=4, random_state=0).fit_predict(M)
    out = {}
    for c in range(k):
        mem = [db.get_paper(ids[i]) for i in np.where(lab == c)[0]]
        cc = Counter(); [cc.update(x.lower() for x in (db.get_card(d["id"]) or {}).get("baselines", [])) for d in mem]
        dd = Counter(); [dd.update(x.lower() for x in (db.get_card(d["id"]) or {}).get("datasets", [])) for d in mem]
        mm = Counter(); [mm.update(x.lower() for x in (db.get_card(d["id"]) or {}).get("metrics", [])) for d in mem]
        out[c] = {"size": len(mem), "mean_year": round(float(np.mean([d["year"] for d in mem])), 1),
                  "frontier_share": round(sum(d["frontier"] for d in mem) / len(mem), 2),
                  "examples": [d["title"] for d in sorted(mem, key=lambda d: -d["citations"])[:4]],
                  "baselines": cc.most_common(8), "datasets": dd.most_common(8), "metrics": mm.most_common(8)}
    json.dump(out, open(p("data/clusters.json"), "w"), ensure_ascii=False, indent=1)
    log(f"cluster: {k}")
