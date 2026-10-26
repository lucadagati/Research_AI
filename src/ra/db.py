"""SQLite: papers, cards, vettori. Un paper = dict JSON con id stabile."""
import json, sqlite3, numpy as np
from .config import p

_c = None
def conn():
    global _c
    if _c is None:
        _c = sqlite3.connect(p("data/db.sqlite"), check_same_thread=False, timeout=60)
        _c.executescript("""
        CREATE TABLE IF NOT EXISTS papers(id TEXT PRIMARY KEY, j TEXT);
        CREATE TABLE IF NOT EXISTS cards(id TEXT, level TEXT, j TEXT, PRIMARY KEY(id, level));
        CREATE TABLE IF NOT EXISTS vec(id TEXT, kind TEXT, v BLOB, PRIMARY KEY(id, kind));
        CREATE TABLE IF NOT EXISTS edges(src TEXT, dst TEXT, PRIMARY KEY(src, dst));
        """)
    return _c

def put_paper(pp):
    conn().execute("INSERT OR REPLACE INTO papers VALUES(?,?)", (pp["id"], json.dumps(pp, ensure_ascii=False))); conn().commit()
def get_paper(i):
    r = conn().execute("SELECT j FROM papers WHERE id=?", (i,)).fetchone(); return json.loads(r[0]) if r else None
def all_papers():
    return [json.loads(r[0]) for r in conn().execute("SELECT j FROM papers")]
def put_card(i, level, card):
    conn().execute("INSERT OR REPLACE INTO cards VALUES(?,?,?)", (i, level, json.dumps(card, ensure_ascii=False))); conn().commit()
def get_card(i, level=None):
    for lv in ([level] if level else ["full", "abstract"]):
        r = conn().execute("SELECT j FROM cards WHERE id=? AND level=?", (i, lv)).fetchone()
        if r: return json.loads(r[0])
    return None
def put_vec(i, kind, v):
    conn().execute("INSERT OR REPLACE INTO vec VALUES(?,?,?)", (i, kind, np.asarray(v, "float32").tobytes())); conn().commit()
def get_vecs(kind):
    rows = conn().execute("SELECT id, v FROM vec WHERE kind=?", (kind,)).fetchall()
    if not rows: return [], np.zeros((0, 768), "float32")
    return [r[0] for r in rows], np.stack([np.frombuffer(r[1], "float32") for r in rows])
def add_edges(src, dsts):
    conn().executemany("INSERT OR IGNORE INTO edges VALUES(?,?)", [(src, d) for d in dsts]); conn().commit()
