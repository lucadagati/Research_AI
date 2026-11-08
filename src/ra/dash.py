"""Dashboard locale (solo libreria standard): invia idee, segui i run, leggi il report, dai il voto (E-D)."""
import base64, hmac, json, os, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from .config import ROOT, p

RATINGS = p("eval", "ratings.jsonl")
QUEUE, QLOCK = [], threading.Lock()

def _creds():
    """utente:password da .dash_auth (fuori da git, chmod 600) oppure da RA_DASH_USER / RA_DASH_PASS"""
    if os.environ.get("RA_DASH_USER") and os.environ.get("RA_DASH_PASS"):
        return os.environ["RA_DASH_USER"], os.environ["RA_DASH_PASS"]
    f = p(".dash_auth")
    if os.path.exists(f):
        u, _, pw = open(f).read().strip().partition(":")
        if u and pw: return u, pw
    raise SystemExit("Credenziali mancanti: crea .dash_auth con 'utente:password' oppure imposta RA_DASH_USER e RA_DASH_PASS")

CREDS = None
STAGES = [("F5", "analisi idea"), ("F6", "recupero"), ("F3", "schede"), ("F7 novit", "novità"), ("F7 gap", "gap"), ("F8", "direzioni"),
          ("F9", "revisore"), ("F10", "piano"), ("Fatto", "completato")]

def alive(pid):
    """vivo davvero: uno zombie (processo morto non ancora raccolto) non conta"""
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
        return st != "Z"
    except OSError: return False

def run_info(rid):
    rd = p("runs", rid); info = {"id": rid, "status": "?", "stage": "", "idea": "", "mtime": os.path.getmtime(rd)}
    try: info["idea"] = open(f"{rd}/idea.txt").read()[:300]
    except OSError: pass
    log = ""
    try: log = open(f"{rd}/run.log", errors="ignore").read()
    except OSError: pass
    log = log.split("--- ripresa del run ---")[-1]
    if os.path.exists(f"{rd}/report.md"): info["status"] = "completato"
    else:
        pid = None
        try: pid = int(open(f"{rd}/pid").read())
        except Exception: pass
        if os.path.exists(f"{rd}/queued"): info["status"] = "in coda"
        elif pid and alive(pid): info["status"] = "in corso"
        elif "Traceback" in log: info["status"] = "fallito"
        else: info["status"] = "interrotto"
        if info["status"] == "fallito":
            err = [l for l in log.splitlines() if re.match(r"[\w.]*(Error|Exception)\b", l.strip())]
            info["error"] = err[-1][:300] if err else "errore non riconosciuto (vedi log)"
    for k, name in STAGES:
        if k in log: info["stage"] = name
    info["log_tail"] = "\n".join(l for l in log.splitlines() if "[harvest]" not in l)[-1200:]
    return info

def runs():
    return sorted((run_info(r) for r in os.listdir(p("runs")) if os.path.isdir(p("runs", r))), key=lambda x: -x["mtime"])

def ratings():
    try: return [json.loads(l) for l in open(RATINGS)]
    except OSError: return []

def summary():
    out = {"eval": {}, "ratings": {}}
    for w in ("cutoff", "ld", "retrieval"):
        f = p("eval", f"results_{w}.json")
        if os.path.exists(f): out["eval"][w] = json.load(open(f)).get("summary")
    rs = ratings()
    if rs:
        out["ratings"] = {k: round(sum(r[k] for r in rs) / len(rs), 2) for k in ("utilita", "novita", "fattibilita")}; out["ratings"]["n"] = len(rs)
    return out

def detail(rid):
    rd = p("runs", rid); d = {"info": run_info(rid)}
    for name in ("directions", "guard", "novelty"):
        try: d[name] = json.load(open(f"{rd}/{name}.json"))
        except Exception: pass
    try: d["report"] = open(f"{rd}/report.md").read()
    except OSError: pass
    d["ratings"] = [r for r in ratings() if r["run"] == rid]
    return d

RID = re.compile(r"[\w.-]+")
def _rid(x):
    """id di run valido (niente path traversal) e cartella esistente"""
    x = str(x)
    if not RID.fullmatch(x) or x in (".", "..") or not os.path.isdir(p("runs", x)): raise ValueError("run inesistente")
    return x

def start(form):
    idea = form.get("idea", "").strip()
    if not idea: return {"error": "idea vuota"}
    rid = time.strftime("%Y%m%d-%H%M%S"); rd = p("runs", rid); os.makedirs(rd)
    cmd = [os.path.join(ROOT, "ra"), "idea", idea, "--run-id", rid]
    if form.get("fast"): cmd.append("--fast")
    if form.get("venue"): cmd += ["--venue", form["venue"]]
    if form.get("weeks"): cmd += ["--weeks", str(int(form["weeks"]))]
    if form.get("cutoff"): cmd += ["--cutoff", form["cutoff"]]
    open(f"{rd}/idea.txt", "w").write(idea)
    return enqueue(cmd, rid)

def busy():
    """c'è già un processo ra (idea/eval/ingest) vivo? Un solo run alla volta: la GPU è condivisa e il tetto è il 50%."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
        except OSError: continue
        if st != "Z" and b"-m" in args and b"ra" in args and any(a in args for a in (b"idea", b"eval", b"ingest")): return True
    return False

def enqueue(cmd, rid, mode="w"):
    rd = p("runs", rid)
    open(f"{rd}/queued", "w").write("1")
    with QLOCK: QUEUE.append((cmd, rid, mode))
    return {"id": rid, "queued": True}

def scheduler():
    while True:
        time.sleep(5)
        try:
            with QLOCK:
                if QUEUE and not busy():
                    cmd, rid, mode = QUEUE.pop(0)
                    try: os.remove(p("runs", rid, "queued"))
                    except OSError: pass
                    launch(cmd, rid, mode)
        except Exception as e: print("[sched]", e, flush=True)

def launch(cmd, rid, mode="w"):
    rd = p("runs", rid)
    proc = subprocess.Popen(cmd, stdout=open(f"{rd}/run.log", mode), stderr=subprocess.STDOUT, start_new_session=True)
    open(f"{rd}/pid", "w").write(str(proc.pid))
    threading.Thread(target=proc.wait, daemon=True).start()     # raccoglie il processo: niente zombie
    return {"id": rid}

def replan(form):
    """nuovo run che riusa corpus/novità/gap/direzioni del run indicato e rigenera solo il piano per la direzione N"""
    import shutil
    src, n = _rid(form["run"]), int(form["direction"]); sd = p("runs", src)
    if not os.path.exists(f"{sd}/directions.json"): return {"error": "le direzioni non sono ancora pronte"}
    rid = f"{src}-d{n}"; rd = p("runs", rid)
    if os.path.exists(rd): return {"error": f"esiste già {rid}", "id": rid}
    os.makedirs(rd)
    for f in ("idea.txt", "meta.json", "spec.json", "corpus.json", "novelty.json", "gaps.json", "directions.json"):
        if os.path.exists(f"{sd}/{f}"): shutil.copy(f"{sd}/{f}", rd)
    meta = json.load(open(f"{rd}/meta.json")) if os.path.exists(f"{rd}/meta.json") else {}
    cmd = [os.path.join(ROOT, "ra"), "idea", open(f"{rd}/idea.txt").read(), "--run-id", rid, "--direction", str(n)]
    if meta.get("venue"): cmd += ["--venue", meta["venue"]]
    if meta.get("weeks"): cmd += ["--weeks", str(meta["weeks"])]
    return enqueue(cmd, rid)

def resume(form):
    rid = _rid(form["run"]); rd = p("runs", rid)
    if run_info(rid)["status"] in ("in corso", "in coda"): return {"error": "già in corso o in coda"}
    meta = {}
    try: meta = json.load(open(f"{rd}/meta.json"))
    except Exception: pass
    cmd = [os.path.join(ROOT, "ra"), "idea", open(f"{rd}/idea.txt").read(), "--run-id", rid]
    if meta.get("fast"): cmd.append("--fast")
    if meta.get("venue"): cmd += ["--venue", meta["venue"]]
    if meta.get("weeks"): cmd += ["--weeks", str(meta["weeks"])]
    if meta.get("cutoff"): cmd += ["--cutoff", meta["cutoff"]]
    open(f"{rd}/run.log", "a").write("\n--- ripresa del run ---\n")
    return enqueue(cmd, rid, "a")

def rate(form):
    rec = {"run": _rid(form["run"]), "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "comment": form.get("comment", "")[:2000]}
    for k in ("utilita", "novita", "fattibilita"): rec[k] = max(1, min(5, int(form[k])))
    rec["direzioni"] = form.get("direzioni", {})
    open(RATINGS, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}

PAGE = open(os.path.join(os.path.dirname(__file__), "dash.html")).read() if os.path.exists(os.path.join(os.path.dirname(__file__), "dash.html")) else "dash.html mancante"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def authed(self):
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                u, _, pw = base64.b64decode(h[6:]).decode().partition(":")
                ok_u = hmac.compare_digest(u.encode(), CREDS[0].encode()); ok_p = hmac.compare_digest(pw.encode(), CREDS[1].encode())
                if ok_u and ok_p: return True
            except Exception: pass
        time.sleep(1.5)                      # frena i tentativi a raffica
        b = b"Autenticazione richiesta"
        self.send_response(401); self.send_header("WWW-Authenticate", 'Basic realm="Research AI", charset="UTF-8"')
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        return False
    def send(self, obj, code=200, ctype="application/json"):
        b = (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if not self.authed(): return
        u = urlparse(self.path)
        if u.path == "/": return self.send(open(os.path.join(os.path.dirname(__file__), "dash.html")).read(), ctype="text/html")
        if u.path == "/api/runs": return self.send({"runs": runs(), "summary": summary()})
        m = re.fullmatch(r"/api/run/([\w.-]+)", u.path)
        if m and os.path.isdir(p("runs", m.group(1))): return self.send(detail(m.group(1)))
        self.send({"error": "non trovato"}, 404)
    def do_POST(self):
        if not self.authed(): return
        try:
            n = int(self.headers.get("Content-Length", 0)); form = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/start": return self.send(start(form))
            if self.path == "/api/rate": return self.send(rate(form))
            if self.path == "/api/resume": return self.send(resume(form))
            if self.path == "/api/replan": return self.send(replan(form))
        except Exception as e: return self.send({"error": str(e)}, 400)
        self.send({"error": "non trovato"}, 404)

def serve(port=8765, host="0.0.0.0"):
    global CREDS
    CREDS = _creds()
    threading.Thread(target=scheduler, daemon=True).start()
    print(f"Dashboard: http://{host}:{port}  (utente: {CREDS[0]})", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()
