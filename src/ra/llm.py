"""Wrapper unico LLM locale (Ollama): chat, JSON validato con pydantic e retry, embedding."""
import fcntl, json, os, re, subprocess, threading, time, requests, numpy as np
from pydantic import BaseModel, ValidationError
from .config import CFG, p as _path

L = CFG["llm"]

# ---------------- tetto di memoria GPU (default 50%) ----------------
# Regola: la somma della VRAM dei MIEI modelli caricati in Ollama non supera gpu_budget_frac * VRAM totale.
# Prima di usare un modello non caricato, scarico i miei modelli più grandi finché ci sta. Non tocco i modelli di altri.
_total = None
_ctx_override = {}
_lock = threading.Lock()

def _norm(m): return m if ":" in m else m + ":latest"
def mine(): return {_norm(L[k]) for k in ("reason", "extract", "extract_fast", "embed")}

def gpu_total():
    global _total
    if _total is None:
        try: _total = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).split()[0]) * 2**20
        except Exception: _total = 46 * 2**30
    return _total
def gpu_cap(): return int(L.get("gpu_budget_frac", 0.5) * gpu_total())

def loaded():
    try: return {m["name"]: m["size_vram"] for m in requests.get(L["host"] + "/api/ps", timeout=10).json()["models"]}
    except Exception: return {}

def est_need(model):
    """stima conservativa della VRAM: file del modello + margine per la cache KV"""
    try:
        for m in requests.get(L["host"] + "/api/tags", timeout=10).json()["models"]:
            if m["name"] == model: return m["size"] + 3 * 2**30
    except Exception: pass
    return 16 * 2**30

def choose_unloads(ours, need, cap):
    """quali dei miei modelli già caricati scaricare perché (loro + nuovo) stia nel tetto: prima i più grandi"""
    out, tot = [], sum(ours.values())
    for k in sorted(ours, key=lambda k: -ours[k]):
        if tot + need <= cap: break
        out.append(k); tot -= ours[k]
    return out

def unload(model):
    for url, body in ((L["host"] + "/api/generate", {"model": model, "keep_alive": 0}), (L["host"] + "/api/embed", {"model": model, "input": [], "keep_alive": 0})):
        try:
            if requests.post(url, json=body, timeout=30).ok: return
        except Exception: pass

def ensure_budget(model):
    model = _norm(model)
    with _lock, open(_path("data", ".gpu.lock"), "w") as fl:
        fcntl.flock(fl, fcntl.LOCK_EX)      # serializza anche tra processi diversi (dashboard, eval, ingest)
        ld = loaded()
        if model in ld: return
        ours = {k: v for k, v in ld.items() if k in mine()}
        for k in choose_unloads(ours, est_need(model), gpu_cap()):
            print(f"[gpu] scarico {k} per restare sotto il {int(L.get('gpu_budget_frac', 0.5) * 100)}% della GPU", flush=True)
            unload(k)
        time.sleep(1)

WARNINGS = []

def check_after(model):
    """Se questo modello DA SOLO supera il tetto, riduco un po' il contesto (mai sotto min_ctx: un prompt troncato rovina l'output).
    Non conto i modelli degli altri: la somma è già gestita da ensure_budget."""
    model = _norm(model); own = loaded().get(model, 0)
    cur = _ctx_override.get(model, L["num_ctx"]); floor = L.get("min_ctx", 12288)
    if own > gpu_cap() and cur > floor:
        _ctx_override[model] = max(floor, cur * 3 // 4)
        print(f"[gpu] {model} da solo oltre il tetto: contesto ridotto a {_ctx_override[model]}", flush=True)
        unload(model)

def chat(model, prompt, system=None, json_mode=False, temperature=0.2, num_ctx=None, timeout=900):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    ensure_budget(model)
    body = {"model": model, "messages": msgs, "stream": False, "keep_alive": L.get("keep_alive", "3m"),
            "options": {"temperature": temperature, "num_ctx": num_ctx or _ctx_override.get(_norm(model), L["num_ctx"])}}
    if json_mode: body["format"] = "json"
    for i in range(3):
        try:
            r = requests.post(L["host"] + "/api/chat", json=body, timeout=timeout)
            r.raise_for_status()
            j = r.json(); ctx = body["options"]["num_ctx"]
            if j.get("prompt_eval_count", 0) >= ctx - 8:     # Ollama tronca in silenzio: lo segnalo
                msg = f"[ctx] prompt probabilmente TRONCATO ({j['prompt_eval_count']} token, contesto {ctx}) con {model}"
                print(msg, flush=True); WARNINGS.append(msg)
            check_after(model)
            return j["message"]["content"]
        except Exception as e:
            err = e; time.sleep(5 * (i + 1))
    raise err

def _extract_json(txt):
    txt = txt.strip()
    try: return json.loads(txt)
    except Exception: pass
    m = re.search(r"\{.*\}", txt, re.S)
    if m: return json.loads(m.group(0))
    raise ValueError("nessun JSON")

def complete(prompt, schema, model=None, system=None, retries=3, temperature=0.2):
    """Chiede JSON conforme a `schema` (pydantic). Ritenta passando l'errore al modello."""
    model = model or L["reason"]
    hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    full = f"{prompt}\n\nRispondi SOLO con un oggetto JSON valido conforme a questo JSON Schema (nessun testo fuori dal JSON):\n{hint}"
    last = ""
    for i in range(retries):
        out = chat(model, full + (f"\n\nIl tentativo precedente era non valido ({last}). Correggi." if last else ""),
                   system=system, json_mode=True, temperature=temperature)
        try:
            return schema.model_validate(_extract_json(out))
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last = str(e)[:300]
    raise RuntimeError(f"JSON non valido dopo {retries} tentativi: {last}")

def embed(texts, kind="document"):
    pre = "search_document: " if kind == "document" else "search_query: "
    out = []
    for i in range(0, len(texts), 32):
        ensure_budget(L["embed"])
        body = {"model": L["embed"], "keep_alive": L.get("keep_alive", "3m"), "input": [pre + t[:6000] for t in texts[i:i + 32]]}
        for k in range(6):      # il runner di Ollama può crollare al caricamento (GPU condivisa): si riprova con attesa crescente
            try:
                r = requests.post(L["host"] + "/api/embed", json=body, timeout=300)
                r.raise_for_status(); break
            except Exception as e:
                if k == 5: raise
                print(f"[embed] tentativo {k+1} fallito ({str(e)[:60]}), riprovo", flush=True); time.sleep(10 * (k + 1))
        out += r.json()["embeddings"]
    a = np.array(out, dtype="float32")
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
