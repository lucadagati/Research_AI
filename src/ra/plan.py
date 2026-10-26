"""F10 piano di ricerca completo (tre chiamate strutturate) + tabelle di pratica."""
import re
from collections import Counter
from typing import List
from pydantic import BaseModel
from . import llm
from .analysis import brief, IT
from .config import CFG, RES

def norm(x): return " ".join(x.lower().replace("-", " ").split())

def practice(corpus):
    cnt = {k: Counter() for k in ("baselines", "datasets", "metrics")}; venues = Counter(); ctypes = Counter()
    for d in corpus:
        c = d.get("card") or {}
        if d.get("rel", 0) < 3 and d.get("rel", 0) < 2: continue
        w = (2 if d["pillar"] else 1) * (2 if d.get("rel") == 3 else 1)
        for k in cnt:
            for x in c.get(k, []) or []: cnt[k][norm(x)] += w
        if d.get("venue"): venues[d["venue"]] += w
        if c.get("contribution_type"): ctypes[c["contribution_type"]] += 1
    top = lambda c, n=10: [f"{k} (x{v})" for k, v in c.most_common(n)]
    return {"baselines": top(cnt["baselines"], 12), "datasets": top(cnt["datasets"]), "metrics": top(cnt["metrics"]),
            "venues": top(venues, 8), "contribution_types": top(ctypes, 6)}

class Hyp(BaseModel):
    id: str
    statement: str
    metric: str
    threshold: str
class PlanA(BaseModel):
    research_questions: List[str]
    hypotheses: List[Hyp]
    contributions: List[str]
    positioning: List[str]          # una riga per lavoro vicino: [Pk] cosa fa / come ti differenzi
    method_overview: str
    design_choices: List[str]       # scelta + alternativa scartata
class Exp(BaseModel):
    id: str
    hypothesis: str
    goal: str
    procedure: str
    baselines: List[str]
    metrics: List[str]
    hardware: str
    estimated_days: int
class PlanB(BaseModel):
    baselines: List[str]
    datasets: List[str]
    metrics_primary: List[str]
    metrics_secondary: List[str]
    experiments: List[Exp]
    ablations: List[str]
    statistical_protocol: str
    threats: List[str]
class Risk(BaseModel):
    risk: str
    early_signal: str
    plan_b: str
class Week(BaseModel):
    weeks: str
    milestone: str
    exit_criterion: str
class PlanC(BaseModel):
    risks: List[Risk]
    venues: List[str]
    paper_skeleton: List[str]
    title_options: List[str]
    abstract_pattern: str
    timeline: List[Week]
    first_step: str

def make_plan(spec, direction, corpus, gp, venue=None, weeks=None, cutoff=None, idea=""):
    """genera il piano, lo fa rivedere e, se ci sono difetti GRAVI, lo rigenera una volta tenendone conto"""
    r = _make_plan(spec, direction, corpus, gp, venue, weeks, cutoff, idea)
    rv = r["review"]; grave = [i for i in rv["issues"] if i["severity"] == "alta"]
    if CFG.get("plan", {}).get("auto_revise", False) and grave and rv["verdict"] != "solido":
        fix = ("PROBLEMI GRAVI TROVATI DAL REVISORE NELLA VERSIONE PRECEDENTE: il nuovo piano NON deve ripeterli (correggili davvero, non limitarti a citarli): "
               + " | ".join(f"[{i['area']}] {i['text']}" for i in grave))
        r2 = _make_plan(spec, direction, corpus, gp, venue, weeks, cutoff, idea, fix=fix)
        r2["review"]["revised"] = True; r2["review_first"] = rv
        return annotate(r2, corpus)
    return annotate(r, corpus)

def _make_plan(spec, direction, corpus, gp, venue=None, weeks=None, cutoff=None, idea="", fix=""):
    pr = practice(corpus)
    canon = "; ".join(f"[{d['key']}] {d['title']} ({d['year']}, {d['citations']} cit)" for d in corpus if d.get("lane") == "canonical")
    orig = f"\nTESTO ORIGINALE DELL'IDEA (fa fede; se elenca hardware o vincoli, hanno PRIORITÀ su RISORSE e non vanno ignorati):\n{idea[:3500]}\n" if idea else ""
    base = f"""{orig}IDEA: {spec.problem} | metodo: {spec.proposed_method} | assunzioni: {'; '.join(spec.implicit_assumptions)}
DIREZIONE SCELTA: {direction['title']} — {direction['research_question']} | contributo: {direction['expected_contribution']} | rischio: {direction['main_risk']}
RISORSE: {RES['hours_per_week']} h/settimana; {'; '.join(RES['compute'])}; board: {'; '.join(RES['boards'])}; testbed: {'; '.join(RES['testbeds'])}; NON disponibile: {'; '.join(RES['no_access'])}
STILE AUTORE: {RES['profile']['style']}; venue già usate: {', '.join(RES['profile']['venues_seen'])}{(chr(10) + 'VENUE RICHIESTA: ' + venue) if venue else ''}{(chr(10) + 'DURATA DISPONIBILE: ' + str(weeks) + ' settimane') if weeks else ''}
PRATICA DEL CAMPO (dai paper del corpus): baseline {pr['baselines']} | dataset {pr['datasets']} | metriche {pr['metrics']} | venue {pr['venues']} | tipi di contributo {pr['contribution_types']}
CANONICI MOLTO CITATI (candidati baseline, anche se solo tangenti): {canon or 'nessuno'}
GAP: """ + "; ".join(f"[{g.type}] {g.statement}" for g in gp[:10]) + "\nCORPUS:\n" + "\n".join(brief(d) for d in corpus[:40])
    cite = " Cita i paper del corpus con [Pk] e usa SOLO chiavi esistenti. Sii specifico e operativo, niente genericità."
    scope = (" REGOLE DI AMBITO: è UN solo paper realizzabile con le ore disponibili: al massimo 2 domande di ricerca e 3 contributi, ognuno legato a un'ipotesi; "
             "l'idea centrale dell'utente è il contributo principale (non trasformare benchmark secondari in ipotesi se non sono nell'idea); "
             "le soglie numeriche (es. latenza, % di risparmio) devono derivare dai paper citati, altrimenti scrivi 'da calibrare con esperimento pilota' invece di inventarle; "
             "un metodo non può essere insieme baseline e tuo contributo.")
    scopeB = (" REGOLE: usa SOLO dataset del dominio dell'idea (non quelli di altri domini del profilo dell'autore); ogni metrica primaria deve essere verificata da almeno un'ipotesi; "
              "assegna a ogni esperimento hardware COMPATIBILE col carico (es. un Coral Edge TPU esegue solo piccole reti INT8, non modelli grandi); non inventare hardware non elencato.")
    if fix: base += "\n\n" + fix
    a = llm.complete(f"{IT}Parte A del piano di ricerca: domande di ricerca (2-3), ipotesi falsificabili con metrica e soglia numerica, contributi attesi formulati come nei pilastri, posizionamento rispetto a 5-8 lavori vicini, panoramica del metodo, scelte progettuali con alternativa scartata.{cite}{scope}\n\n{base}", PlanA)
    cut = f" (solo lavori pubblicati prima del {cutoff})" if cutoff else ""
    b = llm.complete(f"""{IT}Parte B del piano: setup sperimentale. Baseline: nomina metodi CONCRETI e riproducibili (es. 'MPC con modello RC', 'PID tarato', 'GPT-4o zero-shot'), mai etichette generiche come 'standard baseline'; includi quelle standard del campo (vedi PRATICA) e le più forti della frontiera, scartando quelle non pertinenti all'idea. Aggiungi anche le baseline CANONICHE del sotto-campo che conosci dalla letteratura anche se assenti dal corpus, marcandole '(da verificare)' e senza [Pk]; elencane 6-10 in totale, privilegiando i sistemi precedenti più forti e più citati{cut}. Dataset/benchmark standard + uno che rende la valutazione distintiva. Metriche primarie/secondarie come nel campo. Matrice di esperimenti E1..En, ognuno legato a un'ipotesi (hypothesis = id H), con procedura, baseline, metriche, hardware dalle RISORSE disponibili e giorni stimati. Ablation. Protocollo statistico (numero di seed/run, intervalli di confidenza, test, correzione confronti multipli). Minacce alla validità (interna, esterna, di costrutto) con mitigazione.{cite}{scopeB}

{base}

PARTE A GIÀ DEFINITA — ipotesi: {[h.model_dump() for h in a.hypotheses]}""", PlanB)
    days = sum(e.estimated_days for e in b.experiments); need_w = round(days * 8 / RES["hours_per_week"] + 4)
    if weeks: need_w = min(need_w, int(weeks)) if int(weeks) >= 8 else int(weeks)
    tl = (f" VINCOLO NUMERICO SULLA TIMELINE: gli esperimenti previsti sommano {days} giorni di lavoro = {days * 8} ore = {round(days * 8 / RES['hours_per_week'], 1)} settimane a {RES['hours_per_week']} h/settimana, "
          f"più circa 4 settimane di scrittura: la timeline deve coprire circa {need_w} settimane (non meno) e avere almeno {max(6, len(b.experiments) + 3)} milestone "
          f"(setup, una per esperimento, analisi, scrittura, revisione); numera le settimane in modo cumulativo.")
    c = llm.complete(f"""{IT}Parte C del piano: rischi con segnale precoce e piano B, 3 venue target (scegli tra quelle della pratica del campo e quelle già usate dall'autore, con formato/pagine), scheletro del paper (sezioni, figure e tabelle chiave), 3 opzioni di titolo secondo i pattern dei pilastri, struttura dell'abstract, timeline a milestone con criterio di uscita (go/no-go dopo esperimenti pilota; tieni conto delle ore settimanali), e il PRIMO PASSO CONCRETO da fare domani mattina (esperimento pilota di 1-2 giorni).{cite}{tl}

{base}

ESPERIMENTI PREVISTI: {[(e.id, e.goal, e.estimated_days) for e in b.experiments]}""", PlanC)
    rv = review(spec, direction, a, b, c, idea, weeks)
    return {"A": a.model_dump(), "B": b.model_dump(), "C": c.model_dump(), "practice": pr, "review": rv}


class Issue(BaseModel):
    severity: str      # alta | media | bassa
    area: str          # ambito | ipotesi | baseline | dataset | hardware | metriche | novità | tempi
    text: str
class Review(BaseModel):
    verdict: str       # solido | da_rivedere | debole
    issues: List[Issue]

def review(spec, direction, a, b, c, idea, weeks):
    """revisione di coerenza del piano (lo stesso difetto che un revisore noterebbe): viene mostrata nel report"""
    return llm.complete(f"""{IT}Sei un revisore severo del PIANO di ricerca qui sotto. Cerca difetti concreti e verificabili, non commenti generici:
- ambito: troppi contributi/domande per un solo paper o per {RES['hours_per_week']} ore/settimana{(' e ' + str(weeks) + ' settimane') if weeks else ''}
- ipotesi: soglie numeriche inventate o non motivate; ipotesi non collegate all'idea centrale; formulazioni ambigue o incoerenti (es. soglia in unità diverse)
- baseline: un metodo che è anche tuo contributo; baseline non confrontabili; manca il lavoro più vicino
- dataset: dataset di altri domini o che non sono dataset
- hardware: hardware non adatto al carico, o hardware dell'idea ignorato
- metriche: metriche primarie non verificate da nessuna ipotesi
- novità: il lavoro più vicino non è affrontato
- tempi: timeline/esperimenti non realistici
Elenca solo i difetti reali, con severità (alta|media|bassa), e dai un verdetto (solido|da_rivedere|debole).
IDEA ORIGINALE: {idea[:2500]}
DIREZIONE: {direction['title']} — {direction['research_question']}
IPOTESI: {[h.model_dump() for h in a.hypotheses]}
CONTRIBUTI: {a.contributions}
METODO: {a.method_overview[:900]}
BASELINE: {b.baselines}
DATASET: {b.datasets}
METRICHE PRIMARIE: {b.metrics_primary}
ESPERIMENTI: {[(e.id, e.hypothesis, e.hardware, e.estimated_days) for e in b.experiments]}
TIMELINE: {[(w.weeks, w.milestone) for w in c.timeline]}""", Review).model_dump()

# ---------- uniformità di lingua ----------
import re as _re
_EN = _re.compile(r"\b(the|and|with|of|to|for|will|that|from|using|set up|configure|analyze|refine|prepare|data|manuscript|pipeline|implement|design|build|evaluate|write|test|deploy|train|collect|run|results?|baseline)\b", _re.I)
_IT = _re.compile(r"\b(il|la|di|con|per|che|del|della|le|un|una|sono|nel)\b", _re.I)

def _is_english(t):
    en, it = len(_EN.findall(t)), len(_IT.findall(t))
    return len(t) > 20 and en > it and (en >= 2 or (en >= 1 and it == 0 and not _re.search(r"[àèéìòù]", t)))

class Tr(BaseModel):
    items: List[str]

def italianize(obj):
    """traduce in italiano le stringhe inglesi del piano (mantiene termini tecnici, nomi di metodi, [Pk])"""
    strs = []
    def collect(o):
        if isinstance(o, str):
            if _is_english(o): strs.append(o)
        elif isinstance(o, list): [collect(x) for x in o]
        elif isinstance(o, dict): [collect(v) for v in o.values()]
    collect(obj)
    uniq = list(dict.fromkeys(strs)); mp = {}
    for i in range(0, len(uniq), 20):
        blk = uniq[i:i + 20]
        try:
            r = llm.complete("Traduci in italiano ciascuna stringa, nello stesso ordine. Mantieni invariati nomi di metodi, dataset, sigle, numeri, formule e chiavi di citazione tipo [P3]. "
                             f"Restituisci items con {len(blk)} stringhe.\n" + "\n".join(f"{j+1}. {t}" for j, t in enumerate(blk)), Tr, model=CFG["llm"]["extract"])
            if len(r.items) == len(blk): mp.update(dict(zip(blk, r.items)))
        except Exception as e:
            print("[it]", e)
    def rep(o):
        if isinstance(o, str): return mp.get(o, o)
        if isinstance(o, list): return [rep(x) for x in o]
        if isinstance(o, dict): return {k: rep(v) for k, v in o.items()}
        return o
    return rep(obj)


def _n(x): return re.sub(r"[^a-z0-9]+", " ", re.sub(r"\bv?\d+(\.\d+)*\b", "", x.lower())).strip()

def annotate(plan, corpus):
    """controllo deterministico (le istruzioni all'LLM non bastano): dataset non presenti in nessun paper del corpus e baseline
    senza [Pk] vengono marcati 'da verificare', così un nome inventato non passa per vero"""
    known = {_n(x) for d in corpus for x in ((d.get("card") or {}).get("datasets") or []) if x}
    out_d = []
    for ds in plan["B"]["datasets"]:
        k = _n(ds)
        ok = k and any(k in z or z in k for z in known)
        out_d.append(ds if ok or "verificare" in ds or "da costruire" in ds else ds + " (non nel corpus: da verificare)")
    plan["B"]["datasets"] = out_d
    plan["B"]["baselines"] = [b if (re.search(r"\[P\d+", b) or "verificare" in b) else b + " (da verificare)" for b in plan["B"]["baselines"]]
    return plan


def feasibility(pl, weeks=None):
    """calcolo deterministico del carico: giorni di esperimenti -> ore -> settimane alle ore/settimana dell'autore,
    confrontato con la timeline del piano e con le settimane disponibili"""
    days = sum(e.get("estimated_days", 0) for e in pl["B"]["experiments"])
    hours = days * 8                                   # un 'giorno' stimato = giornata di lavoro di 8 ore
    wk_exp = hours / RES["hours_per_week"]
    wk_write = 4                                       # scrittura e revisione del manoscritto (assunzione fissa)
    need = round(wk_exp + wk_write, 1)
    ends = [int(n) for w in pl["C"]["timeline"] for n in re.findall(r"\d+", w["weeks"])]
    plan_weeks = max(ends) if ends else 0
    out = {"experiment_days": days, "hours": hours, "weeks_experiments": round(wk_exp, 1), "weeks_needed": need,
           "hours_per_week": RES["hours_per_week"], "timeline_weeks": plan_weeks, "available_weeks": weeks}
    flags = []
    if plan_weeks and plan_weeks < need * 0.8: flags.append(f"la timeline copre {plan_weeks} settimane ma il carico stimato ne richiede circa {need}")
    if weeks and need > weeks: flags.append(f"servono circa {need} settimane a {RES['hours_per_week']} h/settimana, ne hai {weeks}: ridurre l'ambito")
    if len(pl["C"]["timeline"]) < 5: flags.append(f"timeline con sole {len(pl['C']['timeline'])} milestone: troppo grossolana")
    out["flags"] = flags
    return out
