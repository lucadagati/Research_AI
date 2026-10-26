import argparse, sys
def main():
    ap = argparse.ArgumentParser(prog="ra"); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("idea"); a.add_argument("text"); a.add_argument("--cutoff"); a.add_argument("--venue")
    a.add_argument("--weeks", type=int); a.add_argument("--fast", action="store_true"); a.add_argument("--direction", type=int); a.add_argument("--run-id")
    i = sub.add_parser("ingest"); i.add_argument("--topics", type=int, default=0); i.add_argument("--no-cards", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("which", choices=["cutoff", "ld", "retrieval"]); e.add_argument("--n", type=int, default=5)
    c = sub.add_parser("chat"); c.add_argument("run")
    d = sub.add_parser("dash"); d.add_argument("--port", type=int, default=8765); d.add_argument("--host", default="0.0.0.0")
    x = ap.parse_args()
    if x.cmd == "idea":
        from .run import run_idea
        run_idea(open(x.text).read() if x.text.endswith(".txt") else x.text, x.cutoff, x.venue, x.weeks, x.fast, x.direction, x.run_id)
    elif x.cmd == "ingest":
        from .ingest import ingest; ingest(x.topics, not x.no_cards)
    elif x.cmd == "eval":
        from .evaluate import evaluate, eval_retrieval; (eval_retrieval(x.n) if x.which == 'retrieval' else evaluate(x.which, x.n))
    elif x.cmd == "dash":
        from .dash import serve; serve(x.port, x.host)
    elif x.cmd == "chat":
        from .chat import chat; chat(x.run)
if __name__ == "__main__": main()
