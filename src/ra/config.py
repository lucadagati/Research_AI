import os, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def p(*a): return os.path.join(ROOT, *a)
def load(name="config.yaml"): return yaml.safe_load(open(p(name)))
CFG = load()
RES = load("resources.yaml")
for d in ("data/cache", "data/pdf", "data/text", "data/cards", "runs"):
    os.makedirs(p(d), exist_ok=True)
