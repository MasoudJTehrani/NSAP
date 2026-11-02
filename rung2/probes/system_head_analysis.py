"""What happened to the DEPLOYED scene graph, per attacked head.

The model-level analysis compares clean and patched graphs on the SAME frame. In
deployment we cannot do that after step 0, because the two conditions take different
actions and end up looking at different views. So instead of pairing frames we compare
DISTRIBUTIONS: over every logged step of a condition, how often does the target fact
appear, and what does the target object look like when it does not.
"""
import json, glob, os, re, sys, collections

OBJ = re.compile(r"\(\s*([0-9.]+)\s*,\s*'([^']+)'\s*\)(?:\s*\[([^\]]*)\])?")
REL = re.compile(r"\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)")

def parse(sg):
    sg = str(sg or "")
    objs = [(float(c), l.strip().lower(), [a.strip().lower() for a in (at or "").split(",") if a.strip()])
            for c, l, at in OBJ.findall(sg)]
    part = sg.split("Relations:", 1)[1] if "Relations:" in sg else ""
    rels = [(a.strip().lower(), r.strip().lower(), b.strip().lower()) for a, r, b in REL.findall(part)]
    return objs, rels

def analyse(root, head, cond):
    pat = "attr_*" if head == "attribute" else "rel_*"
    steps = empty = 0
    tv_grounded = disp = 0
    labels = collections.Counter()
    pot_edges = collections.Counter(); pot_on = 0; pot_node = 0
    for f in glob.glob(f"{root}/*/{cond}/{pat}/steps.jsonl"):
        for line in open(f):
            if not line.strip(): continue
            sg = json.loads(line).get("scene_graph_text")
            steps += 1
            s = str(sg or "")
            if "no scene graph" in s or not s.strip():
                empty += 1; continue
            objs, rels = parse(s)
            for _c, lab, attrs in objs:
                labels[lab] += 1
                if head == "attribute" and lab == "television":
                    tv_grounded += 1
                    if "displaying" in attrs: disp += 1
                if head == "relation" and lab == "pot":
                    pot_node += 1
            if head == "relation":
                for s_, r_, o_ in rels:
                    if s_ == "pot" and r_ == "on":
                        pot_on += 1; pot_edges[o_] += 1
    return dict(steps=steps, empty=empty, labels=labels, tv=tv_grounded, disp=disp,
                pot_node=pot_node, pot_on=pot_on, pot_edges=pot_edges)

def main():
    root = sys.argv[1]
    print(f"### {os.path.basename(root)}\n")
    for head in ("attribute", "relation"):
        print(f"=== {head.upper()} episodes ===")
        res = {c: analyse(root, head, c) for c in ("esca_benign", "esca_patch")}
        for c, r in res.items():
            if not r["steps"]: continue
            print(f"  {c:12s} steps={r['steps']:>4}  empty SG={r['empty']:>4} ({r['empty']/r['steps']:.0%})")
            if head == "attribute":
                print(f"      television grounded : {r['tv']:>4} ({r['tv']/r['steps']:.0%} of steps)")
                print(f"      ...carrying [displaying]: {r['disp']:>4} ({r['disp']/max(r['tv'],1):.0%} of those)")
            else:
                print(f"      pot node present    : {r['pot_node']:>4} ({r['pot_node']/r['steps']:.0%})")
                print(f"      (pot, on, X) edges  : {r['pot_on']:>4}")
                print(f"      X distribution      : {dict(r['pot_edges'].most_common(6))}")
            print(f"      top labels in graph : {dict(r['labels'].most_common(5))}")
        b, p = res["esca_benign"], res["esca_patch"]
        if head == "attribute" and b["tv"] and p["tv"]:
            print(f"  --> [displaying] rate: benign {b['disp']/b['tv']:.0%} -> patched {p['disp']/p['tv']:.0%}")
        if head == "relation" and b["pot_on"] and p["pot_on"]:
            bc = sum(v for k, v in b["pot_edges"].items() if "counter" in k) / b["pot_on"]
            pc = sum(v for k, v in p["pot_edges"].items() if "counter" in k) / p["pot_on"]
            print(f"  --> share of (pot,on,X) where X is a counter: benign {bc:.0%} -> patched {pc:.0%}")
        print()

if __name__ == "__main__":
    main()
