#!/usr/bin/env python
"""Read the per-step logs and answer the two questions that decide whether a
system-level attack can work at all:

  Q1  Did the patch actually change the scene graph the PLANNER saw?
      (compare the SG text of esca_benign vs esca_patch, step by step)
  Q2  Did the planner's behaviour change as a result?
      (compare the chosen actions, and the outcome)

If Q1 is "no", the attack never reached the planner: fix grounding or injection.
If Q1 is "yes" and Q2 is "no", the planner is ignoring the symbol and reading the
image instead: fix the TASK (make the fact the only way to resolve the instruction).

  python rung2/analyze_steps.py output/rung2/staged
  python rung2/analyze_steps.py output/rung2/ebnav --episode ep7_Toaster
"""
import argparse
import json
import os
import re


def load_steps(ep_dir):
    p = os.path.join(ep_dir, "steps.jsonl")
    if not os.path.isfile(p):
        return None
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def find_episodes(root):
    """-> {(rep, eval_set, episode): {condition: dir}}"""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if "steps.jsonl" not in filenames:
            continue
        rel = os.path.relpath(dirpath, root).split(os.sep)
        ep = rel[-1]
        cond = next((c for c in rel if c in
                     ("base_benign", "esca_benign", "esca_patch", "lang_benign", "lang_patch")), "?")
        rep = next((c for c in rel if re.fullmatch(r"rep\d+", c)), "")
        eset = next((c for c in rel if c in ("long_horizon", "visual_appearance")), "")
        found.setdefault((rep, eset, ep), {})[cond] = dirpath
    return found


OBJ_RE = re.compile(r"\(\s*([0-9.]+)\s*,\s*'([^']+)'\s*\)(?:\s*\[([^\]]*)\])?")
REL_RE = re.compile(r"\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)")


def sg_facts(text):
    """Parse the SG text into comparable FACTS, ignoring confidence numbers and pixel
    boxes. Without this, a harmless confidence wobble (0.4998 -> 0.4889) reads as a
    changed scene graph and every step looks attacked."""
    if not text:
        return set()
    t = str(text)
    facts = set()
    for _conf, label, attrs in OBJ_RE.findall(t):
        label = label.strip().lower()
        facts.add(("object", label))
        for a in (attrs or "").split(","):
            a = a.strip().lower()
            if a:
                facts.add(("attribute", label, a))
    rel_part = t.split("Relations:", 1)[1] if "Relations:" in t else ""
    for a, r, b in REL_RE.findall(rel_part):
        facts.add(("relation", a.strip().lower(), r.strip().lower(), b.strip().lower()))
    return facts


def fmt(f):
    if f[0] == "object":
        return f"{f[1]}"
    if f[0] == "attribute":
        return f"{f[1]}[{f[2]}]"
    return f"({f[1]}, {f[2]}, {f[3]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="output/rung2/staged or output/rung2/ebnav")
    ap.add_argument("--episode", default="", help="only this episode id")
    ap.add_argument("--max-steps", type=int, default=6, help="steps to print per episode")
    args = ap.parse_args()

    eps = find_episodes(args.root)
    if not eps:
        print(f"No steps.jsonl under {args.root}.\n"
              "These logs are written by runs started AFTER step_log.py was added. "
              "Re-run the benchmark to produce them.")
        return

    tot_sg_changed = tot_act_changed = tot_cmp = tot_fact = 0
    for (rep, eset, ep), conds in sorted(eps.items()):
        if args.episode and args.episode != ep:
            continue
        if "esca_benign" not in conds or "esca_patch" not in conds:
            continue
        b, p = load_steps(conds["esca_benign"]), load_steps(conds["esca_patch"])
        if not b or not p:
            continue
        tot_cmp += 1
        head = " / ".join(x for x in (rep, eset, ep) if x)
        print("=" * 78)
        print(head)
        n_sg, n_act, n_fact_removed = 0, 0, 0
        for i in range(min(len(b), len(p))):
            fb, fp = sg_facts(b[i].get("scene_graph_text")), sg_facts(p[i].get("scene_graph_text"))
            gone, added = fb - fp, fp - fb
            ab, apn = b[i].get("action"), p[i].get("action")
            sg_changed, act_changed = bool(gone or added), ab != apn
            n_sg += sg_changed
            n_act += act_changed
            n_fact_removed += any(g[0] in ('attribute', 'relation') for g in gone)
            if i < args.max_steps:
                print(f"  step {i:>2} | patch on frame: {p[i].get('frame_patched')}")
                print(f"     SG benign: {sorted(map(fmt, fb)) if fb else '(empty)'}")
                print(f"     SG patch : {sorted(map(fmt, fp)) if fp else '(empty)'}")
                if gone:
                    print(f"     >> REMOVED by patch: {sorted(map(fmt, gone))}")
                if added:
                    print(f"     >> ADDED by patch  : {sorted(map(fmt, added))}")
                print(f"     action benign={ab}  patch={apn}"
                      f"{'   <-- DIVERGED' if act_changed else ''}")
        n = min(len(b), len(p))
        print(f"  --> steps compared {n}; SG differed on {n_sg}; "
              f"attribute/relation FACT deleted on {n_fact_removed}; action differed on {n_act}")
        tot_sg_changed += (n_sg > 0)
        tot_act_changed += (n_act > 0)
        tot_fact += (n_fact_removed > 0)

    print("\n" + "=" * 78)
    print(f"episodes compared            : {tot_cmp}")
    print(f"  patch changed the SG        : {tot_sg_changed}")
    print(f"  patch DELETED a target fact : {tot_fact}   <- this is the attack landing")
    print(f"  patch changed the behaviour : {tot_act_changed}   "
          f"(NOT causal on its own: the planner is stochastic, so two identical\n                                 benign runs also diverge. Read the success numbers.)")
    if tot_cmp and tot_sg_changed == 0:
        print("\nVERDICT: the patch never reached the planner. The scene graph is identical\n"
              "with and without it. Fix grounding or injection, not the instruction.")
    elif tot_cmp and tot_sg_changed and tot_act_changed == 0:
        print("\nVERDICT: the patch DID change the scene graph, and the planner ignored it.\n"
              "The instruction is answerable from the image alone. Fix the TASK: make the\n"
              "attacked fact the only way to resolve the instruction.")
    elif tot_cmp:
        print("\nVERDICT: the patch changed the scene graph AND the behaviour on at least\n"
              "some episodes. Check the success numbers to see which direction.")


if __name__ == "__main__":
    main()
