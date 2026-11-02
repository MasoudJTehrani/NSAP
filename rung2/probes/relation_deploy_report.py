#!/usr/bin/env python
"""Why does the deployed relation edge appear or not, and does the patch remove it?

Reads the per-step logs of one relation run and decomposes the target edge
(pot, on, counter) into the conditions that must all hold for it to reach the planner:

  1. the pot passed the aggregation gate and is the subject
  2. some detection is named like a counter, so the pair exists
  3. the pair beats its own negation, p(on) > p(not on)      <- what the attack targets
  4. the pair survives the per-subject cap ($NSAP_REL_TOPN, 0 = no cap)

Steps 3 and 4 need `relation_scores`, logged by rung2/step_log.py. Runs made before that
was added report the first two conditions only.

  python rung2/probes/relation_deploy_report.py output/rung2/<run>
"""
import argparse, collections, glob, json, os, statistics, sys


def counterish(s):
    return "counter" in str(s).lower()


def load(run):
    out = collections.defaultdict(list)
    for f in glob.glob(os.path.join(run, "**", "steps.jsonl"), recursive=True):
        parts = f.split(os.sep)
        cond = next((p for p in parts if p in ("esca_benign", "esca_patch", "lang_benign",
                                               "lang_patch", "base_benign")), "?")
        for line in open(f):
            if line.strip():
                out[cond].append(json.loads(line))
    return out


def report(cond, recs, patched_only=False):
    if patched_only:
        recs = [r for r in recs if r.get("frame_patched")]
    n = len(recs)
    if not n:
        return None
    c = collections.Counter()
    pon, pnot, margins = [], [], []
    for r in recs:
        sg = r.get("sgclip") or {}
        sel = [(x[1] if isinstance(x, list) else x) for x in (sg.get("grounded_objects") or [])]
        pot = any(str(s).lower() == "pot" for s in sel)
        names = [v[0][1] for v in (sg.get("per_box_topk") or {}).values() if v]
        counter = any(counterish(x) for x in names)
        text = str(r.get("scene_graph_text") or "")
        edge = "(pot, on, counter" in text
        c["calls"] += 1
        c["empty"] += "no scene graph" in text
        c["pot_subject"] += pot
        c["pair_available"] += (pot and counter)
        c["edge"] += edge
        rels = sg.get("grounded_relations") or []
        c["relations"] += len(rels)
        rs = (sg.get("relation_scores") or {}).get("pot")
        if rs:
            hit = [row for row in rs if counterish(row[0])]
            if hit:
                o, p, notp, _rel = hit[0]
                c["scored"] += 1
                if isinstance(p, (int, float)) and isinstance(notp, (int, float)):
                    pon.append(p); pnot.append(notp); margins.append(p - notp)
                    c["beats_negation"] += p > notp
                    c["beats_and_surfaced"] += (p > notp and edge)
    return c, pon, pnot, margins


def fmt(a, b):
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else f"{a}/{b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--patched-only", action="store_true",
                    help="for esca_patch, count only calls where the patch was on screen")
    a = ap.parse_args()
    conds = load(a.run)
    if not conds:
        sys.exit(f"no steps.jsonl under {a.run}")
    print(f"# {os.path.basename(a.run.rstrip('/'))}\n")
    for cond in ("esca_benign", "esca_patch", "lang_benign", "lang_patch"):
        if cond not in conds:
            continue
        for po in ([False, True] if cond.endswith("patch") else [False]):
            res = report(cond, conds[cond], patched_only=po)
            if not res:
                continue
            c, pon, pnot, margins = res
            tag = f"{cond}{' (patch on screen)' if po else ''}"
            ne = c["calls"] - c["empty"]
            print(f"## {tag}: {c['calls']} planner calls")
            print(f"   no graph sent            : {fmt(c['empty'], c['calls'])}")
            print(f"   relations per graph      : {c['relations'] / ne:.1f}" if ne else "")
            print(f"   1. pot is the subject    : {fmt(c['pot_subject'], c['calls'])}")
            print(f"   2. counter also detected : {fmt(c['pair_available'], c['pot_subject'])} of those")
            if c["scored"]:
                print(f"   3. pair beats negation   : {fmt(c['beats_negation'], c['scored'])} of {c['scored']} scored pairs")
                print(f"      p(on) mean {statistics.mean(pon):.3f}, p(not on) mean {statistics.mean(pnot):.3f}, "
                      f"margin mean {statistics.mean(margins):+.3f}, median {statistics.median(margins):+.3f}")
            print(f"   4. edge reaches planner  : {fmt(c['edge'], c['calls'])} of all calls, "
                  f"{fmt(c['edge'], c['pair_available'])} when the pair was available")
            print()


if __name__ == "__main__":
    main()
