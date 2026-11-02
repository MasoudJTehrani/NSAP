#!/usr/bin/env python
"""Every number the paper can cite, recomputed from the published artifacts.

Reads output/ only (the NSAP-artifacts dataset, plus output/probes/) and writes one
Markdown file in which every number names the file it came from. It runs no model and
no simulator, so it finishes in seconds and anyone holding the artifacts can check it.

  python rung2/paper_numbers.py                      # -> output/NUMBERS.md
  python rung2/paper_numbers.py --out somewhere.md
"""
import argparse, collections, datetime, glob, itertools, json, math, os, re, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RUNS = "output/rung2"
CONDS = ("base_benign", "esca_benign", "esca_patch", "lang_benign", "lang_patch")
ESCA = "b16d63a"   # upstream ESCA commit every configuration citation refers to

L = []
def out(s=""): L.append(s)
def h(n, s): out(); out("#" * n + " " + s); out()
def para(s): out(s); out()
def src(*ps): para("Source: " + ", ".join(f"`{p}`" for p in ps))
def load(p): return json.load(open(p))
def frac(a, b): return f"{a}/{b} = {100 * a / b:.0f}%" if b else f"{a}/{b}"
def pct(a, b): return f"{100 * a / b:.0f}%" if b else "n/a"
def table(head, rows):
    out("| " + " | ".join(head) + " |"); out("|" + "---|" * len(head))
    for r in rows: out("| " + " | ".join(str(x) for x in r) + " |")
    out()

# --------------------------------------------------------------------------- provenance

def provenance():
    sys.path.insert(0, ROOT)
    try:
        from functions.download_model import (HF_ARTIFACTS_ID, HF_ARTIFACTS_REVISION,
                                               HF_DATA_ID, HF_DATA_REVISION)
    except Exception:
        HF_ARTIFACTS_ID = HF_ARTIFACTS_REVISION = HF_DATA_ID = HF_DATA_REVISION = "?"
    def git(*a):
        try: return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except Exception: return ""
    commit = git("rev-parse", "--short", "HEAD") or "?"
    dirty = " (with uncommitted changes)" if git("status", "--porcelain") else ""
    out("# NSAP: every number, with its source")
    out()
    out(f"Generated {datetime.date.today()} by `rung2/paper_numbers.py` from code `{commit}`{dirty}.")
    out()
    table(["input", "where"], [
        ["trained patches, run logs, result JSON", f"`{HF_ARTIFACTS_ID}` at `{HF_ARTIFACTS_REVISION[:12]}`, unpacked into `output/`"],
        ["probe scripts and their outputs", "`rung2/probes/`, `output/probes/` (not yet in the artifacts repo)"],
        ["simulated dataset", f"`{HF_DATA_ID}` at `{HF_DATA_REVISION[:12]}`"],
        ["ESCA code cited for configuration", f"upstream ESCA at `{ESCA}`"],
    ])
    para("Every path below is relative to the repository root. Rerun this script after "
         "`bash replicate.sh fetch-artifacts` and the file should come back identical.")
    para("Sections 1 and 2 are deterministic. Sections 3 to 6 come from the embodied runs and "
         "describe what the planner received and did. Section 7 holds success rates, which drift "
         "between identical runs, so they are there for the reproducibility finding and for threats "
         "to validity, not as results. Section 8 lists numbers from earlier documents that should "
         "not be used.")

# --------------------------------------------------------------------------- 1. configuration

def configuration():
    h(2, "1. ESCA's configuration: there is no single default")
    para(f"All citations are to upstream ESCA at `{ESCA}`. Our fork changes none of these values.")
    table(["setting", "planner constructor", "ESCA navigation script", "ESCA ALFRED evaluator"], [
        ["`gd_only`", "False", "**True**", "False"],
        ["`top_k`", "1", "1", "1"],
        ["`aggr_thres`", "0.1", "**0.3**", "**0.75**"],
        ["`n_shot`", "1", "3", "10"],
        ["`chat_history`", "False", "True", "False"],
        ["`multistep`", "False", "True", "False"],
        ["`tp`", "1", "4", "4"],
        ["detector box / text threshold", "0.2 / 0.1 with `gd_only`, else 0.1 / 0.1", "same", "0.6 / 0.5"],
    ])
    para("Sources: `embodiedbench/planner/navigation/esca.py:165-178` (constructor), `:763` and `:775` "
         "(detector thresholds), `embodiedbench/evaluator/navigation/esca.py:299-329` (navigation "
         "script), `embodiedbench/evaluator/alfred/esca.py:361-396` (ALFRED).")
    para("**What `gd_only=True` does.** `planner/navigation/esca.py:759-772`: `predict_laser` is never "
         "called, so SGClip does not run and no attribute or relation can exist. The detector query is "
         "built as one string, `' '.join(target_attributes) + ' ' + ' '.join(target_names)` (`:762`), and "
         "`predict_gd` joins its `classes` argument with `'. '.join(classes)` (`:488`). Joining a string "
         "iterates over its characters, so every class is a single character. Section 5 shows the "
         "result in our logs.")
    para("**What the planner receives.** `planner/navigation/esca.py:807-808`: one line per frame, "
         "`f\"{obj_name}: {obj_box}\"`. With SGClip in the loop, `obj_name` is a `(confidence, label)` "
         "tuple, which is why the confidence appears in the text. Our fork adds attribute brackets and a "
         "`Relations:` list only when `NSAP_LASER_ATTR_REL=1`. The category runs leave it off, so their "
         "text is byte-for-byte upstream's. The staged attribute and relation runs turn it on.")
    para("**A guard the navigation script skips.** `evaluator/navigation/esca.py:219-221` rejects "
         "`chat_history` and `multistep` together, but only `embodiedbench/main.py:77` calls that check. "
         "The navigation script builds its evaluator directly (`:331-332`) and runs with both on. Our "
         "faithful runs match the script.")
    table(["our run", "`gd_only`", "`top_k`", "`aggr_thres`", "n_shot / chat_history / multistep / tp", "attr+rel text"], [
        ["`ebnav_tau0.3_faithful1`", "True", "1", "0.3", "3 / True / True / 4", "off"],
        ["`ebnav_tau0.3_faithful1_gdonly0`", "False", "1", "0.3", "3 / True / True / 4", "off"],
        ["`ebnav_tau0.4`", "False", "1", "0.4", "2 / False / False / 1", "off"],
        ["`staged_A_donothing_tau0.1`, `..._k1`", "False", "1", "0.1", "2 / False / False / 1", "on"],
        ["`staged_A_donothing_tau0.1_k1_lang`", "False", "1", "0.1", "2 / False / False / 1", "on"],
        ["`staged_A_donothing_tau0.4`", "False", "1", "0.4", "2 / False / False / 1", "on"],
        ["`staged_A_donothing_tau0.4_k15`", "False", "15", "0.4", "2 / False / False / 1", "on"],
        ["`staged_E_single_pot_tau0.4`", "False", "1", "0.4", "2 / False / False / 1", "on"],
    ])
    src("rung2/nav_harness.py", "rung2/run_category.sh", "rung2/run_staged_variants.sh", "output/rung2/*/*/*/*/run_meta.json")

# --------------------------------------------------------------------------- 2. model level

HEADS = ("category", "attribute", "relation")

def run_attack_evals(kind):
    pat = ("output/chameleon_patches/*/*/*/*_final_eval.json" if kind == "chameleon"
           else "output/universal_patches/*/*_final_eval.json")
    agg = collections.defaultdict(collections.Counter); per = collections.defaultdict(list)
    for f in sorted(glob.glob(pat)):
        d = load(f); m = d["attack_mode"]; a = agg[m]; a["patches"] += 1
        for k in ("n_frames_loaded", "n_frames_with_clean_target", "n_total", "n_flipped",
                  "total_category_changes", "total_attribute_changes", "total_relation_changes", "total_disappeared"):
            a[k] += d.get(k) or 0
        if d.get("n_total"): per[m].append(d["n_flipped"] / d["n_total"])
    return agg, per, pat

def ra_isolation(m, a):
    on = {"category": a["total_category_changes"] + a["total_disappeared"],
          "attribute": a["total_attribute_changes"], "relation": a["total_relation_changes"]}[m]
    tot = a["total_category_changes"] + a["total_attribute_changes"] + a["total_relation_changes"] + a["total_disappeared"]
    return on, tot

def outcome_records(masks, kind):
    return [r for r in load("output/outcome_eval.json") if r["masks"] == masks and r["kind"] == kind]

def oe_agg(recs):
    agg = collections.defaultdict(collections.Counter)
    for r in recs:
        a = agg[r["mode"]]; a["patches"] += 1; a["objects"] += r["n_objects"]
        a["clean"] += r["n_clean_grounded"]; a["flipped"] += r["n_flipped"]
        for k, v in r["iso_buckets"].items(): a["iso_" + k] += v
    return agg

def oe_isolation(m, a):
    on = a["iso_" + m] + (a["iso_disappeared"] if m == "category" else 0)
    return on, sum(a[k] for k in ("iso_category", "iso_attribute", "iso_relation", "iso_disappeared"))

def model_level():
    h(2, "2. Model level (deterministic)")

    h(3, "2.1 Precise instance masks: two evaluations of the same 30 chameleon patches")
    para("Both place the patch with the simulator's exact instance mask and score SGClip on that mask. "
         "They differ in what they count, and the paper has to pick one.")
    para("- **`run_attack.py --eval`** (`functions/eval/edit_distance.py`). A fact is an object for "
         "category and attribute, and a directed target edge for relation. Isolation is changes on the "
         "attacked head over all changes, but for category and attribute attacks it **never counts "
         "relation changes** (`edit_distance.py:124-145`), so relation collateral cannot lower it.\n"
         "- **`rung2/outcome_eval.py`**. Same flip definition for category, and the category counts agree exactly. "
         "For attribute the two scripts do not ground the same number of facts (101 against 113); we have not "
         "traced why. For relation, "
         "an object counts as flipped only if **every** clean `on` edge from it is gone. Isolation "
         "counts category, attribute and relation changes on the patched object for every head.")
    ra, ra_per, ra_pat = run_attack_evals("chameleon")
    oe = oe_agg(outcome_records("gt", "chameleon"))
    rows = []
    for m in HEADS:
        a, b = ra[m], oe[m]; on, tot = ra_isolation(m, a); on2, tot2 = oe_isolation(m, b)
        rng = f"{100 * min(ra_per[m]):.0f} to {100 * max(ra_per[m]):.0f}%" if ra_per[m] else "n/a"
        rows.append([m, a["patches"], frac(a["n_flipped"], a["n_total"]), rng, frac(on, tot),
                     frac(b["flipped"], b["clean"]), frac(on2, tot2)])
    table(["head", "patches", "flip, run_attack", "per-patch range", "isolation, run_attack",
           "flip, outcome_eval", "isolation, outcome_eval"], rows)
    para("Denominators: flip is over facts the **clean** model grounded. Isolation is over all changes "
         "observed on the patched object.")
    src(ra_pat, "output/outcome_eval.json (masks=gt, kind=chameleon)")

    h(3, "2.1b The RQ1 table: one evaluation for flip, isolation and collateral")
    para("Everything in this table comes from `outcome_eval` with ground-truth masks, so flip and isolation "
         "share one frame set and one definition of a change. Flip for relation is per edge: of the clean "
         "`on` edges leaving the patched object, how many the patch flipped. Universal patches are scored on "
         "held-out scenarios, chameleons on the one scenario each was trained on.")
    para("The isolation loss protects different heads per attack (`functions/attack/train.py:625-631` and "
         "`:693-703`): a category attack protects attribute, an attribute attack protects category, and a "
         "relation attack protects both. The relation head is never protected in a category or attribute "
         "attack, because it scores object pairs.")
    PROTECT = {"category": ("attribute",), "attribute": ("category",), "relation": ("category", "attribute")}
    rows = []
    for kind, lab in (("universal", "universal, held-out"), ("chameleon", "chameleon, own scenario")):
        recs = outcome_records("gt", kind); ag = oe_agg(recs)
        tally = collections.defaultdict(collections.Counter)
        for r in recs: tally[r["mode"]].update(r["outcomes"])
        for m in HEADS:
            a = ag[m]; on, tot = oe_isolation(m, a)
            if m == "relation":
                t = tally[m]; ch = t["predicate_changed"] + t["deleted"] + t["redirected"]
                flip = frac(ch, ch + t["unchanged"]) + " edges"
            else:
                flip = frac(a["flipped"], a["clean"])
            prot = ", ".join(f"{h_} {a['iso_' + h_]}" for h_ in PROTECT[m])
            unprot = "relation " + str(a["iso_relation"]) if m != "relation" else "none"
            rows.append([m, lab, flip, frac(on, tot), prot, unprot, a["iso_disappeared"]])
    table(["head", "patch", "flip", "isolation", "changes on protected heads", "changes on unprotected heads",
           "objects that vanished"], rows)
    src("output/outcome_eval.json (masks=gt)", "functions/attack/train.py")

    h(3, "2.2 Universal patches on the held-out split")
    para("Stage 1 patches, trained on 80% of scenarios and scored on the other 20%, so rooms and "
         "object instances are unseen. Chameleons are overfit to one scenario by design, so only these "
         "measure generalisation.")
    ua, _, upat = run_attack_evals("universal")
    ue = oe_agg(outcome_records("gt", "universal"))
    rows = []
    for m in HEADS:
        a, b = ua[m], ue[m]; on, tot = ra_isolation(m, a); on2, tot2 = oe_isolation(m, b)
        rows.append([m, frac(a["n_frames_with_clean_target"], a["n_frames_loaded"]), frac(a["n_flipped"], a["n_total"]),
                     frac(on, tot), frac(b["flipped"], b["clean"]), frac(on2, tot2)])
    table(["head", "frames where clean model grounds the target", "flip, run_attack", "isolation, run_attack",
           "flip, outcome_eval", "isolation, outcome_eval"], rows)
    src(upat, "output/outcome_eval.json (masks=gt, kind=universal)")

    h(3, "2.3 What each attacked fact turns into")
    para("The attack is untargeted: it pushes the fact across its decision boundary and never names a "
         "replacement. Counted over every scored object (category, attribute) or every clean target "
         "edge (relation), including ones the clean model did not ground, so totals exceed the flip "
         "denominators above.")
    tally = collections.defaultdict(collections.Counter); det = collections.defaultdict(collections.Counter)
    for r in outcome_records("gt", "chameleon"):
        tally[r["mode"]].update(r["outcomes"]); det[r["mode"]].update(r["details"])
    rows = []
    for m in HEADS:
        t = tally[m]; n = sum(t.values())
        rows.append([m, n, ", ".join(f"{k} {frac(v, n)}" for k, v in t.most_common()),
                     "; ".join(f"`{k}` x{v}" for k, v in det[m].most_common(4))])
    table(["head", "scored", "outcomes", "most common changes"], rows)
    para("Read: no category fact vanished, every attribute change is a negation, and every relation "
         "change is a predicate flip to its negation. No edge is ever deleted or redirected at the model "
         "level, by construction: the pairs come from the ground-truth masks and every pair is assigned its "
         "top-voted predicate (`functions/eval/scene_graph.py:174`), so an edge can change predicate but "
         "cannot disappear. Deletion and redirection only become possible once a detector decides which "
         "objects exist, which is sections 3 and 4.")
    src("output/outcome_eval.json (masks=gt, kind=chameleon)")

    h(3, "2.4 Ground-truth masks against SAM2 masks")
    para("Same patches, same frames. SAM2 is prompted with the ground-truth box, so this changes the mask "
         "source and nothing else: detection is still perfect.")
    rows = []
    for m in HEADS:
        for masks in ("gt", "sam2"):
            recs = outcome_records(masks, "chameleon"); b = oe_agg(recs)[m]; on, tot = oe_isolation(m, b)
            if m == "relation":
                t = collections.Counter()
                for r in recs:
                    if r["mode"] == m: t.update(r["outcomes"])
                ch = t["predicate_changed"] + t["deleted"] + t["redirected"]
                flip = f"{frac(ch, ch + t['unchanged'])} edges ({frac(b['flipped'], b['clean'])} objects)"
            else:
                flip = frac(b["flipped"], b["clean"])
            rows.append([m, masks, frac(b["clean"], b["objects"]), flip, frac(on, tot)])
    table(["head", "masks", "clean model grounds the fact", "flip", "isolation"], rows)
    src("output/outcome_eval.json (kind=chameleon)")

    h(3, "2.5 Detector-driven scoring: the deployed perception path")
    para("The patch is still placed with the precise mask, which an attacker with physical access can "
         "do, but scoring goes through `predict_gd` then `predict_laser`, so regions are the detector's "
         "filled rectangles and the aggregation gate applies. The target is matched to a graph node by "
         "box overlap.")
    para("- grounding rate = frames where the deployed stack grounds the target fact / frames scored\n"
         "- flip rate = of those, frames where the fact changed\n"
         "- pair flip (relation) = target (subject, object) pairs whose `on` edge changed / target pairs")
    def dep(fn):
        agg = collections.defaultdict(collections.Counter)
        for r in load("output/" + fn):
            a = agg[r["mode"]]; a["frames"] += r["n_frames"]; a["grounded"] += r["n_clean_grounded"]
            a["flipped"] += r["n_flipped"]; a["pairs"] += r["pair_total"] or 0; a["pair_flipped"] += r["pair_flipped"] or 0
        return agg
    k1, k15, rnd = dep("deploy_eval.json"), dep("deploy_eval_topk15.json"), dep("deploy_eval_random.json")
    out("**At ESCA's `top_k=1`**"); out()
    table(["head", "grounding rate", "flip", "pair flip"],
          [[m, frac(k1[m]["grounded"], k1[m]["frames"]), frac(k1[m]["flipped"], k1[m]["grounded"]),
            frac(k1[m]["pair_flipped"], k1[m]["pairs"]) if m == "relation" else "-"] for m in HEADS])
    src("output/deploy_eval.json")
    out("**At `top_k=15`, against a random patch** (same size, anchor and object, random pixels). "
        "The random control exists only at `top_k=15`."); out()
    rows = []
    for m in HEADS:
        a, r = k15[m], rnd[m]
        if m == "relation":
            t, c = (a["pair_flipped"], a["pairs"]), (r["pair_flipped"], r["pairs"]); lab = "pairs"
        else:
            t, c = (a["flipped"], a["grounded"]), (r["flipped"], r["grounded"]); lab = "frames"
        d = (100 * t[0] / t[1] - 100 * c[0] / c[1]) if t[1] and c[1] else float("nan")
        rows.append([m, frac(a["grounded"], a["frames"]), f"{frac(*t)} {lab}", f"{frac(*c)} {lab}", f"{d:+.0f} pts"])
    table(["head", "grounding rate", "flip, trained patch", "flip, random patch", "attack effect"], rows)
    src("output/deploy_eval_topk15.json", "output/deploy_eval_random.json")
    para("At `top_k=15` the relation row rests on 7 target pairs.")

    h(3, "2.6 Probes of the relation and attribute heads")
    rel_l = load("output/probes/rel_laser_style.json"); surf = load("output/probes/surfacing_probe.json")["relation"]
    arg = load("output/probes/rel_argument.json"); voc = load("output/probes/vocab_probe.json")
    col = load("output/probes/colour_disambig.json"); tv = load("output/probes/tvshots/result.json")
    tv_ok = sum(r["correct"] for r in tv); d = sorted({r["dist"] for r in tv})
    blind = re.search(r"vision-blind pose: (\d+)/(\d+)", open("output/probes/tv_search.log").read())
    ranks = arg["ranks"]; r4 = sum(v for k, v in ranks.items() if int(k) >= 4)
    table(["what", "result", "source"], [
        ["relation, LASER's own setup: precise masks, one specified (pot, counter) pair, `aggr_thres=0.1`",
         f"clean says `on` in {frac(rel_l['clean_says_on'], rel_l['pairs'])}; of those, trained patch flips "
         f"{frac(rel_l['trained_flip'], rel_l['attackable'])}, random patch flips {frac(rel_l['random_flip'], rel_l['attackable'])}",
         "`output/probes/rel_laser_style.json`"],
        ["relation direction: the true direction scores higher than the reverse",
         f"{frac(surf['argmax_correct'], surf['pairs'])} of ordered pairs (chance is 50%)", "`output/probes/surfacing_probe.json`"],
        ["relation direction: ESCA's rule emits both directions of the same pair",
         f"{frac(surf['current_emits_BOTH'], surf['pairs'])}", "`output/probes/surfacing_probe.json`"],
        ["relation argument: true receptacle (AI2-THOR `parentReceptacles`) ranked first",
         f"{frac(arg['stats']['true_receptacle_is_argmax'], arg['stats']['cases'])}; ranks 1/2/3/4+ = "
         f"{ranks.get('1', 0)}/{ranks.get('2', 0)}/{ranks.get('3', 0)}/{r4}", "`output/probes/rel_argument.json`"],
        ["relation predicates: frames in which each of 12 predicates grounds at all",
         ", ".join(f"`{k}` {v}/{voc['pred_tries'][k]}" for k, v in sorted(voc["pred_hits"].items(), key=lambda x: -x[1])),
         "`output/probes/vocab_probe.json`"],
        ["colour on small objects", f"correct colour granted {frac(voc['colour']['true'], voc['colour']['n'])}, a wrong "
         f"colour also granted {frac(voc['colour']['false'], voc['colour']['n'])}", "`output/probes/vocab_probe.json`"],
        ["two same-category objects of different colours",
         f"{col['collapsed']} pairs scored, all {col['collapsed']} given identical colours; "
         f"{col['unmatched_box']} of {col['unmatched_box'] + col['collapsed']} pairs dropped because detector boxes did not match instances",
         "`output/probes/colour_disambig.json`"],
        ["gpt-4o reads the television's on/off state from the planner's own frames",
         f"{frac(tv_ok, len(tv))} correct at {d[0]} to {d[-1]} m", "`output/probes/tvshots/result.json`"],
        ["scenes with any reachable pose where gpt-4o misreads the screen",
         f"{blind.group(1)}/{blind.group(2)}" if blind else "?", "`output/probes/tv_search.log`"],
        ["of those poses, SGClip still grounds `television` with `displaying`",
         "1/4 (FloorPlan202). The log's own summary says 2/4 because it counts FloorPlan207, whose grounded label is `cabinet`",
         "`output/probes/sgclip_poses.log`"],
    ])

    h(3, "2.7 Stealth: LPIPS against the clean frame")
    para("Whole frame, AlexNet LPIPS, both patch types pasted at coverage 0.5 so they cover the same "
         "area. Coverage applies per side, so the training-time mix of 0.3 and 0.5 would otherwise "
         "differ by 2.78 times in area. Lower is less visible. Whole-frame values are diluted by the "
         "unpatched area, so read them relative to each other.")
    st = load("output/stealth_eval.json")
    rows = []
    for m in HEADS:
        c = [r for r in st if r["mode"] == m and r["kind"] == "chameleon"]
        u = [r for r in st if r["mode"] == m and r["kind"] == "universal"]
        mc = statistics.mean(r["lpips_full"] for r in c); mu = statistics.mean(r["lpips_full"] for r in u)
        rc = statistics.mean(r["lpips_region"] for r in c); ru = statistics.mean(r["lpips_region"] for r in u)
        rows.append([m, len(c), f"{mc:.3f}", f"{mu:.3f}", f"{100 * (mc - mu) / mu:+.0f}%", f"{rc:.3f}", f"{ru:.3f}",
                     f"{statistics.median(r['lpips_best'] for r in c):.3f} / {statistics.median(r['lpips_median'] for r in c):.3f} / "
                     f"{statistics.median(r['lpips_worst'] for r in c):.3f}"])
    table(["head", "chameleons", "whole frame, chameleon", "whole frame, universal", "chameleon vs universal",
           "patch region, chameleon", "patch region, universal", "chameleon, per-patch best / median / worst viewpoint (median over patches)"], rows)
    flips = {r["patch"]: r["flip_rate"] for r in outcome_records("gt", "chameleon") if r["flip_rate"] is not None}
    pairs = [(r["lpips_full"], flips[r["patch"]]) for r in st if r["kind"] == "chameleon" and r["patch"] in flips]
    xs, ys = zip(*pairs); mx, my = statistics.mean(xs), statistics.mean(ys)
    rr = sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    def pearson(ps):
        if len(ps) < 3: return float("nan")
        xs, ys = zip(*ps); mx, my = statistics.mean(xs), statistics.mean(ys)
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        return sum((x - mx) * (y - my) for x, y in ps) / den if den else float("nan")
    within = []
    for m in HEADS:
        ps = [(r["lpips_full"], flips[r["patch"]]) for r in st
              if r["kind"] == "chameleon" and r["mode"] == m and r["patch"] in flips]
        within.append(f"{m} {pearson(ps):+.2f} (n={len(ps)})")
    para(f"Pearson r between whole-frame LPIPS and flip rate (`outcome_eval`, masks=gt): **{rr:+.2f}** pooled "
         f"over all {len(pairs)} chameleons, and within each head " + ", ".join(within) + ". Part of the pooled "
         "value is the difference between heads: attribute patches are both the most visible and the strongest. "
         "With 9 to 11 patches per head, no within-head value is reliable on its own. Positive r means more "
         "visible patches flip more.")
    src("output/stealth_eval.json", "output/outcome_eval.json")

# --------------------------------------------------------------------------- step logs

OBJ = re.compile(r"\(\s*([0-9.]+)\s*,\s*'([^']+)'\s*\)(?:\s*\[([^\]]*)\])?")
REL = re.compile(r"\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)")
BOX = re.compile(r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]")

def last_frame(text):
    parts = re.split(r"Its scene graph at frame \d+ is:", str(text or ""))
    return parts[-1] if len(parts) > 1 else str(text or "")

def received(rec):
    """What the planner was sent for the CURRENT frame: objects (label, attrs), relations, boxes."""
    text = rec.get("scene_graph_text")
    if not text or "no scene graph grounded" in str(text):
        return None
    seg = last_frame(text)
    body, _, relpart = seg.partition("Relations:")
    # Objects come from the logged grounding, which is the CURRENT frame. The text cannot be
    # trusted for that in multistep runs: ESCA appends a frame's line only when that frame has
    # a detection (planner/navigation/esca.py:806), so the last line may be an older frame.
    # Attributes are read from the text, and only the staged runs (single frame) carry any.
    attrs = collections.defaultdict(list)
    for _c, lab, at in OBJ.findall(body):
        attrs[lab.strip().lower()].extend(a.strip().lower() for a in (at or "").split(",") if a.strip())
    objs = [(lab, attrs.get(lab, [])) for lab in
            (((x[1] if isinstance(x, list) else str(x)).lower()) for x in (rec.get("sgclip") or {}).get("grounded_objects") or [])]
    rels = [(a.strip().lower(), r.strip().lower(), b.strip().lower()) for a, r, b in REL.findall(relpart)]
    boxes = [tuple(map(int, b)) for b in BOX.findall(body)]
    return {"objs": objs, "rels": rels, "boxes": boxes}

def episodes():
    """-> list of dicts: run, rep, cond, eset, ep, head, steps"""
    eps = []
    for f in sorted(glob.glob(f"{RUNS}/*/rep*/**/steps.jsonl", recursive=True)):
        parts = f.split("/"); run, rep = parts[2], parts[3]
        cond = next((p for p in parts if p in CONDS), None); ep = parts[-2]
        eset = next((p for p in parts if p in ("long_horizon", "visual_appearance")), "")
        head = "attribute" if ep.startswith("attr_") else "relation" if ep.startswith("rel_") else f"category, {eset}"
        steps = [json.loads(l) for l in open(f) if l.strip()]
        eps.append(dict(run=run, rep=rep, cond=cond, eset=eset, ep=ep, head=head, steps=steps, path=f))
    return eps

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

# --------------------------------------------------------------------------- 2.8 content ablation

def content_ablation():
    """Chameleon against two arms that change one thing at a time.

    Comparing chameleons against the Stage-1 universal patches cannot isolate the content
    loss: those differ in the loss, the initialisation, the step budget, the coverage, the
    training data and the evaluation split. These arms hold the scenario, the anchor, the
    steps, the coverage, the crops and the seed fixed:

        chameleon  object-init, lambda_content 20 (or 5)
        content0   object-init, lambda_content 0      -> chameleon minus this = the loss
        noinit     warm start,  lambda_content 0      -> content0 minus this = the init
    """
    oc, st = "output/ablation_outcome.json", "output/ablation_stealth.json"
    if not (os.path.exists(oc) and os.path.exists(st)):
        return
    h(3, "2.8 What the content loss and the object initialisation each buy")
    para("The chameleon patches differ from the Stage-1 universal patches in six ways at once, "
         "so that comparison cannot attribute anything to the content loss. These two arms change "
         "one thing at a time. Every arm uses the same scenario, the same anchor and size from the "
         "same Stage-1 patch, the same 600 steps, the same coverage, the same crops and the same "
         "seed as the chameleon it is compared against. `content0` keeps the object initialisation "
         "and sets `lambda_content` to 0. `noinit` also drops the object initialisation and warm "
         "starts from the Stage-1 patch instead.")
    rows_o = [r for r in load(oc)] + [r for r in load("output/outcome_eval.json")]
    rows_s = [r for r in load(st)] + [r for r in load("output/stealth_eval.json")]
    ARMS = ("chameleon", "content0", "noinit")

    def key(r): return (r.get("mode"), str(r.get("scenario") or ""))
    # keep only scenarios present in every arm, so every comparison is paired
    seen = {a: {key(r) for r in rows_o if r.get("kind") == a and r.get("masks", "gt") == "gt"} for a in ARMS}
    paired = set.intersection(*[seen[a] for a in ARMS]) if all(seen.values()) else set()
    rows = []
    for m in HEADS:
        ks = {k for k in paired if k[0] == m}
        if not ks: continue
        cells = []
        for a in ARMS:
            sel = [r for r in rows_o if r.get("kind") == a and r.get("masks", "gt") == "gt" and key(r) in ks]
            if m == "relation":
                # Per edge, the definition the RQ1 table uses: of the clean `on` edges
                # leaving the patched object, how many the patch changed.
                oc = collections.Counter()
                for r in sel: oc.update(r.get("outcomes") or {})
                fl = oc["predicate_changed"] + oc["deleted"] + oc["redirected"]
                tot = sum(v for k2, v in oc.items() if k2 != "no_target_edge")
            else:
                fl = sum(r["n_flipped"] for r in sel); tot = sum(r["n_clean_grounded"] for r in sel)
            ss = [r for r in rows_s if r.get("kind") == a and key(r) in ks and r.get("lpips_region") is not None]
            reg = statistics.mean(r["lpips_region"] for r in ss) if ss else None
            full = statistics.mean(r["lpips_full"] for r in ss if r.get("lpips_full") is not None) if ss else None
            cells.append((frac(fl, tot), f"{reg:.3f}" if reg is not None else "-",
                          f"{full:.4f}" if full is not None else "-"))
        rows.append([m, len(ks)] + [x for c in cells for x in c])
    table(["head", "scenarios"] + [f"{a}: {w}" for a in ARMS for w in ("flip", "LPIPS region", "LPIPS frame")], rows)
    para("Flip is pooled over the paired scenarios of that head. For category and attribute the "
         "denominator counts the facts the clean model grounded; for relation it counts the clean "
         "`on` edges leaving the patched object, the same definition the RQ1 table uses. LPIPS is the mean over patches, with every patch pasted at "
         "coverage 0.5 so the areas match. Lower LPIPS means the patch is closer to what was there "
         "before. Read chameleon against `content0` for the content loss, and `content0` against "
         "`noinit` for the object initialisation.")
    src(oc, st, "output/outcome_eval.json", "output/stealth_eval.json")


# --------------------------------------------------------------------------- 3. what the planner receives

def planner_receives(eps):
    h(2, "3. What the planner receives")
    para("One row per run and head, over every planner call in `esca_benign`, the condition with a "
         "scene graph and no patch. \"No graph sent\" means the planner received no scene-graph text at "
         "all. The other columns describe the current frame; in multistep runs the text also repeats up to "
         "two earlier frames.")
    groups = collections.defaultdict(list)
    for e in eps:
        if e["cond"] == "esca_benign": groups[(e["run"], e["head"])].extend(e["steps"])
    rows = []
    for (run, head), steps in sorted(groups.items()):
        rs = [received(s) for s in steps]; ne = [r for r in rs if r]
        labs = [l for r in ne for l, _ in r["objs"]]
        rows.append([f"`{run}`", head, len(steps), frac(len(steps) - len(ne), len(steps)),
                     f"{statistics.mean(len(r['objs']) for r in ne):.1f}" if ne else "-",
                     frac(sum(1 for r in ne if any(a for _, a in r["objs"])), len(ne)),
                     frac(sum(1 for r in ne if r["rels"]), len(ne)),
                     frac(sum(1 for l in labs if len(l.strip()) <= 1), len(labs))])
    table(["run", "head", "planner calls", "no graph sent", "objects per graph", "graphs with an attribute",
           "graphs with a relation", "one-character labels"], rows)
    src(f"{RUNS}/<run>/rep*/esca_benign/**/steps.jsonl")

    h(3, "3.1 Labels the planner actually sees")
    rows = []
    for (run, head), steps in sorted(groups.items()):
        c = collections.Counter(l for s in steps for r in [received(s)] if r for l, _ in r["objs"])
        n = sum(c.values())
        rows.append([f"`{run}`", head, ", ".join(f"`{k!s}` {pct(v, n)}" for k, v in c.most_common(6))])
    table(["run", "head", "six most frequent labels (share of all labels)"], rows)
    src(f"{RUNS}/<run>/rep*/esca_benign/**/steps.jsonl")

    h(3, "3.2 What fills the attribute slot")
    para("ESCA scores the attributes gpt-4o proposed for the current step, so the slot holds whatever "
         "gpt-4o asked about, not only states.")
    rows = []
    for (run, head), steps in sorted(groups.items()):
        c = collections.Counter(a for s in steps for r in [received(s)] if r for _, at in r["objs"] for a in at)
        n = sum(c.values())
        if n: rows.append([f"`{run}`", head, n, ", ".join(f"`{k}` {pct(v, n)}" for k, v in c.most_common(6))])
    table(["run", "head", "attribute values sent", "six most frequent"], rows)
    src(f"{RUNS}/staged_*/rep*/esca_benign/*/steps.jsonl")

# --------------------------------------------------------------------------- 4. benign vs patched

def target_stats(steps, head):
    s = collections.Counter(); x = collections.Counter()
    for st in steps:
        s["calls"] += 1; s["patched"] += bool(st.get("frame_patched")); r = received(st)
        if not r: s["empty"] += 1; continue
        labs = [l for l, _ in r["objs"]]
        if len(labs) != len(set(labs)): s["dup_label"] += 1
        if head == "attribute":
            for l, a in r["objs"]:
                if "displaying" in a:
                    s["disp_any"] += 1; s["disp_on_tv"] += l == "television"
            tv = [a for l, a in r["objs"] if l == "television"]
            if tv:
                s["tv"] += 1; s["disp"] += any("displaying" in a for a in tv)
        else:
            if "pot" in labs: s["pot"] += 1
            for a, rel, b in r["rels"]:
                if rel == "on":
                    s["on_any"] += 1; s["on_from_pot"] += a == "pot"
                if a == "pot" and rel == "on":
                    s["pot_on"] += 1; x[b] += 1
                    if "counter" in b: s["pot_on_counter"] += 1
            s["has_target_edge"] += any(a == "pot" and rel == "on" and "counter" in b for a, rel, b in r["rels"])
        s["nonempty"] += 1
    return s, x

def patched_graphs(eps):
    h(2, "4. What the patch did to the graph the planner received (staged runs)")
    para("Benign and patched episodes diverge after their first action, so frames cannot be paired "
         "step by step, and in the staged tasks the agent starts facing away from the target (the patch "
         "is on screen at step 0 in 0 episodes). These rows therefore compare distributions over every "
         "planner call of each condition, which is the definition used in the earlier analysis.")
    para("- attribute: of calls whose graph contains `television`, how many also carry `displaying`\n"
         "- relation: of all `(pot, on, X)` edges sent, how many have a counter as X, and how many calls "
         "contain a `(pot, on, counter)` edge at all")
    by = collections.defaultdict(list)
    for e in eps:
        if e["head"] in ("attribute", "relation"): by[(e["run"], e["head"], e["cond"])].extend(e["steps"])
    rows_a, rows_r = [], []
    for run in sorted({k[0] for k in by}):
        for pair in (("esca_benign", "esca_patch"), ("lang_benign", "lang_patch")):
            for head in ("attribute", "relation"):
                if (run, head, pair[0]) not in by: continue
                (sb, xb), (sp, xp) = (target_stats(by[(run, head, c)], head) for c in pair)
                if head == "attribute":
                    rows_a.append([f"`{run}`", f"{pair[0]} -> {pair[1]}",
                                   f"{frac(sb['tv'], sb['calls'])} -> {frac(sp['tv'], sp['calls'])}",
                                   f"{frac(sb['disp'], sb['tv'])} -> {frac(sp['disp'], sp['tv'])}",
                                   f"{frac(sb['disp'], sb['calls'])} -> {frac(sp['disp'], sp['calls'])}",
                                   frac(sp["patched"], sp["calls"]), frac(sb["disp_on_tv"], sb["disp_any"])])
                else:
                    rows_r.append([f"`{run}`", f"{pair[0]} -> {pair[1]}",
                                   f"{frac(sb['pot'], sb['calls'])} -> {frac(sp['pot'], sp['calls'])}",
                                   f"{frac(sb['pot_on_counter'], sb['pot_on'])} -> {frac(sp['pot_on_counter'], sp['pot_on'])}",
                                   f"{frac(sb['has_target_edge'], sb['calls'])} -> {frac(sp['has_target_edge'], sp['calls'])}",
                                   ", ".join(f"{k} {v}" for k, v in xb.most_common(4)),
                                   frac(sp["patched"], sp["calls"]), frac(sb["on_from_pot"], sb["on_any"])])
    out("**Attribute tasks**"); out()
    table(["run", "conditions", "calls with `television`", "...carrying `displaying`", "calls with `television` + `displaying`",
           "patched calls with the patch on screen",
           "benign: `displaying` assertions that are on the television"], rows_a)
    out("**Relation tasks**"); out()
    table(["run", "conditions", "calls with a `pot` node", "`(pot, on, X)` with X a counter",
           "calls with `(pot, on, counter)`", "benign X, most frequent", "patched calls with the patch on screen",
           "benign: `on` edges whose subject is the pot"], rows_r)
    dup = collections.Counter(); tot = collections.Counter()
    for (run, head, cond), steps in by.items():
        if head != "relation": continue
        s, _ = target_stats(steps, head); dup[run] += s["dup_label"]; tot[run] += s["nonempty"]
    para("Two nodes with the same label in one relation-task graph, which telling two pots apart would "
         "need. At `top_k=1` this is impossible by construction, since the graph admits one node: "
         + "; ".join(f"`{r}` {frac(dup[r], tot[r])}" for r in sorted(tot)) + ".")
    src(f"{RUNS}/staged_*/rep*/<condition>/(attr|rel)_*/steps.jsonl")

# --------------------------------------------------------------------------- 5. gd_only + 6. behaviour at step 0

def step0_pairs(eps):
    h(2, "5. What ESCA's own navigation config sends, and what the patch can change")
    para("Under `gd_only=True` SGClip never runs (section 1), so the patch cannot act through SGClip, and "
         "every label is a single character. Inside the scene graph the only thing left for the patch to "
         "change is the detector's box. It can still reach the planner through the image, because the "
         "planner sees the patched frame too.")
    rows = []
    for run in ("ebnav_tau0.3_faithful1", "ebnav_tau0.3_faithful1_gdonly0"):
        for cond in ("esca_benign", "esca_patch"):
            labs = collections.Counter(l for e in eps if e["run"] == run and e["cond"] == cond
                                       for s in e["steps"] for r in [received(s)] if r for l, _ in r["objs"])
            n = sum(labs.values())
            rows.append([f"`{run}`", cond, n, frac(sum(v for k, v in labs.items() if len(k.strip()) <= 1), n),
                         ", ".join(f"`{k!r}` {v}" for k, v in labs.most_common(5))])
    table(["run", "condition", "labels sent", "one character long", "most frequent"], rows)
    src(f"{RUNS}/ebnav_tau0.3_faithful1*/rep*/esca_*/*/*/steps.jsonl")
    para("**Paired check at step 0.** Every condition and repetition of an episode starts from the "
         "same pose, so at step 0 the frames differ only by the patch. We keep episodes where the patch "
         "was on screen at step 0 and compare step-0 outputs across repetitions: benign against benign "
         "gives the noise floor (gpt-4o proposes the detector query, so even benign graphs vary), benign "
         "against patched gives floor plus patch. Same box means IoU at least 0.5, or both empty.")
    h(2, "6. Did the planner's first action follow the patch?")
    para("Same pairs as section 5. If benign-against-patched agreement is no lower than "
         "benign-against-benign agreement, the patch did not move the first action beyond what "
         "resampling gpt-4o already does.")
    rows5, rows6 = [], []
    for run in ("ebnav_tau0.3_faithful1", "ebnav_tau0.3_faithful1_gdonly0"):
        idx = collections.defaultdict(lambda: collections.defaultdict(list))
        for e in eps:
            if e["run"] == run and e["cond"] in ("esca_benign", "esca_patch") and e["steps"]:
                idx[(e["eset"], e["ep"])][e["cond"]].append(e["steps"][0])
        stats = {k: collections.Counter() for k in ("bb", "bp")}; n_ep = 0; ious = {"bb": [], "bp": []}
        for key, c in idx.items():
            pats = [s for s in c["esca_patch"] if s.get("frame_patched")]
            if not pats or len(c["esca_benign"]) < 2: continue
            n_ep += 1
            for kind, prs in (("bb", itertools.combinations(c["esca_benign"], 2)),
                              ("bp", itertools.product(c["esca_benign"], pats))):
                for a, b in prs:
                    ra, rb = received(a), received(b); s = stats[kind]; s["pairs"] += 1
                    ba = (ra or {}).get("boxes") or []; bb_ = (rb or {}).get("boxes") or []
                    if not ba and not bb_: s["same_box"] += 1
                    elif ba and bb_:
                        v = iou(ba[0], bb_[0]); ious[kind].append(v); s["same_box"] += v >= 0.5
                    la = [l for l, _ in (ra or {}).get("objs", [])][:1]; lb = [l for l, _ in (rb or {}).get("objs", [])][:1]
                    s["same_label"] += la == lb
                    s["same_action"] += (a.get("action") or [None])[0] == (b.get("action") or [None])[0]
        bb, bp = stats["bb"], stats["bp"]
        mi = lambda k: f"{statistics.mean(ious[k]):.2f}" if ious[k] else "-"
        rows5.append([f"`{run}`", n_ep, f"{frac(bb['same_box'], bb['pairs'])} (mean IoU {mi('bb')})",
                      f"{frac(bp['same_box'], bp['pairs'])} (mean IoU {mi('bp')})",
                      frac(bb["same_label"], bb["pairs"]), frac(bp["same_label"], bp["pairs"])])
        rows6.append([f"`{run}`", n_ep, frac(bb["same_action"], bb["pairs"]), frac(bp["same_action"], bp["pairs"])])
    L_insert = L.index("## 6. Did the planner's first action follow the patch?") - 1
    block = []
    def tab(head, rows):
        block.append("| " + " | ".join(head) + " |"); block.append("|" + "---|" * len(head))
        for r in rows: block.append("| " + " | ".join(str(x) for x in r) + " |")
        block.append("")
    tab(["run", "episodes", "same box, benign vs benign", "same box, benign vs patched",
         "same label, benign vs benign", "same label, benign vs patched"], rows5)
    block.append(f"Source: `{RUNS}/ebnav_tau0.3_faithful1*/rep*/esca_*/*/*/steps.jsonl` (step 0)"); block.append("")
    L[L_insert:L_insert] = block
    table(["run", "episodes", "same first action, benign vs benign", "same first action, benign vs patched"], rows6)
    src(f"{RUNS}/ebnav_tau0.3_faithful1*/rep*/esca_*/*/*/steps.jsonl (step 0)")
    para("First action means the first action of the plan returned at step 0. Five episodes; pairs within "
         "an episode share its frames and are not independent, so treat these as a description, not a test. "
         "The staged runs cannot be checked this way: their episodes start with the target out of view.")

# --------------------------------------------------------------------------- 7. success rates

def success(all_eps):
    h(2, "7. Success rates: for the drift finding and threats to validity only")
    para("Success is `reached_target` in the staged runs (within 1 m of the target and nearer to it than "
         "to the distractor) and EB-Navigation's own `task_success` in the category runs.")
    staged = {}
    for f in sorted(glob.glob(f"{RUNS}/staged_*/summary.json")):
        run = f.split("/")[2]; t = collections.defaultdict(lambda: [0, 0]); reps = set()
        for key, eps in load(f).items():
            rep, cond = key.split("_", 1); reps.add(rep)
            for e in eps:
                c = t[(e["task"], cond)]; c[0] += e["reached_target"]; c[1] += 1
        staged[run] = (t, len(reps))
    rows = []
    for run, (t, nrep) in staged.items():
        for head in ("attribute", "relation"):
            cells = [frac(*t[(head, c)]) if t[(head, c)][1] else "-" for c in CONDS]
            if all(x == "-" for x in cells): continue
            rows.append([f"`{run}`", head, nrep] + cells)
    table(["run", "head", "reps"] + list(CONDS), rows)
    src(f"{RUNS}/staged_*/summary.json")
    rows = []
    ebnav = {}
    for f in sorted(glob.glob(f"{RUNS}/ebnav_*/summary.json")):
        run = f.split("/")[2]; d = load(f); ebnav[run] = d
        for eset, v in d.items():
            rows.append([f"`{run}`", eset] + [v.get(c, "-") for c in ("base_benign", "esca_benign", "esca_patch")])
    table(["run", "eval set", "base_benign", "esca_benign", "esca_patch"], rows)
    src(f"{RUNS}/ebnav_*/summary.json")
    para("`ebnav_tau0.4` predates the step logger. Its success rates are all that survive, so it cannot "
         "be analysed at the level of sections 3 to 6.")

    h(3, "7.1 Drift")
    # A repeat set is a group of runs that agree on everything we control for a head: the gate
    # threshold, top_k, the episode set and the instruction text. Grouping is derived from the run
    # metadata and the logs rather than asserted, so adding a run puts it in the right group.
    instr = collections.defaultdict(set)
    meta = collections.defaultdict(lambda: {"aggr": set(), "eps": set()})
    for f in glob.glob(f"{RUNS}/staged_*/rep*/*/*/run_meta.json"):
        m_ = load(f); run = f.split("/")[2]
        hd = "attribute" if m_["episode"].startswith("attr") else "relation"
        instr[(run, hd)].add(m_["instruction"])
        meta[(run, hd)]["aggr"].add(str(m_.get("aggr_thres")))
        meta[(run, hd)]["eps"].add(m_["episode"])
    # top_k is not written to the metadata, so we read the largest number of objects any one graph
    # of the run carried, which the gate caps at top_k.
    topk = collections.defaultdict(int)
    for e in all_eps:
        if not e["run"].startswith("staged_"): continue
        hd = "attribute" if e["ep"].startswith("attr") else "relation"
        for st in e["steps"]:
            r = received(st)
            if r: topk[(e["run"], hd)] = max(topk[(e["run"], hd)], len(r["objs"]))

    def key(run, hd):
        m_ = meta[(run, hd)]
        return (hd, tuple(sorted(m_["aggr"])), topk[(run, hd)], tuple(sorted(m_["eps"])),
                tuple(sorted(instr[(run, hd)])))

    groups = collections.defaultdict(list)
    for run, (t, nrep) in staged.items():
        for hd in ("attribute", "relation"):
            if t[(hd, "base_benign")][1] or t[(hd, "esca_benign")][1]:
                groups[key(run, hd)].append(run)

    para("Two runs count as a repeat of the same configuration only if, for that head, they share the "
         "gate threshold, `top_k`, the episode set and the instruction text. The groups below are "
         "derived from each run's metadata and logs, not asserted: `top_k` is read as the largest "
         "number of objects any one graph of the run carried, which the gate caps at `top_k`. Runs "
         "that differ only in the number of repetitions and in when they were executed are repeats.")

    def pc(t, hd, c):
        x = t[(hd, c)]
        return 100 * x[0] / x[1] if x[1] else None

    rows = []
    for k, runs_ in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        if len(runs_) < 2: continue
        hd, aggr, tk, epset, ins = k
        for c in ("base_benign", "esca_benign", "esca_patch"):
            cells = [frac(*staged[r][0][(hd, c)]) if staged[r][0][(hd, c)][1] else "-" for r in runs_]
            vals = [pc(staged[r][0], hd, c) for r in runs_ if staged[r][0][(hd, c)][1]]
            spread = f"{max(vals) - min(vals):.0f} pts" if len(vals) > 1 else "-"
            rows.append([hd, f"thres {'/'.join(aggr)}, top_k {tk}", c, "; ".join(cells), spread])
    table(["head", "configuration", "condition", "success per run, in run order", "spread"], rows)
    para("Runs in each group, in the order of the cells above: " + "; ".join(
        f"**{k[0]}** " + ", ".join(f"`{r}`" for r in v) for k, v in sorted(groups.items()) if len(v) > 1) + ".")

    def eff(t, hd, x, y):
        a_, b_ = pc(t, hd, x), pc(t, hd, y)
        return None if a_ is None or b_ is None else b_ - a_
    lines = []
    for k, runs_ in sorted(groups.items()):
        if len(runs_) < 2: continue
        hd = k[0]
        ben = [eff(staged[r][0], hd, "base_benign", "esca_benign") for r in runs_]
        atk = [eff(staged[r][0], hd, "esca_benign", "esca_patch") for r in runs_]
        f_ = lambda xs: ", ".join("n/a" if v is None else f"{v:+.0f}" for v in xs)
        lines.append(f"**{hd}**: scene-graph benefit {f_(ben)}; attack effect {f_(atk)}")
    para("Scene-graph benefit (esca_benign minus base_benign) and attack effect (esca_patch minus "
         "esca_benign) within each repeat group, in points, same run order: " + "; ".join(lines) +
         ". A quantity that changes sign inside a group is not resolved at these sample sizes.")
    para("Runs left in a group of one, so they cannot be read as repeats: " + "; ".join(
        f"`{v[0]}` ({k[0]})" for k, v in sorted(groups.items()) if len(v) == 1) + ".")
    out("**`base_benign` across sessions, grouped by identical instruction text.** `base_benign` sends no "
        "scene graph, so neither `aggr_thres` nor `top_k` can affect it; the instruction can."); out()
    rows = []
    for head in ("relation", "attribute"):
        groups = collections.defaultdict(list)
        for run, (t, _) in staged.items():
            if run.startswith("staged_A_") and t[(head, "base_benign")][1]:
                groups[tuple(sorted(instr[(run, head)]))].append(f"`{run}` {frac(*t[(head, 'base_benign')])}")
        for ins, cells in groups.items():
            rows.append([head, "; ".join(cells), (ins[0][:95] + "...") if ins else "?"])
    table(["head", "base_benign success per session", "instruction"], rows)
    x, y = ebnav["ebnav_tau0.3_faithful1"], ebnav["ebnav_tau0.3_faithful1_gdonly0"]
    para("**EB-Navigation `base_benign`, identical configuration in both runs** (`gd_only` only changes "
         "how a scene graph is built, and `base_benign` builds none): " + "; ".join(
            f"{e} {x[e]['base_benign']} then {y[e]['base_benign']}" for e in x) + ".")
    src(f"{RUNS}/staged_A_*/summary.json", f"{RUNS}/ebnav_tau0.3_faithful1*/summary.json")

    h(3, "7.2 Patch exposure")
    rows = []
    for f in sorted(glob.glob(f"{RUNS}/staged_*/summary.json")):
        run = f.split("/")[2]; d = load(f)
        for head in ("attribute", "relation"):
            for cond in ("esca_patch", "lang_patch"):
                fw = [e["frames_with_patch"] for k, v in d.items() if k.endswith(cond) for e in v if e["task"] == head]
                if fw: rows.append([f"`{run}`", head, cond, frac(sum(1 for v in fw if v == 0), len(fw)), statistics.median(fw)])
    table(["run", "head", "condition", "episodes where the patch never appeared", "median frames with the patch"], rows)
    para("An episode where the patch never appeared cannot be attributed to the attack either way.")
    src(f"{RUNS}/staged_*/summary.json")

# --------------------------------------------------------------------------- 8. do not use

def do_not_use():
    h(2, "8. Numbers from earlier documents not to use")
    table(["number", "where it appeared", "why not", "use instead"], [
        ["attribute isolation 100% (99/99)", "ALL_NUMBERS 2a, FINDINGS 1",
         "`run_attack --eval` does not count relation changes for attribute attacks", "section 2.1, `outcome_eval`"],
        ["Sep 4 staged run (every row)", "ALL_NUMBERS 3b", "no logs survive", "section 7"],
        ["Sep 7 staged attribute row", "ALL_NUMBERS 3b", "not in that run's `summary.json`, which kept relation only", "section 7"],
        ["attribute authored 32 / 40 / 38%", "FINDINGS 2", "pools the Sep 4 and Sep 7 runs above", "section 7"],
        ["relation region shape: 19 edges under masks, 6 under rectangles", "FINDINGS 3, 4",
         "`mask_mismatch_probe.py` and its output are lost", "nothing here isolates region shape: 2.1 against 2.5 also changes detection. Rerun a probe if the paper needs it"],
        ["base_benign 0% to 44% across six sessions", "earlier REPLICATION.md",
         "two of the six sessions have no published logs, and the 44% session used a different attribute instruction",
         "section 7.1, relation"],
        ["attribute base_benign 44% then 20% as drift between identical runs", "ALL_NUMBERS 3c, earlier chat, earlier NUMBERS.md",
         "the two runs gave the attribute tasks different instructions", "section 7.1, relation rows only"],
        ["relation flip 69% against 18%", "earlier chat", "compared a pair-level definition with an edge-survival one", "section 2.5"],
        ["attribute stealth 122% worse", "earlier chat", "patch sizes differed by 2.78 times in area", "section 2.7"],
        ["deployment isolation, graph-wide", "earlier chat", "counted collateral across the whole graph, and `top_k=1` leaves nothing to isolate", "not reported"],
        ["SGClip grounds 2/4 blind poses", "`output/probes/sgclip_poses.log` summary line",
         "FloorPlan207's grounded label is `cabinet`", "1/4, section 2.6"],
        ["stealth and strength trade off, r = -0.53", "earlier chat",
         "inputs and sign convention not recorded, and pooling across heads mixes head differences into it", "section 2.7, within-head r"],
        ["`lang_benign`, `lang_patch` in `staged_A_donothing_tau0.4`", "the run folder only",
         "a leftover one-repetition test (6 episodes per rep, no step logs), never in `summary.json`", "not reported"],
    ])
    para("The Sep 7 attribute episodes do still have step logs in `staged_A_donothing_tau0.4`, so they are "
         "used in sections 3 and 4. Only their success outcomes are missing.")

    h(3, "8.1 Earlier numbers this file confirms")
    table(["number", "where it appeared", "confirmed in"], [
        ["attribute flip 92% (59/64) on held-out scenarios", "FINDINGS 1", "section 2.2"],
        ["deployment: `television` with `displaying` in 46% of calls, `(pot, on, counter)` in 6%", "FINDINGS 3, 4", "section 4, `staged_A_donothing_tau0.4`"],
        ["no scene graph in 49% (strict) and 12% (loose) of attribute calls", "FINDINGS 5", "section 3"],
        ["`displaying` is on the television 95% (strict), 50% (loose)", "FINDINGS 5", "section 4, last column"],
        ["`on` edges from a pot 80% (strict), 55% (loose)", "FINDINGS 5", "section 4, last column"],
        ["direction 53%, both directions 68%, argument 26%", "FINDINGS 4", "section 2.6"],
        ["gpt-4o reads the screen 17/20; blind poses 4/10; SGClip at those poses 1/4", "FINDINGS 4", "section 2.6"],
        ["detector-driven flip 88 / 93 / 86%, random 32 / 3 / 86%", "ALL_NUMBERS 2b", "section 2.5, `top_k=15`"],
        ["LASER-style relation 59%, 22% trained, 48% random", "ALL_NUMBERS 2c", "section 2.6"],
        ["same configuration twice: relation benefit -16 then +13, attack +2 then -18", "ALL_NUMBERS 3c", "section 7.1"],
    ])

# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="output/NUMBERS.md")
    args = ap.parse_args()
    eps = episodes()
    provenance(); configuration(); model_level(); content_ablation(); planner_receives(eps); patched_graphs(eps); step0_pairs(eps)
    success(eps); do_not_use()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(L).rstrip() + "\n")
    print(f"wrote {args.out}  ({len(L)} lines, {len(eps)} episode logs read)")

if __name__ == "__main__":
    main()
