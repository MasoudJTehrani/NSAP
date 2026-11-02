"""Per-step logging for the NSAP deployment benchmarks.

Every step the robot takes is written to disk twice, into the episode's own directory:

    <episode>/steps.jsonl   one JSON object per step, complete and machine-readable
    <episode>/trace.txt     the same content, human-readable
    <episode>/run_meta.json the run's configuration (condition, rep, tau, model, patch, ...)

WHY jsonl: trace.txt truncates the MLLM output and stringifies the scene graph, so it
cannot be analysed programmatically. steps.jsonl keeps the FULL scene graph (grounded
objects with scores, grounded relations, per-box top-k, the classification vocabulary),
the FULL SG text handed to the planner, and the FULL MLLM response.

WHY truncate: trace.txt used to be opened in append mode, so re-running a benchmark
silently interleaved two experiments in one file. A real run was found holding 7 steps at
aggr_thres=0.3 and 5 at 0.4, with nothing marking the boundary. Both files are now
truncated when an episode starts, and run_meta.json records which run produced them.
"""

import json
import os
import time



def _rel_scores(planner, top=10):
    """-> {subject name: [[object name, p(rel), p(not rel), rel], ...]} sorted by p(rel)."""
    scored = getattr(planner, "_nsap_rel_scored", None)
    names = getattr(planner, "_nsap_rel_names", None)
    if not scored or not names:
        return None
    pos, neg = {}, {}
    for (pair, rel), p in scored.items():
        try:
            s_, o_ = pair
        except Exception:
            continue
        (neg if str(rel).startswith("not ") else pos)[(s_, o_, str(rel).replace("not ", "", 1))] = float(p)
    out = {}
    for (s_, o_, rel), p in pos.items():
        sn, on = names.get(s_), names.get(o_)
        if sn is None or on is None or sn == on:
            continue
        out.setdefault(str(sn), []).append([str(on), round(p, 4), round(neg.get((s_, o_, rel), float("nan")), 4), rel])
    for k in out:
        out[k] = sorted(out[k], key=lambda r: -r[1])[:top]
    return out or None

def _jsonable(o, _depth=0):
    """Best-effort conversion to something json.dump can write. Never raises."""
    if _depth > 12:
        return "<max-depth>"
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v, _depth + 1) for v in o]
    # numpy / torch scalars and arrays
    for attr in ("tolist", "item"):
        if hasattr(o, attr):
            try:
                return _jsonable(getattr(o, attr)(), _depth + 1)
            except Exception:
                pass
    try:
        return str(o)
    except Exception:
        return "<unprintable>"


def _g(planner, name, default=None):
    return getattr(planner, name, default)


class StepLogger:
    """One instance per episode. Truncates its files on construction."""

    def __init__(self, viz_dir, meta=None):
        self.dir = viz_dir
        os.makedirs(viz_dir, exist_ok=True)
        self.jsonl = os.path.join(viz_dir, "steps.jsonl")
        self.trace = os.path.join(viz_dir, "trace.txt")
        self.meta = dict(meta or {})
        self.meta.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.n = 0
        # TRUNCATE: a fresh episode must not inherit a previous run's steps.
        open(self.jsonl, "w").close()
        with open(self.trace, "w") as f:
            f.write(f"# NSAP step trace\n# {json.dumps(_jsonable(self.meta))}\n\n")
        with open(os.path.join(viz_dir, "run_meta.json"), "w") as f:
            json.dump(_jsonable(self.meta), f, indent=2)

    def log(self, planner, step, instruction, img_path, action, reasoning,
            patched_frames=None, frame_patched=None, extra=None):
        """Record one decision. `extra` carries runner-specific fields (distances, poses)."""
        rec = {
            "step": step,
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame": os.path.basename(img_path) if img_path else None,
            "frame_patched": frame_patched,
            "patched_frames_so_far": patched_frames,
            "instruction": instruction,
            "aggr_thres": _g(planner, "aggr_thres"),
            "concepts_requested": _g(planner, "_nsap_last_concepts"),
            "detector": {
                "query": _g(planner, "_nsap_gd_query"),
                "n_boxes": _g(planner, "_nsap_gd_nboxes"),
                "labels": _g(planner, "_nsap_gd_labels"),
            },
            "sgclip": {
                "grounded_objects": _g(planner, "_nsap_last_labels"),
                "grounded_relations": _g(planner, "_nsap_last_relations"),
                "per_box_topk": _g(planner, "_nsap_predicted_name"),
                "query_names": _g(planner, "_nsap_query_names"),
                # Per subject, every candidate object scored for the requested relation,
                # with the probability of the relation and of its negation. This is what
                # decides whether an edge is surfaced, so a missing edge can be traced to
                # the score rather than guessed at.
                "relation_scores": _rel_scores(planner),
            },
            "scene_graph_text": _g(planner, "_nsap_last_sg_text"),
            "mllm": reasoning if isinstance(reasoning, dict) else {"raw": reasoning},
            "action": action,
        }
        if extra:
            rec.update(extra)
        rec = _jsonable(rec)

        with open(self.jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._append_trace(rec)
        self.n += 1
        return rec

    def _append_trace(self, rec):
        d, s, m = rec.get("detector", {}), rec.get("sgclip", {}), rec.get("mllm", {})
        L = ["=" * 84,
             f"STEP {rec['step']}   frame={rec['frame']}   "
             f"(patched this frame: {rec.get('frame_patched')}; "
             f"patched frames so far: {rec.get('patched_frames_so_far')})",
             f"INSTRUCTION: {rec.get('instruction')}"]
        c = rec.get("concepts_requested")
        if c:
            L.append(f"[1] MLLM asked SGClip to ground -> objects={c.get('objects')}  "
                     f"attributes={c.get('attributes')}  relations={c.get('relations')}")
        L.append(f"[1b] detector query={d.get('query')} -> GroundingDINO found "
                 f"{d.get('n_boxes')} boxes: {d.get('labels')}")
        L.append(f"[2] SGClip grounded objects:   {s.get('grounded_objects')}")
        L.append(f"[2b] SGClip top-k per box (prob,name) [aggr_thres={rec.get('aggr_thres')}]: "
                 f"{s.get('per_box_topk')}")
        L.append(f"[2b] classify vocab (query_names): {s.get('query_names')}")
        L.append(f"[2] SGClip grounded relations: {s.get('grounded_relations')}")
        L.append(f"[3] SG TEXT fed to the MLLM:\n    {rec.get('scene_graph_text')}")
        for k in ("target_dist", "distractor_dist", "agent_position", "reached_target",
                  "reached_distractor"):
            if k in rec:
                L.append(f"[3b] {k}: {rec[k]}")
        if isinstance(m, dict):
            for k, v in m.items():
                L.append(f"[4] MLLM {k}: {str(v)[:700]}")
        L.append(f"[5] ACTION chosen: {rec.get('action')}")
        L.append(f"    (full, untruncated record for this step: steps.jsonl line {self.n + 1})")
        with open(self.trace, "a") as f:
            f.write("\n".join(L) + "\n\n")
