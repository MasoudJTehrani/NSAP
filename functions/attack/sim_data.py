"""
SimAttackDataset — adapter from a SceneGraphBench (AI2-THOR) dataset into the NSAP
attack pipeline.

This is NSAP's dataset. `__getitem__` returns an item dict, and
`build_instance_list` / `build_relation_instance_list` return the flat instance
dicts that train.py and the eval functions consume.
The only per-instance difference is `_png_path` (sim frames are PNGs), which
data.load_frame handles.

Because the sim ships pixel-perfect ground truth, target objects are selected by
TRUE category (exact GT match, no over-segmentation guessing). The clean-prediction
gate in train.py still runs on top, keeping only objects VINE currently reads
correctly — so training focuses on objects the model actually grounds.
"""

import os
import json
import math
from functools import lru_cache

import numpy as np
import torch

# Candidate vocabularies VINE classifies against (the gpt_spec keyword lists).
# NOTE: the ATTACKED attribute/relation word need NOT appear here. The attribute
# and relation attacks use a 2-way contrast [k, "not k"] built from the target
# word (e.g. "displaying"), not a multi-way argmax over these lists. These lists
# are only the multi-way candidate set for eval/isolation of the NON-attacked heads.
ATTRIBUTE_WORDS = ['open', 'closed', 'on', 'off', 'sliced', 'cooked',
                   'clean', 'dirty', 'filled', 'broken']
RELATION_WORDS = ['on', 'in', 'near']


@lru_cache(maxsize=2048)
def _instance_array(path: str):
    from PIL import Image
    return np.array(Image.open(path).convert('RGB'))   # HxWx3 uint8


def _mask_from_color(inst_arr, color):
    """Recover an object's boolean mask from the instance map + its RGB color."""
    m = np.all(inst_arr == np.array(color, dtype=inst_arr.dtype), axis=-1)
    return torch.from_numpy(m) if m.any() else None     # bool [H,W] or None


def _bbox_dict(b):
    x1, y1, x2, y2 = b
    return {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}


def explode_to_frames(item):
    """Split a multi-frame sim item (one scenario) into one single-frame item per
    frame, so eval scores EACH frame as an independent sample (far more statistical
    power than one selected frame per scenario). Carries gt_categories through."""
    if item is None:
        return []
    out = []
    for fid, frame in enumerate(item['frames']):
        if fid not in item['masks'] or not item['masks'][fid]:
            continue
        out.append({
            'data_id':  f"{item['data_id']}_f{fid}",
            'frames':   [frame],
            'masks':    {0: item['masks'][fid]},
            'bboxes':   {0: item['bboxes'].get(fid, {})},
            'gpt_spec': item['gpt_spec'],
            'gt_categories': item.get('gt_categories', {}),
        })
    return out


class SimAttackDataset:
    """Loads a SceneGraphBench dataset — NSAP's training/eval data source."""

    TRAIN_RATIO = 0.8

    def __init__(self, sim_data_dir, target_category=None, target_attribute=None,
                 target_relation=None, split=None, max_clips=0, only_scenarios=None,
                 data_source=None):
        self.root = sim_data_dir
        manifest_path = os.path.join(sim_data_dir, 'manifest.json')
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"No SceneGraphBench dataset at '{sim_data_dir}' (missing manifest.json). "
                f"Get it with `python NSAP.py --with-dataset` (downloads to dataset/sim/) "
                f"or generate one from the SceneGraphBench/ submodule, then pass --sim-data."
            )
        manifest = json.load(open(manifest_path))
        entries = manifest['scenarios']
        if data_source:   # restrict to one merge source (unified manifest 'source' tag)
            entries = [e for e in entries if e.get('source') == data_source]

        if only_scenarios is not None:
            # Restrict to an explicit set of scenario ids (used by the per-scenario
            # chameleon experiment). Ignores category filter and split.
            want = set(only_scenarios)
            entries = [e for e in entries if e['scenario_id'] in want]
            entries = sorted(entries, key=lambda e: e['scenario_id'])
        else:
            # Each scenario is built around one target object; filter by its category.
            # EXACT match (not substring): sim categories are exact words, and 'pot'
            # must not match 'potato'.
            if target_category:
                entries = [e for e in entries
                           if e['target_category'].lower() == target_category.lower()]
            entries = sorted(entries, key=lambda e: e['scenario_id'])

            # split=None uses ALL data (no train/test split).
            n_train = math.floor(self.TRAIN_RATIO * len(entries))
            if split == 'train':
                entries = entries[:n_train]
            elif split == 'test':
                entries = entries[n_train:]
            if max_clips and len(entries) > max_clips:
                entries = entries[:max_clips]

        self.scenarios = [os.path.join(sim_data_dir, 'scenarios', e['scenario_id'])
                          for e in entries]
        if split is not None or only_scenarios is not None:
            tag = f"scenarios={only_scenarios}" if only_scenarios is not None else f"split '{split}'"
            print(f"  [sim] {tag}: {len(self.scenarios)} scenarios")

    def __len__(self):
        return len(self.scenarios)

    # ── shared helpers ───────────────────────────────────────────────────────
    def _load_sg(self, sdir):
        return json.load(open(os.path.join(sdir, 'scene_graph.json')))

    def _gpt_spec(self, sg):
        """consts = categories present (deployment-faithful candidate set);
        unary/binary = attribute/relation vocabularies; binary_predicates from GT."""
        cats = sorted({o['category'] for fr in sg['frames'] for o in fr['objects']})
        seen, bin_preds = set(), []
        for fr in sg['frames']:
            id2cat = {o['id']: o['category'] for o in fr['objects']}
            for r in fr['relations']:
                sc, oc = id2cat.get(r['subject']), id2cat.get(r['object'])
                if sc and oc and (r['predicate'], sc, oc) not in seen:
                    seen.add((r['predicate'], sc, oc))
                    bin_preds.append([r['predicate'], sc, oc])
        return {'video_id': sg['scenario_id'], 'consts': cats,
                'unary_kws': ATTRIBUTE_WORDS, 'binary_kws': RELATION_WORDS,
                'binary_predicates': bin_preds}

    # ── eval item (frames + masks + per-frame bboxes for one scenario) ───────
    def __getitem__(self, idx):
        from PIL import Image
        sdir = self.scenarios[idx]
        sg = self._load_sg(sdir)
        frames, masks, bboxes = [], {}, {}
        for fi, fr in enumerate(sg['frames']):
            arr = np.array(Image.open(os.path.join(sdir, fr['frame'])).convert('RGB'))
            frames.append(torch.from_numpy(arr.astype(np.float32)))
            inst_arr = _instance_array(os.path.join(sdir, fr['instance_map']))
            masks[fi], bboxes[fi] = {}, {}
            for o in fr['objects']:
                m = _mask_from_color(inst_arr, o['color'])
                if m is None:
                    continue
                masks[fi][o['id']] = m
                bboxes[fi][o['id']] = _bbox_dict(o['bbox'])
        # Ground-truth category per object id (stable across frames), so eval can
        # anchor targets on the REAL object, not on what VINE happens to predict.
        gt_categories = {o['id']: o['category'] for fr in sg['frames'] for o in fr['objects']}
        return {'data_id': sg['scenario_id'], 'frames': frames,
                'masks': masks, 'bboxes': bboxes, 'gpt_spec': self._gpt_spec(sg),
                'gt_categories': gt_categories}

    # ── training instances (mirrors build_instance_list) ─────────────────────
    def build_instance_list(self, target_category, target_predicate=None):
        instances = []
        for sdir in self.scenarios:
            sg = self._load_sg(sdir)
            gpt_spec = self._gpt_spec(sg)
            if target_category and not any(c.lower() == target_category.lower()
                                           for c in gpt_spec['consts']):
                continue
            for fi, fr in enumerate(sg['frames']):
                objs = [o for o in fr['objects']
                        if not target_category or o['category'].lower() == target_category.lower()]
                if target_predicate:
                    # substring so a target matches its GT form ('open' in 'opened',
                    # 'on' in 'turned on'); GT attributes are clean words, so this
                    # does not over-match in practice.
                    objs = [o for o in objs
                            if any(target_predicate.lower() in a.lower() for a in o.get('attributes', []))]
                if not objs:
                    continue
                inst_arr = _instance_array(os.path.join(sdir, fr['instance_map']))
                png = os.path.join(sdir, fr['frame'])
                for o in objs:
                    m = _mask_from_color(inst_arr, o['color'])
                    if m is None:
                        continue
                    instances.append({
                        'data_id': sg['scenario_id'], 'frame_id': fi, 'obj_id': o['id'],
                        '_png_path': png,
                        'mask': m, 'bbox': _bbox_dict(o['bbox']), 'gpt_spec': gpt_spec,
                    })
        return instances

    # ── relation instances (mirrors build_relation_instance_list) ────────────
    def build_relation_instance_list(self, target_relation, subject_category=None):
        instances = []
        for sdir in self.scenarios:
            sg = self._load_sg(sdir)
            gpt_spec = self._gpt_spec(sg)
            if not any(target_relation.lower() in p[0].lower() for p in gpt_spec['binary_predicates']):
                continue
            for fi, fr in enumerate(sg['frames']):
                id2o = {o['id']: o for o in fr['objects']}
                png = os.path.join(sdir, fr['frame'])
                inst_arr = None
                for r in fr['relations']:
                    if target_relation.lower() not in r['predicate'].lower():
                        continue
                    so, oo = id2o.get(r['subject']), id2o.get(r['object'])
                    if not so or not oo:
                        continue
                    if subject_category and so['category'].lower() != subject_category.lower():
                        continue
                    if inst_arr is None:
                        inst_arr = _instance_array(os.path.join(sdir, fr['instance_map']))
                    sm = _mask_from_color(inst_arr, so['color'])
                    om = _mask_from_color(inst_arr, oo['color'])
                    if sm is None or om is None:
                        continue
                    instances.append({
                        'data_id': sg['scenario_id'], 'frame_id': fi,
                        'subj_obj_id': so['id'], 'obj_obj_id': oo['id'],
                        '_png_path': png,
                        'subj_mask': sm, 'obj_mask': om,
                        'subj_bbox': _bbox_dict(so['bbox']), 'obj_bbox': _bbox_dict(oo['bbox']),
                        'gpt_spec': gpt_spec,
                    })
        return instances
