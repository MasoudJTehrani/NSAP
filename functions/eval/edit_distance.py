"""
Scene graph edit distance — proves attack isolation.

All metrics are computed on the TARGET object(s) only:
  - category/attribute attack: objects whose clean-SG category matches target_category
  - relation attack: (subj, obj) pairs whose clean-SG relation matches target_relation

Per-object rates are normalised by n_target_objects so they are comparable
across clips regardless of how many objects the clip contains.
"""

from typing import Dict, Tuple
from dataclasses import dataclass, field
from functions.eval.scene_graph import SceneGraph


@dataclass
class EditDistanceResult:
    # How many target objects were found in the clean scene graph
    n_target_objects: int = 0
    # How many target relations were found (relation mode only)
    n_target_relations: int = 0

    # Raw change counts on target objects
    category_changes: int = 0    # target objects whose category label changed
    attribute_changes: int = 0   # target objects whose attribute label changed
    relation_changes: int = 0    # target relations that changed (relation mode)
    target_disappeared: int = 0  # target objects absent from attacked SG

    # ── rates (normalised by n_target_objects / n_target_relations) ──────────

    @property
    def _obj_denom(self) -> int:
        return max(self.n_target_objects, 1)

    @property
    def _rel_denom(self) -> int:
        return max(self.n_target_relations, 1)

    @property
    def category_change_rate(self) -> float:
        return self.category_changes / self._obj_denom

    @property
    def attribute_change_rate(self) -> float:
        return self.attribute_changes / self._obj_denom

    @property
    def relation_change_rate(self) -> float:
        return self.relation_changes / self._rel_denom

    @property
    def disappear_rate(self) -> float:
        return self.target_disappeared / self._obj_denom

    @property
    def total_changes(self) -> int:
        return (
            self.category_changes
            + self.attribute_changes
            + self.relation_changes
            + self.target_disappeared
        )

    def isolation_score(self, attack_mode: str) -> float:
        """
        Fraction of changes on target objects that hit the attacked head.
        1.0 = perfectly isolated; 0.0 = all changes were collateral.
        Returns 1.0 when there are no changes (nothing to isolate).
        """
        if self.total_changes == 0:
            return 1.0
        if attack_mode == 'category':
            return (self.category_changes + self.target_disappeared) / self.total_changes
        if attack_mode == 'attribute':
            return self.attribute_changes / self.total_changes
        if attack_mode == 'relation':
            return self.relation_changes / self.total_changes
        return 0.0


def _match(label: str, target: str) -> bool:
    return target.lower() in label.lower()


def compute_edit_distance(
    sg_clean: SceneGraph,
    sg_attacked: SceneGraph,
    attack_mode: str = 'category',
    target_category: str = '',
    target_relation: str = '',
) -> EditDistanceResult:
    """
    Compute edit distance restricted to the target object(s).

    Args:
        sg_clean:        SceneGraph from original un-patched frames
        sg_attacked:     SceneGraph from patched frames
        attack_mode:     'category', 'attribute', or 'relation'
        target_category: category string being attacked (e.g. 'camera')
        target_relation: relation string being attacked (e.g. 'wearing')

    Returns:
        EditDistanceResult with per-target-object rates
    """
    result = EditDistanceResult()

    # ── Identify target objects ───────────────────────────────────────────────
    if attack_mode in ('category', 'attribute'):
        target_obj_ids = {
            obj_id for obj_id, cat in sg_clean.categories.items()
            if _match(cat, target_category)
        } if target_category else set(sg_clean.categories.keys())

        result.n_target_objects = len(target_obj_ids)

        attacked_obj_ids = set(sg_attacked.categories.keys())
        result.target_disappeared = len(target_obj_ids - attacked_obj_ids)

        for obj_id in target_obj_ids & attacked_obj_ids:
            clean_cat    = sg_clean.categories.get(obj_id)
            attacked_cat = sg_attacked.categories.get(obj_id)
            if clean_cat and attacked_cat and clean_cat != attacked_cat:
                result.category_changes += 1

            clean_attr    = sg_clean.attributes.get(obj_id)
            attacked_attr = sg_attacked.attributes.get(obj_id)
            if clean_attr and attacked_attr and clean_attr != attacked_attr:
                result.attribute_changes += 1

    elif attack_mode == 'relation':
        target_pairs = {
            pair for pair, rel in sg_clean.relations.items()
            if _match(rel, target_relation)
        } if target_relation else set(sg_clean.relations.keys())

        # Optionally filter by subject category
        if target_category:
            target_pairs = {
                pair for pair in target_pairs
                if _match(sg_clean.categories.get(pair[0], ''), target_category)
            }

        result.n_target_relations = len(target_pairs)
        # Use target subjects as the "target objects" for rate normalisation
        target_obj_ids = {pair[0] for pair in target_pairs}
        result.n_target_objects = len(target_obj_ids)

        attacked_obj_ids = set(sg_attacked.categories.keys())
        result.target_disappeared = len(target_obj_ids - attacked_obj_ids)

        for pair in target_pairs:
            clean_rel    = sg_clean.relations.get(pair)
            attacked_rel = sg_attacked.relations.get(pair)
            if clean_rel and attacked_rel and clean_rel != attacked_rel:
                result.relation_changes += 1

        for obj_id in target_obj_ids & attacked_obj_ids:
            clean_cat    = sg_clean.categories.get(obj_id)
            attacked_cat = sg_attacked.categories.get(obj_id)
            if clean_cat and attacked_cat and clean_cat != attacked_cat:
                result.category_changes += 1

            clean_attr    = sg_clean.attributes.get(obj_id)
            attacked_attr = sg_attacked.attributes.get(obj_id)
            if clean_attr and attacked_attr and clean_attr != attacked_attr:
                result.attribute_changes += 1

    return result


def evaluate_isolation(
    vine_model,
    clip_items: list,
    patch,
    anchor,
    config,
    device,
) -> dict:
    """
    Compute target-object edit distance and isolation over test clips.

    Returns dict with per-rate metrics (all normalised by n_target_objects):
        mean_n_target_objects     float  — avg target objects per clip
        mean_category_change_rate float  — fraction of target objs whose category changed
        mean_attribute_change_rate float — fraction of target objs whose attribute changed
        mean_relation_change_rate  float — fraction of target relations that changed
        mean_disappear_rate        float — fraction of target objs that disappeared
        mean_isolation_score       float — fraction of changes on the attacked head
        per_clip                   list[EditDistanceResult]
    """
    from functions.attack.patch import paste_patch, compute_patch_size
    from functions.eval.scene_graph import predict_scene_graph

    all_edits = []

    for item in clip_items:
        if item is None:
            continue

        clean_frames = item['frames']
        masks        = item['masks']
        bboxes       = item['bboxes']
        gpt_spec     = item['gpt_spec']

        patched_frames = []
        for frame_id, frame in enumerate(clean_frames):
            frame_bboxes = bboxes.get(frame_id, {})
            patched      = frame.clone()
            for obj_id, bbox in frame_bboxes.items():
                ph, pw  = compute_patch_size(bbox, config.coverage_ratio)
                patched = paste_patch(
                    patched.to(device), patch.pixels_hwc,
                    anchor.cx, anchor.cy, ph, pw,
                )
            patched_frames.append(patched.cpu())

        sg_clean    = predict_scene_graph(vine_model, clean_frames,   masks, bboxes, gpt_spec, device)
        sg_attacked = predict_scene_graph(vine_model, patched_frames, masks, bboxes, gpt_spec, device)

        ed = compute_edit_distance(
            sg_clean, sg_attacked,
            attack_mode      = config.attack_mode,
            target_category  = config.target_category,
            target_relation  = config.target_relation,
        )
        all_edits.append(ed)

    if not all_edits:
        return {}

    n = len(all_edits)
    return {
        'mean_n_target_objects':      sum(e.n_target_objects       for e in all_edits) / n,
        'mean_category_change_rate':  sum(e.category_change_rate   for e in all_edits) / n,
        'mean_attribute_change_rate': sum(e.attribute_change_rate  for e in all_edits) / n,
        'mean_relation_change_rate':  sum(e.relation_change_rate   for e in all_edits) / n,
        'mean_disappear_rate':        sum(e.disappear_rate         for e in all_edits) / n,
        'mean_isolation_score':       sum(e.isolation_score(config.attack_mode) for e in all_edits) / n,
        'per_clip':                   all_edits,
    }
