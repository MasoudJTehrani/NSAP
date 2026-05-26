"""
Scene graph edit distance — proves attack isolation.

Computes the structural difference between a clean and attacked scene graph.
Breakdowns:
  - node label substitution (category changed)
  - node attribute substitution (attribute changed)
  - edge label substitution (relation changed)
  - node insertion / deletion (object appeared or disappeared)

An isolation score is derived: fraction of total edits attributable to the attacked head.
"""

from typing import Dict, Tuple
from dataclasses import dataclass, field
from functions.eval.scene_graph import SceneGraph


@dataclass
class EditDistanceResult:
    # Raw operation counts
    category_substitutions: int = 0    # object changed class
    attribute_substitutions: int = 0   # object changed attribute
    relation_substitutions: int = 0    # relation label changed
    node_insertions: int = 0           # new object appeared after patch
    node_deletions: int = 0            # object disappeared after patch

    @property
    def total(self) -> int:
        return (
            self.category_substitutions +
            self.attribute_substitutions +
            self.relation_substitutions +
            self.node_insertions +
            self.node_deletions
        )

    def isolation_score(self, attack_mode: str) -> float:
        """
        Fraction of total edits that are on the attacked head.
        Value of 1.0 = perfectly isolated; 0.0 = attack spilled everywhere.
        """
        if self.total == 0:
            return 1.0   # no edits at all — clean isolation

        if attack_mode == 'category':
            attacked_edits = self.category_substitutions + self.node_insertions + self.node_deletions
        elif attack_mode == 'attribute':
            attacked_edits = self.attribute_substitutions
        elif attack_mode == 'relation':
            attacked_edits = self.relation_substitutions
        else:
            return 0.0

        return attacked_edits / self.total

    def summary(self, attack_mode: str) -> str:
        return (
            f"EditDistance(total={self.total} | "
            f"cat_sub={self.category_substitutions} "
            f"attr_sub={self.attribute_substitutions} "
            f"rel_sub={self.relation_substitutions} "
            f"ins={self.node_insertions} "
            f"del={self.node_deletions} | "
            f"isolation={self.isolation_score(attack_mode):.2f})"
        )


def compute_edit_distance(
    sg_clean: SceneGraph,
    sg_attacked: SceneGraph,
) -> EditDistanceResult:
    """
    Compute the edit distance between two scene graphs.

    Operations:
      - Category substitution: same obj_id, different category label
      - Attribute substitution: same obj_id, different attribute label
      - Relation substitution: same (subj, obj) pair, different relation label
      - Node insertion: obj_id present in sg_attacked but not sg_clean
      - Node deletion: obj_id present in sg_clean but not sg_attacked

    Args:
        sg_clean:    SceneGraph from the original un-patched frames
        sg_attacked: SceneGraph from the patched frames

    Returns:
        EditDistanceResult
    """
    result = EditDistanceResult()

    clean_obj_ids    = set(sg_clean.categories.keys())
    attacked_obj_ids = set(sg_attacked.categories.keys())

    # Node insertions and deletions
    result.node_insertions = len(attacked_obj_ids - clean_obj_ids)
    result.node_deletions  = len(clean_obj_ids - attacked_obj_ids)

    # Category and attribute substitutions for objects present in both
    shared_obj_ids = clean_obj_ids & attacked_obj_ids
    for obj_id in shared_obj_ids:
        clean_cat  = sg_clean.categories.get(obj_id)
        attack_cat = sg_attacked.categories.get(obj_id)
        if clean_cat is not None and attack_cat is not None and clean_cat != attack_cat:
            result.category_substitutions += 1

        clean_attr  = sg_clean.attributes.get(obj_id)
        attack_attr = sg_attacked.attributes.get(obj_id)
        if clean_attr is not None and attack_attr is not None and clean_attr != attack_attr:
            result.attribute_substitutions += 1

    # Relation substitutions for pairs present in both
    clean_rel_pairs    = set(sg_clean.relations.keys())
    attacked_rel_pairs = set(sg_attacked.relations.keys())
    shared_pairs       = clean_rel_pairs & attacked_rel_pairs

    for pair in shared_pairs:
        clean_rel  = sg_clean.relations[pair]
        attack_rel = sg_attacked.relations[pair]
        if clean_rel != attack_rel:
            result.relation_substitutions += 1

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
    Compute edit distance and isolation score over a list of test clips.

    Returns:
        dict with:
            'mean_edit_distance':       float
            'mean_isolation_score':     float
            'mean_category_sub':        float
            'mean_attribute_sub':       float
            'mean_relation_sub':        float
            'per_clip':                 list of EditDistanceResult
    """
    from functions.attack.patch import paste_patch, compute_patch_size
    from functions.eval.scene_graph import predict_scene_graph

    all_edits = []

    for item in clip_items:
        if item is None:
            continue

        clean_frames   = item['frames']
        masks          = item['masks']
        bboxes         = item['bboxes']
        gpt_spec       = item['gpt_spec']

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

        ed = compute_edit_distance(sg_clean, sg_attacked)
        all_edits.append(ed)

    if not all_edits:
        return {}

    n = len(all_edits)
    return {
        'mean_edit_distance':   sum(e.total for e in all_edits) / n,
        'mean_isolation_score': sum(e.isolation_score(config.attack_mode) for e in all_edits) / n,
        'mean_category_sub':    sum(e.category_substitutions for e in all_edits) / n,
        'mean_attribute_sub':   sum(e.attribute_substitutions for e in all_edits) / n,
        'mean_relation_sub':    sum(e.relation_substitutions for e in all_edits) / n,
        'mean_node_ins':        sum(e.node_insertions for e in all_edits) / n,
        'mean_node_del':        sum(e.node_deletions  for e in all_edits) / n,
        'per_clip':             all_edits,
    }
