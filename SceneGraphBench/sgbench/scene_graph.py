"""
Ground-truth scene-graph extraction from one AI2-THOR event.

Categories come from objectType, attributes from the simulator's state flags, and
relations from receptacle links (+ proximity). Nothing here predicts anything — it
reads the simulator's own ground truth.
"""

import numpy as np

from .vocab import thor_type_to_word, ATTRIBUTE_FLAGS, ENCLOSING_RECEPTACLES


def build_id_map(metadata_objects):
    """Stable thor objectId(str) -> int id, fixed for a whole scenario so relations
    reference consistent ids across frames. Sorted for determinism."""
    ids = sorted(o['objectId'] for o in metadata_objects)
    return {oid: i for i, oid in enumerate(ids)}


def object_attributes(obj):
    """Active attribute words for an object, gated by its capability flags."""
    words = []
    for flag, (capability, w_true, w_false) in ATTRIBUTE_FLAGS.items():
        if not obj.get(capability, False):
            continue
        if flag not in obj or obj[flag] is None:
            continue
        words.append(w_true if obj[flag] else w_false)
    return [w for w in words if w]


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def extract_frame_graph(event, id_map, min_area=400, near_dist=0.5):
    """
    Build the ground-truth scene graph for the current frame.

    Returns dict:
      {
        "objects": [ {id, category, thor_id, color:[r,g,b], bbox:[x1,y1,x2,y2],
                      attributes:[...]} , ... ],     # visible + prominent only
        "relations": [ {subject:int, predicate:str, object:int}, ... ]
      }
    Only objects that are visible AND occupy >= min_area pixels are emitted (the
    regime a robot actually perceives). Relations are emitted when BOTH endpoints
    are present in this frame.
    """
    instance_masks = getattr(event, 'instance_masks', {}) or {}
    id2color = getattr(event, 'object_id_to_color', {}) or {}
    meta_by_id = {o['objectId']: o for o in event.metadata['objects']}

    objects, present = [], set()
    for thor_id, obj in meta_by_id.items():
        if not obj.get('visible', False) or thor_id not in instance_masks:
            continue
        m = np.asarray(instance_masks[thor_id]).astype(bool)
        if m.ndim != 2 or int(m.sum()) < min_area:
            continue
        bbox = _bbox_from_mask(m)
        if bbox is None:
            continue
        oid = id_map[thor_id]
        present.add(thor_id)
        color = list(id2color.get(thor_id, [0, 0, 0]))
        objects.append({
            'id': oid,
            'category': thor_type_to_word(obj['objectType']),
            'thor_id': thor_id,
            'color': [int(c) for c in color[:3]],
            'bbox': bbox,
            'attributes': object_attributes(obj),
        })

    # Relations: receptacle links (on/in) where both endpoints are visible here.
    relations = []
    for thor_id in present:
        obj = meta_by_id[thor_id]
        for parent in obj.get('parentReceptacles') or []:
            if parent in present:
                ptype = meta_by_id[parent]['objectType']
                pred = 'in' if ptype in ENCLOSING_RECEPTACLES else 'on'
                relations.append({'subject': id_map[thor_id], 'predicate': pred,
                                  'object': id_map[parent]})

    # Relations: proximity ("near") between visible small objects.
    present_list = list(present)
    for i in range(len(present_list)):
        for j in range(i + 1, len(present_list)):
            a, b = meta_by_id[present_list[i]], meta_by_id[present_list[j]]
            pa, pb = a.get('position'), b.get('position')
            if not pa or not pb:
                continue
            d = ((pa['x'] - pb['x']) ** 2 + (pa['z'] - pb['z']) ** 2) ** 0.5
            if d <= near_dist:
                relations.append({'subject': id_map[present_list[i]],
                                  'predicate': 'near', 'object': id_map[present_list[j]]})

    return {'objects': objects, 'relations': relations}
