import os
import json
import pickle
import math
import torch
from fractions import Fraction
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from laser.loading import load_video


def _parse_frac_range(s: str) -> Tuple[float, float]:
    """Parse '[0, 1/4]' or '[1/4,3/8]' → (0.0, 0.25)."""
    s = s.strip().strip('[]')
    parts = s.split(',', 1)
    return float(Fraction(parts[0].strip())), float(Fraction(parts[1].strip()))


def _relevant_frame_ids(gpt_spec: dict, n_frames: int, target_predicate: str) -> Set[int]:
    """
    Return the set of frame indices (into the loaded frame list) where
    target_predicate is temporally active, using windowed_prog +
    'video location precise'.

    Each windowed_prog entry is [[start_slot, end_slot], program_string].
    Slots start_slot..end_slot-1 index into 'video location precise', which
    gives the fractional video position for each slot.

    Falls back to all frames when temporal data is absent or the predicate
    does not appear in any window.
    """
    windowed_prog = gpt_spec.get('windowed_prog', [])
    video_loc     = gpt_spec.get('video location precise', [])

    if not windowed_prog or not video_loc or n_frames == 0:
        return set(range(n_frames))

    pred_lower = target_predicate.lower()
    fracs: List[Tuple[float, float]] = []

    for window in windowed_prog:
        slot_range, prog_str = window[0], window[1]
        start_slot, end_slot = slot_range[0], slot_range[1]

        if pred_lower not in prog_str.lower():
            continue

        for slot_idx in range(start_slot, min(end_slot, len(video_loc))):
            try:
                fracs.append(_parse_frac_range(video_loc[slot_idx]))
            except Exception:
                pass

    if not fracs:
        return set(range(n_frames))

    frame_ids: Set[int] = set()
    for start_f, end_f in fracs:
        lo = max(0, int(start_f * n_frames))
        hi = min(n_frames, math.ceil(end_f * n_frames))
        if hi > lo:
            frame_ids.update(range(lo, hi))

    return frame_ids if frame_ids else set(range(n_frames))


def _mask_pkl_to_tensors(raw_masks: dict) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Convert raw pickle masks {frame_id: {obj_id: array(1,H,W)}} to
    {frame_id: {obj_id: Tensor[H,W] bool}}.
    """
    result = {}
    for frame_id, obj_masks in raw_masks.items():
        result[frame_id] = {}
        for obj_id, mask in obj_masks.items():
            m = torch.from_numpy(np.array(mask, dtype=bool))
            if m.dim() == 3:
                m = m.squeeze(0)    # (1,H,W) -> (H,W)
            elif m.dim() == 2:
                pass
            result[frame_id][obj_id] = m
    return result


def _bboxes_from_masks(masks: Dict[int, Dict[int, torch.Tensor]]) -> Dict[int, Dict[int, dict]]:
    """
    Compute {x1,y1,x2,y2} bounding boxes from binary mask tensors.
    Skips empty masks.
    """
    bboxes: Dict[int, Dict[int, dict]] = {}
    for frame_id, obj_masks in masks.items():
        bboxes[frame_id] = {}
        for obj_id, mask in obj_masks.items():
            nonzero = mask.nonzero(as_tuple=False)   # [N, 2] (row=y, col=x)
            if nonzero.numel() == 0:
                continue
            y1 = nonzero[:, 0].min().item()
            y2 = nonzero[:, 0].max().item()
            x1 = nonzero[:, 1].min().item()
            x2 = nonzero[:, 1].max().item()
            if x2 >= x1 and y2 >= y1:
                bboxes[frame_id][obj_id] = {
                    'x1': x1, 'y1': y1,
                    'x2': max(x2, x1 + 1), 'y2': max(y2, y1 + 1),
                }
    return bboxes


class EscaAttackDataset(Dataset):
    """
    Loads ESCA video clips and their per-object masks for adversarial patch training.

    Each item contains all frames and objects for one video clip.
    Use build_instance_list() to get a flat list filtered by target category.

    The dataset is split deterministically into 80% train / 20% test by position
    in the filtered record list. Pass split='train' or split='test' to get the
    appropriate partition. Pass split=None (default) to get all records.

    Args:
        dataset_dir:       root directory of the ESCA dataset
        dataset_filename:  JSON annotation file (e.g. 'gpt_specs2_id_LLaVA_all.jsonl')
        target_category:   if given, only videos containing this string are loaded
        split:             'train' (first 80%), 'test' (last 20%), or None (all)
        target_fps:        FPS to sample from videos (default 1, matching VINE's eval)
        max_vid_len:       maximum frames per clip (default 10)
        attack_mode:       'category', 'attribute', or 'relation'
    """

    TRAIN_RATIO = 0.8

    def __init__(
        self,
        dataset_dir: str,
        dataset_filename: str,
        target_category: Optional[str] = None,
        target_attribute: Optional[str] = None,
        split: Optional[str] = None,
        target_fps: int = 1,
        max_vid_len: int = 10,
    ):
        self.dataset_dir = dataset_dir
        self.target_fps = target_fps
        self.max_vid_len = max_vid_len

        videos_dir = os.path.join(dataset_dir, "videos")
        masks_root = os.path.join(dataset_dir, "mini_LLaVA_10000_masks")

        # Build fast lookup: stem (no .mp4) → full path
        video_lookup: Dict[str, str] = {}
        if os.path.isdir(videos_dir):
            for fname in os.listdir(videos_dir):
                stem = fname[:-4] if fname.endswith('.mp4') else fname
                video_lookup[stem] = os.path.join(videos_dir, fname)

        # Build fast lookup: stem (no _mask.pkl) → full path
        mask_lookup: Dict[str, str] = {}
        for root, _, files in os.walk(masks_root):
            for fname in files:
                if fname.endswith('_mask.pkl'):
                    stem = fname[:-len('_mask.pkl')]
                    mask_lookup[stem] = os.path.join(root, fname)

        # Load main annotation file (JSONL: one JSON object per line)
        ann_path = os.path.join(dataset_dir, dataset_filename)
        with open(ann_path) as f:
            raw = [json.loads(line) for line in f if line.strip()]

        # Collect all matching records (no percentage cap — split handles partitioning)
        all_records = []
        for datapoint in raw:
            video_id = datapoint['video_id']
            gpt_spec = datapoint

            # target_category always filters against consts (object names)
            if target_category is not None:
                consts = gpt_spec.get('consts', [])
                if not any(target_category.lower() in c.lower() for c in consts):
                    continue

            # target_attribute additionally filters against unary_kws (attribute mode)
            if target_attribute is not None:
                unary_kws = gpt_spec.get('unary_kws', [])
                if not any(target_attribute.lower() in u.lower() for u in unary_kws):
                    continue

            # Video and mask files use ytb_ or v_ prefix
            file_stem = None
            for prefix in (f'ytb_{video_id}', f'v_{video_id}'):
                if prefix in video_lookup and prefix in mask_lookup:
                    file_stem = prefix
                    break

            if file_stem is None:
                continue

            all_records.append({
                'data_id':    video_id,
                'video_path': video_lookup[file_stem],
                'mask_path':  mask_lookup[file_stem],
                'gpt_spec':   gpt_spec,
            })

        # Deterministic 80/20 train/test split by position in the filtered list
        n_train = math.floor(self.TRAIN_RATIO * len(all_records))
        if split == 'train':
            self.records = all_records[:n_train]
        elif split == 'test':
            self.records = all_records[n_train:]
        else:
            self.records = all_records

        if split is not None:
            print(f"  Split '{split}': {len(self.records)} / {len(all_records)} clips "
                  f"({'80' if split == 'train' else '20'}% hold-out)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Optional[dict]:
        """
        Returns a dict with keys:
            data_id    : str
            frames     : List[Tensor[H,W,3]] float32 RGB
            masks      : Dict[frame_id, Dict[obj_id, Tensor[H,W] bool]]
            bboxes     : Dict[frame_id, Dict[obj_id, {x1,y1,x2,y2}]]
            gpt_spec   : dict
        Returns None if the video cannot be loaded.
        """
        rec = self.records[idx]

        try:
            raw_frames = load_video(rec['video_path'], target_fps=self.target_fps)
        except Exception:
            return None

        if len(raw_frames) == 0:
            return None

        # Sub-sample to max_vid_len (same logic as LLaVADataset)
        if len(raw_frames) > self.max_vid_len:
            rate = math.ceil(len(raw_frames) / self.max_vid_len)
            raw_frames = [f for i, f in enumerate(raw_frames) if i % rate == 0]

        frames = [torch.from_numpy(f.astype(np.float32)) for f in raw_frames]

        try:
            raw_masks = pickle.load(open(rec['mask_path'], 'rb'))
        except Exception:
            return None

        masks  = _mask_pkl_to_tensors(raw_masks)
        bboxes = _bboxes_from_masks(masks)

        return {
            'data_id':  rec['data_id'],
            'frames':   frames,
            'masks':    masks,
            'bboxes':   bboxes,
            'gpt_spec': rec['gpt_spec'],
        }

    def build_instance_list(
        self,
        target_category: str,
        target_predicate: Optional[str] = None,
    ) -> List[dict]:
        """
        Returns a flat list of per-object instances from clips already filtered
        by __init__. target_category is confirmed against consts as a secondary check.

        If target_predicate is given (used for attribute attacks), only frames
        where that predicate is temporally active are included, using windowed_prog
        and 'video location precise' from the clip's gpt_spec.

        Each entry in the list:
            data_id    : str
            frame_id   : int
            obj_id     : int
            frame      : Tensor[H,W,3] float32 RGB
            mask       : Tensor[H,W] bool
            bbox       : {x1,y1,x2,y2}
            gpt_spec   : dict
        """
        instances = []
        for idx in range(len(self)):
            item = self[idx]
            if item is None:
                continue

            consts = item['gpt_spec'].get('consts', [])
            if not any(target_category.lower() in c.lower() for c in consts):
                continue

            n_frames = len(item['frames'])
            relevant = (
                _relevant_frame_ids(item['gpt_spec'], n_frames, target_predicate)
                if target_predicate else set(range(n_frames))
            )

            for frame_id, frame in enumerate(item['frames']):
                if frame_id not in relevant:
                    continue
                frame_masks  = item['masks'].get(frame_id, {})
                frame_bboxes = item['bboxes'].get(frame_id, {})
                for obj_id, mask in frame_masks.items():
                    if obj_id not in frame_bboxes:
                        continue
                    instances.append({
                        'data_id':  item['data_id'],
                        'frame_id': frame_id,
                        'obj_id':   obj_id,
                        'frame':    frame,
                        'mask':     mask,
                        'bbox':     frame_bboxes[obj_id],
                        'gpt_spec': item['gpt_spec'],
                    })
        return instances

    def build_relation_instance_list(
        self,
        target_relation: str,
        subject_category: Optional[str] = None,
    ) -> List[dict]:
        """
        Returns a flat list of per-pair instances for relation attack.

        Only frames where target_relation is temporally active are included,
        using windowed_prog and 'video location precise'.

        Each entry:
            data_id          : str
            frame_id         : int
            subj_obj_id      : int
            obj_obj_id       : int
            subj_frame       : Tensor[H,W,3] — full frame (for pasting on subject)
            subj_mask        : Tensor[H,W] bool
            obj_mask         : Tensor[H,W] bool
            subj_bbox        : {x1,y1,x2,y2}
            obj_bbox         : {x1,y1,x2,y2}
            gpt_spec         : dict
        """
        instances = []
        for idx in range(len(self)):
            item = self[idx]
            if item is None:
                continue

            bin_preds = item['gpt_spec'].get('binary_predicates', [])
            if not any(target_relation.lower() in p[0].lower() for p in bin_preds):
                continue

            n_frames = len(item['frames'])
            relevant = _relevant_frame_ids(item['gpt_spec'], n_frames, target_relation)

            for frame_id, frame in enumerate(item['frames']):
                if frame_id not in relevant:
                    continue
                frame_masks  = item['masks'].get(frame_id, {})
                frame_bboxes = item['bboxes'].get(frame_id, {})

                obj_ids = list(frame_masks.keys())
                for i, subj_id in enumerate(obj_ids):
                    for obj_id in obj_ids:
                        if subj_id == obj_id:
                            continue
                        if subj_id not in frame_bboxes or obj_id not in frame_bboxes:
                            continue
                        instances.append({
                            'data_id':     item['data_id'],
                            'frame_id':    frame_id,
                            'subj_obj_id': subj_id,
                            'obj_obj_id':  obj_id,
                            'frame':       frame,
                            'subj_mask':   frame_masks[subj_id],
                            'obj_mask':    frame_masks[obj_id],
                            'subj_bbox':   frame_bboxes[subj_id],
                            'obj_bbox':    frame_bboxes[obj_id],
                            'gpt_spec':    item['gpt_spec'],
                        })
        return instances


def _walk_paths(directory: str):
    if not os.path.isdir(directory):
        return
    for root, _, files in os.walk(directory):
        for f in files:
            yield os.path.join(root, f)
