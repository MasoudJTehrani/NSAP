import os
import json
import pickle
import math
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import numpy as np

from laser.loading import load_video


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

    Args:
        dataset_dir:       root directory of the ESCA dataset
        dataset_filename:  JSON annotation file (e.g. 'gpt_specs2_id_LLaVA_all.jsonl')
        target_category:   if given, only videos containing this string in their consts are loaded
        data_percentage:   fraction of dataset to load (0–100)
        target_fps:        FPS to sample from videos (default 1, matching VINE's eval)
        max_vid_len:       maximum frames per clip (default 10)
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_filename: str,
        target_category: Optional[str] = None,
        data_percentage: float = 100.0,
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
        max_count = math.floor(data_percentage / 100.0 * len(raw))

        records = []
        for datapoint in raw:
            if len(records) >= max_count:
                break

            video_id = datapoint['video_id']
            gpt_spec = datapoint

            # Category filter
            if target_category is not None:
                consts = gpt_spec.get('consts', [])
                if not any(target_category.lower() in c.lower() for c in consts):
                    continue

            # Video and mask files use ytb_ or v_ prefix
            file_stem = None
            for prefix in (f'ytb_{video_id}', f'v_{video_id}'):
                if prefix in video_lookup and prefix in mask_lookup:
                    file_stem = prefix
                    break

            if file_stem is None:
                continue

            records.append({
                'data_id':    video_id,
                'video_path': video_lookup[file_stem],
                'mask_path':  mask_lookup[file_stem],
                'gpt_spec':   gpt_spec,
            })

        self.records = records

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

    def build_instance_list(self, target_category: str) -> List[dict]:
        """
        Returns a flat list of per-object instances where the object's predicted category
        matches target_category (matched against gpt_spec['consts']).

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

            for frame_id, frame in enumerate(item['frames']):
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

            for frame_id, frame in enumerate(item['frames']):
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
