import os
import json
import pickle
import math
import torch
from fractions import Fraction
from functools import lru_cache
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

import cv2 as _cv2


def load_video(video_path, target_fps=None):
    if not os.path.exists(video_path):
        print("video path does not exist")
        return []
    cap = _cv2.VideoCapture(video_path)
    sample_rate = 1
    if target_fps is not None:
        original_fps = cap.get(_cv2.CAP_PROP_FPS)
        if target_fps > original_fps:
            raise ValueError("Target FPS cannot be higher than original FPS.")
        sample_rate = int(original_fps / target_fps)
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_rate == 0:
            frames.append(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


@lru_cache(maxsize=8)
def _cached_frames(video_path: str, target_fps: int, max_vid_len: int):
    """Load and subsample a video; cache the last 8 clips to avoid re-reading disk."""
    raw = load_video(video_path, target_fps if target_fps > 0 else None)
    if len(raw) > max_vid_len:
        rate = math.ceil(len(raw) / max_vid_len)
        raw = [f for i, f in enumerate(raw) if i % rate == 0]
    return raw  # list of HxWx3 uint8 numpy arrays (RGB)


def load_frame(inst: dict) -> torch.Tensor:
    """Lazy-load the frame for an instance as a float32 RGB tensor [H,W,3]."""
    frames = _cached_frames(inst['_video_path'], inst['_target_fps'], inst['_max_vid_len'])
    idx = min(inst['_frame_idx'], len(frames) - 1)
    return torch.from_numpy(frames[idx].astype(np.float32))


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


def _resize_masks_to_frames(
    masks: Dict[int, Dict[int, torch.Tensor]],
    frames: List[torch.Tensor],
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Resize every per-object mask to its frame's (H, W).

    Mask files ship at arbitrary resolutions (e.g. 640x360, often portrait) that
    rarely match the re-downloaded video. Bboxes are derived from masks, and the
    patch is placed using those bboxes, so masks MUST live in the same coordinate
    space as the decoded frames. Resizing here (once, early) makes masks, bboxes,
    patch placement and visualization all share the frame coordinate system.

    Uses nearest-neighbour to preserve the binary mask, matching the resize that
    the eval scene-graph path (functions/eval/scene_graph.py) applied downstream.
    """
    import torch.nn.functional as F

    resized: Dict[int, Dict[int, torch.Tensor]] = {}
    for frame_id, obj_masks in masks.items():
        if frame_id >= len(frames):
            continue
        H, W = int(frames[frame_id].shape[0]), int(frames[frame_id].shape[1])
        resized[frame_id] = {}
        for obj_id, m in obj_masks.items():
            if m.shape[0] != H or m.shape[1] != W:
                m = F.interpolate(
                    m.float().unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='nearest',
                ).squeeze(0).squeeze(0).bool()
            resized[frame_id][obj_id] = m
    return resized


def object_areas(masks: Dict[int, Dict[int, torch.Tensor]]) -> Dict[int, int]:
    """Max mask area (pixel count) per obj_id across all frames of a clip."""
    areas: Dict[int, int] = {}
    for obj_masks in masks.values():
        for obj_id, m in obj_masks.items():
            a = int(m.sum().item())
            if a > areas.get(obj_id, 0):
                areas[obj_id] = a
    return areas


def largest_object_id(masks: Dict[int, Dict[int, torch.Tensor]], candidates) -> Optional[int]:
    """
    Return the candidate obj_id with the largest mask area, or None.

    The mask set is heavily over-segmented (SAM2/GroundingDINO emit dozens of
    objects per clip, many of them noise specks). There is no ground-truth
    obj_id->category map, so the target is chosen as the largest-area object among
    those VINE classifies as the target category — i.e. the actual subject, not a
    fragment. Used by both training and eval so they pick the SAME object.
    """
    areas = object_areas(masks)
    best, best_area = None, 0
    for obj_id in candidates:
        a = areas.get(obj_id, 0)
        if a > best_area:
            best, best_area = obj_id, a
    return best


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
        max_clips: int = 0,
    ):
        self.dataset_dir = dataset_dir
        self.target_fps = target_fps
        self.max_vid_len = max_vid_len
        self.max_clips = max_clips

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

        # One pass: collect all mask-covered matching clips (for stable split),
        # and record which of them also have video on disk.
        all_stems: List[str]          = []   # mask-only — used to compute split
        seen_stems: Set[str]          = set()
        record_map: Dict[str, dict]   = {}   # stem → record (only when video present)

        for datapoint in raw:
            video_id = datapoint['video_id']
            gpt_spec = datapoint

            if target_category is not None:
                consts = gpt_spec.get('consts', [])
                if not any(target_category.lower() in c.lower() for c in consts):
                    continue

            if target_attribute is not None:
                unary_kws = gpt_spec.get('unary_kws', [])
                if not any(target_attribute.lower() in u.lower() for u in unary_kws):
                    continue

            file_stem = None
            for prefix in (f'ytb_{video_id}', f'v_{video_id}'):
                if prefix in mask_lookup:
                    file_stem = prefix
                    break

            if file_stem is None or file_stem in seen_stems:
                continue
            seen_stems.add(file_stem)
            all_stems.append(file_stem)

            if file_stem in video_lookup:
                record_map[file_stem] = {
                    'data_id':    video_id,
                    'video_path': video_lookup[file_stem],
                    'mask_path':  mask_lookup[file_stem],
                    'gpt_spec':   gpt_spec,
                }

        # Split computed on the full mask-covered set — stable regardless of
        # which videos happen to be downloaded at any given time.
        n_train = math.floor(self.TRAIN_RATIO * len(all_stems))
        if split == 'train':
            selected_stems = all_stems[:n_train]
        elif split == 'test':
            selected_stems = all_stems[n_train:]
        else:
            selected_stems = all_stems

        # Records = selected stems that also have a video on disk
        self.records = [record_map[s] for s in selected_stems if s in record_map]

        if split is not None:
            print(f"  Split '{split}': {len(self.records)} / {len(selected_stems)} clips "
                  f"on disk ({len(all_stems)} mask-covered total)")

        # Cap clips to bound CPU RAM (each clip's masks are ~tens of MB; tens of
        # thousands of per-instance masks otherwise exhaust RAM). 0 = no cap.
        if max_clips and len(self.records) > max_clips:
            self.records = self.records[:max_clips]
            print(f"  Capped to first {max_clips} clips (--max-clips).")

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

        try:
            raw_masks = pickle.load(open(rec['mask_path'], 'rb'))
        except Exception:
            return None

        # Frames and masks must be frame-for-frame aligned: the mask pkl is keyed by
        # the same frame index that load_video produces at this target_fps. A count
        # mismatch means we cannot trust the alignment, so skip the clip rather than
        # silently pairing a frame with the wrong object mask / bbox.
        # (mirrors LLaVADataset.__getitem__ in src/LASER/laser/training/llava_dataset.py)
        if len(raw_frames) != len(raw_masks):
            return None

        # Sub-sample frames AND masks together with the SAME rate, re-indexing the
        # mask frame-ids (new_fid = fid // rate) so that masks[k] / bboxes[k] always
        # correspond to frames[k]. Subsampling frames alone (the previous behaviour)
        # left masks on their original indices, so the bbox for frame k was read from
        # a different original frame and the patch landed off the moving target.
        if len(raw_frames) > self.max_vid_len:
            rate = math.ceil(len(raw_frames) / self.max_vid_len)
            raw_frames = [f for i, f in enumerate(raw_frames) if i % rate == 0]
            raw_masks = {
                int(fid // rate): obj_masks
                for fid, obj_masks in raw_masks.items()
                if fid % rate == 0
            }

        frames = [torch.from_numpy(f.astype(np.float32)) for f in raw_frames]
        masks  = _mask_pkl_to_tensors(raw_masks)
        # Put masks in the frame coordinate system BEFORE deriving bboxes, so the
        # bbox (and therefore patch placement + visualization) matches the decoded
        # frame rather than the mask file's native resolution.
        masks  = _resize_masks_to_frames(masks, frames)
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

            video_path = self.records[idx]['video_path']
            for frame_id in range(n_frames):
                if frame_id not in relevant:
                    continue
                frame_masks  = item['masks'].get(frame_id, {})
                frame_bboxes = item['bboxes'].get(frame_id, {})
                for obj_id, mask in frame_masks.items():
                    if obj_id not in frame_bboxes:
                        continue
                    instances.append({
                        'data_id':      item['data_id'],
                        'frame_id':     frame_id,
                        'obj_id':       obj_id,
                        '_video_path':  video_path,
                        '_frame_idx':   frame_id,
                        '_target_fps':  self.target_fps,
                        '_max_vid_len': self.max_vid_len,
                        'mask':         mask,
                        'bbox':         frame_bboxes[obj_id],
                        'gpt_spec':     item['gpt_spec'],
                    })
            # item['frames'] goes out of scope here — tensors freed immediately
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

            video_path = self.records[idx]['video_path']
            for frame_id in range(n_frames):
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
                            'data_id':      item['data_id'],
                            'frame_id':     frame_id,
                            'subj_obj_id':  subj_id,
                            'obj_obj_id':   obj_id,
                            '_video_path':  video_path,
                            '_frame_idx':   frame_id,
                            '_target_fps':  self.target_fps,
                            '_max_vid_len': self.max_vid_len,
                            'subj_mask':    frame_masks[subj_id],
                            'obj_mask':     frame_masks[obj_id],
                            'subj_bbox':    frame_bboxes[subj_id],
                            'obj_bbox':     frame_bboxes[obj_id],
                            'gpt_spec':     item['gpt_spec'],
                        })
            # item['frames'] goes out of scope here — tensors freed immediately
        return instances


def get_split_video_stems(
    dataset_dir: str,
    dataset_filename: str,
    target_category: Optional[str] = None,
    target_attribute: Optional[str] = None,
    target_relation: Optional[str] = None,
    split: Optional[str] = None,
    train_ratio: float = 0.8,
) -> List[str]:
    """
    Return video stems (e.g. 'ytb_XXXX') belonging to the given split,
    computed from ALL mask-covered clips that match the target filters.
    Stable regardless of which videos are currently downloaded.

    Used by download_videos to limit downloads to the relevant split.
    """
    masks_root = os.path.join(dataset_dir, "mini_LLaVA_10000_masks")
    mask_stems: Set[str] = set()
    for root, _, files in os.walk(masks_root):
        for fname in files:
            if fname.endswith('_mask.pkl'):
                mask_stems.add(fname[:-len('_mask.pkl')])

    ann_path = os.path.join(dataset_dir, dataset_filename)
    all_stems: List[str] = []
    seen:      Set[str]  = set()

    with open(ann_path) as f:
        for line in f:
            if not line.strip():
                continue
            dp = json.loads(line)
            video_id = dp['video_id']
            gpt_spec = dp

            if target_category is not None:
                consts = gpt_spec.get('consts', [])
                if not any(target_category.lower() in c.lower() for c in consts):
                    continue

            if target_attribute is not None:
                unary_kws = gpt_spec.get('unary_kws', [])
                if not any(target_attribute.lower() in u.lower() for u in unary_kws):
                    continue

            if target_relation is not None:
                binary_kws = gpt_spec.get('binary_kws', [])
                if not any(target_relation.lower() in b.lower() for b in binary_kws):
                    continue

            file_stem = None
            for prefix in (f'ytb_{video_id}', f'v_{video_id}'):
                if prefix in mask_stems:
                    file_stem = prefix
                    break

            if file_stem is None or file_stem in seen:
                continue
            seen.add(file_stem)
            all_stems.append(file_stem)

    n_train = math.floor(train_ratio * len(all_stems))
    if split == 'train':
        return all_stems[:n_train]
    elif split == 'test':
        return all_stems[n_train:]
    return all_stems


def _walk_paths(directory: str):
    if not os.path.isdir(directory):
        return
    for root, _, files in os.walk(directory):
        for f in files:
            yield os.path.join(root, f)
