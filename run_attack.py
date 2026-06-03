"""
NSAP — Neuro-Symbolic Adversarial Patch
Entry point for training and evaluating adversarial patches against VINE.

Usage examples
--------------
# Category attack on 'person', quick test (50 steps):
python run_attack.py --mode category --target person --steps 50

# Attribute attack on 'girl' (unary keywords), full run:
python run_attack.py --mode attribute --target girl

# Relation attack ('next to'):
python run_attack.py --mode relation --relation "next to" --target girl

# Evaluate a saved patch:
python run_attack.py --eval --patch output/patches/category_person/category_person_final.pt
"""

import os
import sys
import json
import shutil
import argparse

# Make laser and project modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "LASER"))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def _attack_subdir(base_dir: str, mode: str, target: str, relation: str = "", attribute: str = "") -> str:
    """Return the per-attack output directory, e.g. output/patches/category_person/"""
    if mode == "relation" and relation:
        slug = f"relation_{relation.replace(' ', '_')}_{target.replace(' ', '_')}"
    elif mode == "attribute" and attribute:
        slug = f"attribute_{target.replace(' ', '_')}_{attribute.replace(' ', '_')}"
    else:
        slug = f"{mode}_{target.replace(' ', '_')}"
    return os.path.join(base_dir, slug)


def _build_download_targets(mode: str, target: str, attribute: str, relation: str) -> list:
    """Convert attack args to the target-dict format used by download_videos."""
    t = {}
    if target:    t['category']  = target
    if mode == 'attribute' and attribute: t['attribute'] = attribute
    if mode == 'relation'  and relation:  t['relation']  = relation
    return [t] if t else []


def _ensure_videos(mode: str, target: str, attribute: str, relation: str,
                   split: str = None) -> None:
    from functions.download_videos import ensure_videos_for_targets
    from functions.attack.data import get_split_video_stems

    targets = _build_download_targets(mode, target, attribute, relation)
    if not targets:
        return

    split_stems = None
    if split is not None:
        stems = get_split_video_stems(
            dataset_dir      = os.path.join(os.path.dirname(__file__), "dataset"),
            dataset_filename = "gpt_specs2_id_LLaVA_all.jsonl",
            target_category  = target   or None,
            target_attribute = attribute if mode == 'attribute' else None,
            target_relation  = relation  if mode == 'relation'  else None,
            split            = split,
        )
        split_stems = set(stems)
        print(f"\n=== Ensuring videos for attack (split='{split}', {len(split_stems)} eligible clips) ===")
    else:
        print(f"\n=== Ensuring videos for attack ===")

    n = ensure_videos_for_targets(targets, split_stems=split_stems)
    if n == 0:
        print("WARNING: no matching clips found on disk — attack may have no data.")


def _cleanup_videos() -> None:
    from functions.download_videos import cleanup_all_videos
    print(f"\n=== Cleaning up downloaded videos ===")
    cleanup_all_videos()


def load_vine(device: str) -> "VineModel":
    from model.vine.vine_config import VineConfig
    from model.vine.vine_model import VineModel

    config = VineConfig(
        use_hf_repo=False,
        local_dir=os.path.join(os.path.dirname(__file__), "model", "vine"),
        local_filename="model.safetensors",
        device=device,
    )
    print(f"Loading VineModel on {device}...")
    model = VineModel(config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("VineModel loaded.")
    return model


def run_train(args):
    from functions.attack.train import train_patch, AttackConfig

    _ensure_videos(args.mode, args.target, args.attribute, args.relation, split='train')

    attack_dir = _attack_subdir(args.output_dir, args.mode, args.target, args.relation, args.attribute)

    # Clear previous run for the same attack type + target
    if os.path.isdir(attack_dir):
        print(f"Clearing previous output in {attack_dir} ...")
        shutil.rmtree(attack_dir)
    os.makedirs(attack_dir, exist_ok=True)

    config = AttackConfig(
        attack_mode      = args.mode,
        target_category  = args.target,
        target_relation  = args.relation,
        target_attribute = args.attribute,
        coverage_ratio   = args.coverage,
        num_steps        = args.steps,
        eot_batch_size   = args.batch,
        learning_rate    = args.lr,
        output_dir       = attack_dir,
        ref_gram_cache   = os.path.join(args.output_dir, "gram_cache"),
        clip_image_size  = args.clip_size,
        use_chameleon    = args.use_chameleon,
        max_clips        = args.max_clips,
    )

    device     = args.device
    vine_model = load_vine(device)

    # Load SAM2 predictor from the local prerequisite checkpoint
    from functions.download_model import SAM2_CHECKPOINT, SAM2_CONFIG
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam_model     = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    sam_predictor = SAM2ImagePredictor(sam_model)
    print(f"SAM2 loaded from {SAM2_CHECKPOINT.name}.")

    patch = train_patch(
        vine_model       = vine_model,
        dataset_dir      = os.path.join(os.path.dirname(__file__), "dataset"),
        dataset_filename = "gpt_specs2_id_LLaVA_all.jsonl",
        config           = config,
        sam_predictor    = sam_predictor,
    )
    print("\nTraining complete.")

    if not args.keep_videos:
        _cleanup_videos()

    return patch


def run_validate_bboxes(args):
    """Validate bbox derivation + frame/mask alignment, writing overlay images."""
    from functions.attack.data import EscaAttackDataset
    from functions.eval.validate_bboxes import validate_bboxes

    _ensure_videos(args.mode, args.target, args.attribute, args.relation, split='test')

    dataset = EscaAttackDataset(
        dataset_dir      = os.path.join(os.path.dirname(__file__), "dataset"),
        dataset_filename = "gpt_specs2_id_LLaVA_all.jsonl",
        target_category  = args.target or None,
        target_attribute = args.attribute if args.mode == 'attribute' else None,
        split            = 'test',
    )

    attack_dir = _attack_subdir(args.output_dir, args.mode, args.target, args.relation, args.attribute)
    os.makedirs(attack_dir, exist_ok=True)
    validate_bboxes(dataset, n_clips=args.n_clips, out_dir=attack_dir)

    if not args.keep_videos:
        _cleanup_videos()


def run_eval(args):
    from functions.attack.train import AttackConfig
    from functions.attack.patch import AdversarialPatch
    from functions.attack.anchor import Anchor
    from functions.attack.data import EscaAttackDataset
    from functions.eval.flip_rate import compute_flip_rate_across_clips
    from functions.eval.edit_distance import evaluate_isolation
    from functions.eval.visualize import save_sample_images

    device     = args.device
    vine_model = load_vine(device)

    ckpt = torch.load(args.patch, map_location=device, weights_only=False)
    config: AttackConfig = ckpt['config']
    anchor: Anchor       = ckpt['anchor']

    # Eval materializes every test clip's frames + masks at once, so allow the
    # CLI --max-clips to override the (larger) cap baked into the checkpoint.
    if getattr(args, 'max_clips', 0):
        config.max_clips = args.max_clips

    _ensure_videos(
        config.attack_mode,
        config.target_category,
        getattr(config, 'target_attribute', ''),
        getattr(config, 'target_relation', ''),
        split='test',
    )

    ph, pw = ckpt['w'].shape[1], ckpt['w'].shape[2]
    patch  = AdversarialPatch(ph, pw, torch.device(device))
    patch.w.data.copy_(ckpt['w'])
    patch.eval()

    dataset = EscaAttackDataset(
        dataset_dir      = os.path.join(os.path.dirname(__file__), "dataset"),
        dataset_filename = "gpt_specs2_id_LLaVA_all.jsonl",
        target_category  = config.target_category,
        target_attribute = config.target_attribute if config.attack_mode == 'attribute' else None,
        split            = 'test',
        max_clips        = getattr(config, 'max_clips', 0),
    )

    test_items = [dataset[i] for i in range(len(dataset))]

    print(f"\n=== Flip Rate (target objects only, {len(test_items)} clips) ===")
    flip_results = compute_flip_rate_across_clips(
        vine_model, test_items, patch, anchor, config, torch.device(device)
    )
    clips_with = flip_results.get('n_clips_with_targets', '?')
    print(f"Clips with target objects : {clips_with} / {len(test_items)}")
    print(f"Flip rate : {flip_results['flip_rate']:.1%}  "
          f"({flip_results['n_flipped']}/{flip_results['n_total']} target objects flipped)")

    print(f"\n=== Scene Graph Edit Distance (target objects only) ===")
    iso_results = evaluate_isolation(
        vine_model, test_items, patch, anchor, config, torch.device(device)
    )
    mode   = config.attack_mode
    target = config.target_category or config.target_relation

    attacked_label   = {'category': 'category', 'attribute': 'attribute', 'relation': 'relation'}[mode]
    collateral_heads = [h for h in ('category', 'attribute', 'relation') if h != attacked_label]

    print(f"Mean target objects per clip : {iso_results['mean_n_target_objects']:.1f}")
    print(f"[attacked]  {attacked_label}_change_rate  : "
          f"{iso_results[f'mean_{attacked_label}_change_rate']:.1%}  ← higher = stronger attack")
    for h in collateral_heads:
        print(f"[collateral] {h}_change_rate : "
              f"{iso_results[f'mean_{h}_change_rate']:.1%}  ← lower = better isolation")
    print(f"Disappear rate               : {iso_results['mean_disappear_rate']:.1%}")
    print(f"Isolation score              : {iso_results['mean_isolation_score']:.1%}  "
          f"(fraction of changes on attacked head — higher = better)")

    # Save results as JSON in the same directory as the patch file
    results = {
        "patch":        args.patch,
        "attack_mode":  config.attack_mode,
        "target":       target,
        "n_test_clips": len(test_items),
        # Flip rate
        "flip_rate":    flip_results['flip_rate'],
        "n_flipped":    flip_results['n_flipped'],
        "n_total":      flip_results['n_total'],
        # Edit distance rates (all normalised by n_target_objects per clip)
        "mean_n_target_objects":      iso_results['mean_n_target_objects'],
        "mean_category_change_rate":  iso_results['mean_category_change_rate'],
        "mean_attribute_change_rate": iso_results['mean_attribute_change_rate'],
        "mean_relation_change_rate":  iso_results['mean_relation_change_rate'],
        "mean_disappear_rate":        iso_results['mean_disappear_rate'],
        "mean_isolation_score":       iso_results['mean_isolation_score'],
    }

    patch_dir  = os.path.dirname(os.path.abspath(args.patch))
    patch_stem = os.path.splitext(os.path.basename(args.patch))[0]
    results_path = os.path.join(patch_dir, f"{patch_stem}_eval.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print(f"\n=== Sample Images ===")
    save_sample_images(
        vine_model, test_items, patch, anchor, config,
        torch.device(device), patch_dir, n_samples=10,
    )

    if not args.keep_videos:
        _cleanup_videos()


def main():
    parser = argparse.ArgumentParser(description="NSAP adversarial patch attack on VINE")

    # Mode
    parser.add_argument("--mode",   default="category",
                        choices=["category", "attribute", "relation"],
                        help="Attack mode")
    parser.add_argument("--target", default="person",
                        help="Target category string (e.g. 'person', 'girl', 'cup')")
    parser.add_argument("--relation",  default="next to",
                        help="Target relation string for --mode relation")
    parser.add_argument("--attribute", default="",
                        help="Target attribute string for --mode attribute (e.g. 'standing')")

    # Training
    parser.add_argument("--steps",    type=int,   default=2000,  help="Training steps")
    parser.add_argument("--batch",    type=int,   default=8,     help="EoT batch size")
    parser.add_argument("--lr",       type=float, default=0.01,  help="Adam learning rate")
    parser.add_argument("--coverage", type=float, default=0.3,   help="Patch coverage ratio")
    parser.add_argument("--clip-size",type=int,   default=224,
                        dest="clip_size", help="CLIP input size (224 or 336)")
    parser.add_argument("--no-chameleon", action="store_false", dest="use_chameleon",
                        help="Disable Chameleon naturalness losses (gram/tv/dino). "
                             "Frees VGG16 + DINOv2 from GPU — use on small-memory cards.")
    parser.add_argument("--max-clips", type=int, default=0, dest="max_clips",
                        help="Cap clips loaded into RAM (0 = no cap). Each clip's masks "
                             "cost ~tens of MB; cap this on machines with limited RAM.")

    # Eval
    parser.add_argument("--eval",  action="store_true", help="Run evaluation instead of training")
    parser.add_argument("--patch", default=None, help="Path to saved patch .pt file for --eval")

    # Validation
    parser.add_argument("--validate-bboxes", action="store_true", dest="validate_bboxes",
                        help="Validate bbox derivation + frame/mask alignment and write overlay images")
    parser.add_argument("--n-clips", type=int, default=10, dest="n_clips",
                        help="Number of clips to check for --validate-bboxes")

    # Videos
    parser.add_argument(
        "--keep-videos", action="store_true", dest="keep_videos",
        help="Keep downloaded videos after training/eval (default: delete them)",
    )

    # Misc
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="output/patches", dest="output_dir")

    args = parser.parse_args()

    if args.validate_bboxes:
        run_validate_bboxes(args)
    elif args.eval:
        if not args.patch:
            parser.error("--eval requires --patch <path>")
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
