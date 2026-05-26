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


def _attack_subdir(base_dir: str, mode: str, target: str, relation: str = "") -> str:
    """Return the per-attack output directory, e.g. output/patches/category_person/"""
    if mode == "relation" and relation:
        slug = f"relation_{relation.replace(' ', '_')}_{target.replace(' ', '_')}"
    else:
        slug = f"{mode}_{target.replace(' ', '_')}"
    return os.path.join(base_dir, slug)


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

    attack_dir = _attack_subdir(args.output_dir, args.mode, args.target, args.relation)

    # Clear previous run for the same attack type + target
    if os.path.isdir(attack_dir):
        print(f"Clearing previous output in {attack_dir} ...")
        shutil.rmtree(attack_dir)
    os.makedirs(attack_dir, exist_ok=True)

    config = AttackConfig(
        attack_mode     = args.mode,
        target_category = args.target,
        target_relation = args.relation,
        coverage_ratio  = args.coverage,
        num_steps       = args.steps,
        eot_batch_size  = args.batch,
        learning_rate   = args.lr,
        output_dir      = attack_dir,
        ref_gram_cache  = os.path.join(args.output_dir, "gram_cache"),
        clip_image_size = args.clip_size,
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
    return patch


def run_eval(args):
    from functions.attack.train import AttackConfig
    from functions.attack.patch import AdversarialPatch
    from functions.attack.anchor import Anchor
    from functions.attack.data import EscaAttackDataset
    from functions.eval.flip_rate import compute_flip_rate_across_clips
    from functions.eval.edit_distance import evaluate_isolation

    device     = args.device
    vine_model = load_vine(device)

    ckpt = torch.load(args.patch, map_location=device, weights_only=False)
    config: AttackConfig = ckpt['config']
    anchor: Anchor       = ckpt['anchor']

    ph, pw = ckpt['w'].shape[1], ckpt['w'].shape[2]
    patch  = AdversarialPatch(ph, pw, torch.device(device))
    patch.w.data.copy_(ckpt['w'])
    patch.eval()

    dataset = EscaAttackDataset(
        dataset_dir      = os.path.join(os.path.dirname(__file__), "dataset"),
        dataset_filename = "gpt_specs2_id_LLaVA_all.jsonl",
        target_category  = config.target_category,
        data_percentage  = 10.0,
    )

    # Hold-out: skip first 80% (those were training), evaluate on last 20%
    n = len(dataset)
    test_indices = list(range(int(0.8 * n), n))
    test_items   = [dataset[i] for i in test_indices]

    print(f"\n=== Flip Rate Evaluation ({len(test_items)} clips) ===")
    flip_results = compute_flip_rate_across_clips(
        vine_model, test_items, patch, anchor, config, torch.device(device)
    )
    print(f"Flip rate: {flip_results['flip_rate']:.3f}  "
          f"({flip_results['n_flipped']}/{flip_results['n_total']} pairs flipped)")

    print(f"\n=== Scene Graph Edit Distance & Isolation ===")
    iso_results = evaluate_isolation(
        vine_model, test_items, patch, anchor, config, torch.device(device)
    )
    print(f"Mean edit distance : {iso_results['mean_edit_distance']:.2f}")
    print(f"Mean isolation     : {iso_results['mean_isolation_score']:.2f}")
    print(f"  category_sub     : {iso_results['mean_category_sub']:.2f}")
    print(f"  attribute_sub    : {iso_results['mean_attribute_sub']:.2f}")
    print(f"  relation_sub     : {iso_results['mean_relation_sub']:.2f}")

    # Save results as JSON in the same directory as the patch file
    results = {
        "patch":        args.patch,
        "attack_mode":  config.attack_mode,
        "target":       config.target_category,
        "n_test_clips": len(test_items),
        "flip_rate":    flip_results['flip_rate'],
        "n_flipped":    flip_results['n_flipped'],
        "n_total":      flip_results['n_total'],
        "mean_edit_distance":   iso_results['mean_edit_distance'],
        "mean_isolation_score": iso_results['mean_isolation_score'],
        "mean_category_sub":    iso_results['mean_category_sub'],
        "mean_attribute_sub":   iso_results['mean_attribute_sub'],
        "mean_relation_sub":    iso_results['mean_relation_sub'],
        "mean_node_ins":        iso_results.get('mean_node_ins', 0.0),
        "mean_node_del":        iso_results.get('mean_node_del', 0.0),
    }

    patch_dir  = os.path.dirname(os.path.abspath(args.patch))
    patch_stem = os.path.splitext(os.path.basename(args.patch))[0]
    results_path = os.path.join(patch_dir, f"{patch_stem}_eval.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="NSAP adversarial patch attack on VINE")

    # Mode
    parser.add_argument("--mode",   default="category",
                        choices=["category", "attribute", "relation"],
                        help="Attack mode")
    parser.add_argument("--target", default="person",
                        help="Target category string (e.g. 'person', 'girl', 'cup')")
    parser.add_argument("--relation", default="next to",
                        help="Target relation string for --mode relation")

    # Training
    parser.add_argument("--steps",    type=int,   default=2000,  help="Training steps")
    parser.add_argument("--batch",    type=int,   default=8,     help="EoT batch size")
    parser.add_argument("--lr",       type=float, default=0.01,  help="Adam learning rate")
    parser.add_argument("--coverage", type=float, default=0.3,   help="Patch coverage ratio")
    parser.add_argument("--clip-size",type=int,   default=224,
                        dest="clip_size", help="CLIP input size (224 or 336)")

    # Eval
    parser.add_argument("--eval",  action="store_true", help="Run evaluation instead of training")
    parser.add_argument("--patch", default=None, help="Path to saved patch .pt file for --eval")

    # Misc
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="output/patches", dest="output_dir")

    args = parser.parse_args()

    if args.eval:
        if not args.patch:
            parser.error("--eval requires --patch <path>")
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
