"""
NSAP — Neuro-Symbolic Adversarial Patch
Entry point for training and evaluating adversarial patches against VINE.

Data source: the self-contained SceneGraphBench (AI2-THOR) sim dataset. Get it with
`python NSAP.py --with-dataset` (downloads to dataset/sim/, used automatically) or
generate one from the SceneGraphBench/ submodule, then pass it with --sim-data <dir>.

Usage examples
--------------
# Category attack on 'Toaster', quick test (50 steps):
python run_attack.py --sim-data <dataset> --mode category --target Toaster --steps 50

# Attribute attack: a television that is 'displaying' (screen on):
python run_attack.py --sim-data <dataset> --mode attribute --target Television --attribute displaying

# Relation attack: a pot that is 'on' a surface (staged scenes only):
python run_attack.py --sim-data dataset/sim --mode relation --relation "on" --target Pot --data-source pot

# Evaluate a saved patch:
python run_attack.py --sim-data <dataset> --eval --patch output/patches/category_Toaster/category_Toaster_final.pt
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
    """Return the per-attack output directory, e.g. output/patches/category_Fridge/"""
    if mode == "relation" and relation:
        slug = f"relation_{relation.replace(' ', '_')}_{target.replace(' ', '_')}"
    elif mode == "attribute" and attribute:
        slug = f"attribute_{target.replace(' ', '_')}_{attribute.replace(' ', '_')}"
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

    if not args.sim_data:
        raise SystemExit("--sim-data <dataset dir> is required. Get the dataset with "
                         "`python NSAP.py --with-dataset` (-> dataset/sim/) or generate "
                         "one from the SceneGraphBench/ submodule.")

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
        eot_crops        = args.eot_crops,
        num_steps        = args.steps,
        eot_batch_size   = args.batch,
        learning_rate    = args.lr,
        output_dir       = attack_dir,
        ref_gram_cache   = os.path.join(args.output_dir, "gram_cache"),
        clip_image_size  = args.clip_size,
        use_chameleon    = args.use_chameleon,
        max_clips        = args.max_clips,
        min_target_conf  = args.min_target_conf,
        sim_data_dir     = args.sim_data,
        data_source      = args.data_source,
        use_all_data     = args.all_data,
        init_patch_path  = args.init_patch,
        only_scenario    = args.only_scenario,
        lambda_c         = args.lambda_c,
        lambda_tv        = args.lambda_tv,
        lambda_dino      = args.lambda_dino,
        object_init      = args.object_init,
        lambda_content   = args.lambda_content,
        seed             = args.seed,
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
        config           = config,
        sam_predictor    = sam_predictor,
    )
    print("\nTraining complete.")

    return patch


def run_eval(args):
    from functions.attack.train import AttackConfig
    from functions.attack.patch import AdversarialPatch
    from functions.attack.anchor import Anchor
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
    # Allow overriding the target-confidence gate at eval time too.
    if getattr(args, 'min_target_conf', 0.0):
        config.min_target_conf = args.min_target_conf
    # Allow pointing eval at a sim dataset (overrides whatever the ckpt stored).
    if getattr(args, 'sim_data', ''):
        config.sim_data_dir = args.sim_data

    if not getattr(config, 'sim_data_dir', ''):
        raise SystemExit("--sim-data <dataset dir> is required. Get it with "
                         "`python NSAP.py --with-dataset` or generate one from SceneGraphBench/.")

    ph, pw = ckpt['w'].shape[1], ckpt['w'].shape[2]
    patch  = AdversarialPatch(ph, pw, torch.device(device))
    patch.w.data.copy_(ckpt['w'])
    patch.eval()

    from functions.attack.sim_data import SimAttackDataset
    _only = [args.only_scenario] if getattr(args, 'only_scenario', '') else None
    dataset = SimAttackDataset(
        config.sim_data_dir,
        target_category  = config.target_category if config.attack_mode != 'relation' else None,
        target_attribute = config.target_attribute if config.attack_mode == 'attribute' else None,
        target_relation  = config.target_relation if config.attack_mode == 'relation' else None,
        split            = None if _only else 'test',
        max_clips        = getattr(config, 'max_clips', 0),
        only_scenarios   = _only,
        data_source      = getattr(config, 'data_source', ''),
    )

    test_items = [dataset[i] for i in range(len(dataset))]

    # Score EACH frame as its own sample (more statistical power) rather than
    # one selected frame per scenario.
    from functions.attack.sim_data import explode_to_frames
    test_items = [fr for it in test_items for fr in explode_to_frames(it)]
    # Relation eval re-scores EVERY object pair per frame (heavy); cap the
    # frame-samples so it finishes. Category/attribute heads are cheap → keep all.
    if config.attack_mode == 'relation' and len(test_items) > 150:
        step = max(1, len(test_items) // 150)
        test_items = test_items[::step]
    print(f"  [sim] per-frame eval: {len(test_items)} frame-samples")

    # Post-explosion each sample is a single frame, so n_loaded is a frame-sample count.
    n_loaded = sum(1 for it in test_items if it is not None)

    print(f"\n=== Test set ===")
    print(f"Frame-samples                : {len(test_items)}")
    _mtc = getattr(config, 'min_target_conf', 0.0)
    if _mtc > 0:
        print(f"Target confidence gate       : category prob ≥ {_mtc:.2f}")

    print(f"\n=== Flip Rate (attacked head, target objects only) ===")
    flip_results = compute_flip_rate_across_clips(
        vine_model, test_items, patch, anchor, config, torch.device(device)
    )
    clips_with = flip_results.get('n_clips_with_targets', '?')
    print(f"Clips with a target object   : {clips_with} / {n_loaded}")
    print(f"Target objects (1 / clip)    : {flip_results['n_total']}")
    print(f"Flipped                      : {flip_results['n_flipped']} / {flip_results['n_total']}")
    print(f"Flip rate                    : {flip_results['flip_rate']:.1%}")

    print(f"\n=== Scene Graph Edit Distance (target objects only) ===")
    iso_results = evaluate_isolation(
        vine_model, test_items, patch, anchor, config, torch.device(device)
    )
    mode   = config.attack_mode
    target = config.target_category or config.target_relation

    attacked_label   = mode
    collateral_heads = [h for h in ('category', 'attribute', 'relation') if h != attacked_label]

    head_ct = {
        'category':  iso_results['total_category_changes'],
        'attribute': iso_results['total_attribute_changes'],
        'relation':  iso_results['total_relation_changes'],
    }
    n_obj = iso_results['total_target_objects']
    n_rel = iso_results.get('total_target_relations', 0)
    # Per-head denominator: category/attribute changes are counted per target
    # OBJECT, relation changes per target EDGE. Normalising the relation row by
    # objects (subjects) is what made it read >100%; by edges it equals flip rate.
    def _rate(head: str, ct: int) -> str:
        d = n_rel if head == 'relation' else n_obj
        return f"{ct/d:.0%}" if d > 0 else "0%"
    attacked_ct = head_ct[attacked_label] + (
        iso_results['total_disappeared'] if mode == 'category' else 0)

    _basis = f"{n_obj} objects" + (f", {n_rel} relation edges" if mode == 'relation' else "")
    print(f"Target objects evaluated     : {n_obj}"
          + (f"   ({n_rel} relation edges)" if mode == 'relation' else ""))
    print(f"Changes on target ({_basis}):")
    print(f"  [attacked]   {attacked_label:9s}: {head_ct[attacked_label]:4d}  "
          f"({_rate(attacked_label, head_ct[attacked_label])})   ← higher = stronger attack")
    for h in collateral_heads:
        print(f"  [collateral] {h:9s}: {head_ct[h]:4d}  ({_rate(h, head_ct[h])})"
              f"   ← lower = better isolation")
    print(f"  disappeared          : {iso_results['total_disappeared']:4d}  "
          f"({_rate('category', iso_results['total_disappeared'])})")
    print(f"  TOTAL changes        : {iso_results['total_changes']:4d}")

    print(f"Isolation score:")
    _pool = iso_results['pooled_isolation_score']
    if _pool is None:
        _why = "no target objects found" if n_obj == 0 else "attack changed nothing on the target"
        print(f"  pooled    : n/a  ({_why}, nothing to isolate)")
    else:
        print(f"  pooled    : {_pool:.1%}  "
              f"({attacked_ct}/{iso_results['total_changes']} of ALL changes hit the attacked head)")

    # Save results as JSON in the same directory as the patch file.
    # "_fields" documents every metric so the JSON is self-explaining.
    results = {
        "_fields": {
            "attack_mode": "which head was attacked: category | attribute | relation",
            "target": "attacked category (category/attribute) or relation predicate (relation)",
            "n_frames_requested": "frame-samples selected for the test split",
            "n_frames_loaded": "frame-samples that decoded (differs from requested only on a read error)",
            "n_frames_with_clean_target": "frames where the CLEAN model grounded >=1 target fact (effective sample size, NOT ground-truth presence)",
            "flip_rate": "HEADLINE attack strength = n_flipped / n_total",
            "n_total": "target FACTS the clean model grounded: objects (category/attribute) or relation EDGES (relation)",
            "n_flipped": "of n_total, how many the patch changed (relation: on-edge deleted or redirected)",
            "n_objects_patched": "distinct objects the isolation pass patched (relation: distinct SUBJECT pots; equals n_total for category/attribute)",
            "n_target_relations": "target relation EDGES (relation mode only; reconciles with n_total)",
            "total_category_changes": "target objects whose CATEGORY changed after patching",
            "total_attribute_changes": "target objects whose ATTRIBUTE changed",
            "total_relation_changes": "target edges whose RELATION changed",
            "total_disappeared": "target objects that left the scene graph entirely (category attack: counts as success; otherwise a confound, tracked apart)",
            "total_changes": "sum of the four change buckets above",
            "pooled_isolation_score": "fraction of ALL changes that hit the attacked head (1.0 = surgical; null = nothing changed)",
        },
        "patch":        args.patch,
        "attack_mode":  config.attack_mode,
        "target":       target,
        "n_frames_requested":         len(test_items),
        "n_frames_loaded":            n_loaded,
        "n_frames_with_clean_target": flip_results.get('n_clips_with_targets'),
        # ── attack strength ──
        "flip_rate":    flip_results['flip_rate'],
        "n_flipped":    flip_results['n_flipped'],
        "n_total":      flip_results['n_total'],
        # ── isolation (scene-graph edit distance) ──
        "n_objects_patched":       iso_results['total_target_objects'],
        "n_target_relations":      iso_results.get('total_target_relations', 0),
        "total_category_changes":  iso_results['total_category_changes'],
        "total_attribute_changes": iso_results['total_attribute_changes'],
        "total_relation_changes":  iso_results['total_relation_changes'],
        "total_disappeared":       iso_results['total_disappeared'],
        "total_changes":           iso_results['total_changes'],
        "pooled_isolation_score":  iso_results['pooled_isolation_score'],
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


def main():
    parser = argparse.ArgumentParser(description="NSAP adversarial patch attack on VINE")

    # Mode
    parser.add_argument("--mode",   default="category",
                        choices=["category", "attribute", "relation"],
                        help="Attack mode")
    parser.add_argument("--target", default="Toaster",
                        help="Target category string (AI2-THOR sim, e.g. 'Toaster', 'Pot', 'Television')")
    parser.add_argument("--relation",  default="next to",
                        help="Target relation string for --mode relation (e.g. 'next to', 'on')")
    parser.add_argument("--attribute", default="",
                        help="Target attribute string for --mode attribute (e.g. 'open', 'closed', 'on')")

    # Training
    parser.add_argument("--steps",    type=int,   default=2000,  help="Training steps")
    parser.add_argument("--batch",    type=int,   default=8,     help="EoT batch size")
    parser.add_argument("--lr",       type=float, default=0.01,  help="Adam learning rate")
    parser.add_argument("--coverage", type=float, default=0.3,   help="Patch coverage ratio")
    parser.add_argument("--eot-crops", type=int, default=0, dest="eot_crops",
                        help="Fix-B: # of jittered detector-like crops per step (0=off)")
    parser.add_argument("--lambda-c", type=float, default=1.0, dest="lambda_c",
                        help="Chameleon camouflage weight (higher = blends more, weaker attack).")
    parser.add_argument("--lambda-tv", type=float, default=1e-4, dest="lambda_tv",
                        help="Smoothness (total-variation) weight (higher = smoother, less visible noise).")
    parser.add_argument("--lambda-dino", type=float, default=0.5, dest="lambda_dino",
                        help="DINOv2 semantic-camouflage weight.")
    parser.add_argument("--object-init", action="store_true", dest="object_init",
                        help="Initialise the patch FROM a real crop of the target object "
                             "(the object's appearance under the footprint) so it STARTS hidden, "
                             "instead of a loud warm-start/random noise. Reuses --init-patch's anchor/size.")
    parser.add_argument("--lambda-content", type=float, default=0.0, dest="lambda_content",
                        help="Appearance anchor: pixel-L2 to the real object crop. The stealth↔flip "
                             "knob (high=hidden/weaker flip, low=stronger flip/louder). 0=off.")
    parser.add_argument("--clip-size",type=int,   default=224,
                        dest="clip_size", help="CLIP input size (224 or 336)")
    parser.add_argument("--no-chameleon", action="store_false", dest="use_chameleon",
                        help="Disable Chameleon naturalness losses (gram/tv/dino). "
                             "Frees VGG16 + DINOv2 from GPU — use on small-memory cards.")
    parser.add_argument("--max-clips", type=int, default=0, dest="max_clips",
                        help="Cap clips loaded into RAM (0 = no cap). Each clip's masks "
                             "cost ~tens of MB; cap this on machines with limited RAM.")
    parser.add_argument("--min-target-conf", type=float, default=0.0, dest="min_target_conf",
                        help="Min VINE softmax confidence for the target category to accept an "
                             "object as a target (0 = off). Drops weakly-classified targets. "
                             "Try 0.5.")
    parser.add_argument("--sim-data", default="", dest="sim_data",
                        help="Path to the SceneGraphBench (AI2-THOR) dataset to train/eval on "
                             "(NSAP's only data source). Defaults to dataset/sim/ if present "
                             "(from `python NSAP.py --with-dataset`).")
    parser.add_argument("--data-source", default="", dest="data_source",
                        help="Restrict to scenarios from one merge source (the 'source' tag in "
                             "the unified manifest, e.g. 'pot' for the staged relation scenes).")
    parser.add_argument("--all-data", action="store_true", dest="all_data",
                        help="Train on ALL sim scenarios (no train/test split).")
    parser.add_argument("--init-patch", default="", dest="init_patch",
                        help="Warm-start from this patch (.pt): reuse its anchor + weights and "
                             "continue training (e.g. Phase-2 chameleon camouflage).")
    parser.add_argument("--only-scenario", default="", dest="only_scenario",
                        help="Restrict train/eval to a single sim scenario id (chameleon experiment).")

    # Eval
    parser.add_argument("--eval",  action="store_true", help="Run evaluation instead of training")
    parser.add_argument("--patch", default=None, help="Path to saved patch .pt file for --eval")

    # Misc
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible runs (diversity sampling + EoT jitter).")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="output/patches", dest="output_dir")

    args = parser.parse_args()

    # Auto-use the downloaded dataset only when it is itself a single dataset root
    # (has manifest.json). A multi-dataset snapshot (dataset/sim/<name>/...) must be
    # selected explicitly, e.g. --sim-data dataset/sim/attribute_tv.
    if not args.sim_data:
        _default_sim = os.path.join(os.path.dirname(__file__), "dataset", "sim")
        if os.path.isfile(os.path.join(_default_sim, "manifest.json")):
            args.sim_data = _default_sim

    if args.eval:
        if not args.patch:
            parser.error("--eval requires --patch <path>")
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
