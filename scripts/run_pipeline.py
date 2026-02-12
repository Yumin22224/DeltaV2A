#!/usr/bin/env python
"""
DeltaV2A Pipeline Runner (Phase A-C)

Commands:
  precompute    - Phase A: Build vocab, correspondence matrix, inverse mapping DB
  train         - Phase B: Train controller + visual siamese encoder
  infer         - Phase C: Run inference (I, I', A) -> A'
  all           - precompute + train

Usage:
    python scripts/run_pipeline.py precompute --config configs/pipeline.yaml
    python scripts/run_pipeline.py train --config configs/pipeline.yaml
    python scripts/run_pipeline.py infer --config configs/pipeline.yaml \\
        --original img_orig.jpg --edited img_edit.jpg \\
        --audio input.wav --output output.wav
    python scripts/run_pipeline.py all --config configs/pipeline.yaml
"""

import argparse
import sys
import json
import shutil
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent))


# =============================================================================
# CONFIG
# =============================================================================

def load_config(config_path: str) -> dict:
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_embedders(config: dict):
    """Load CLIP + CLAP embedders (with CLAP argv hack)."""
    import sys as _sys
    _saved_argv = _sys.argv[:]
    _sys.argv = [_sys.argv[0]]
    try:
        from src.models import CLIPEmbedder, CLAPEmbedder
    finally:
        _sys.argv = _saved_argv

    device = config.get('device', 'cpu')
    clip_cfg = config['model']['clip']
    clap_cfg = config['model']['clap']

    print(f"\nLoading embedders on {device}...")
    clip = CLIPEmbedder(
        model_name=clip_cfg['name'],
        pretrained=clip_cfg['pretrained'],
        device=device,
    )
    clap = CLAPEmbedder(
        model_id=clap_cfg['model_id'],
        enable_fusion=clap_cfg['enable_fusion'],
        max_duration=clap_cfg['max_duration'],
        device=device,
    )
    print("  Embedders loaded.")
    return clip, clap


def _resolve_data_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    candidate = (base_dir / p).resolve()
    if candidate.exists():
        return candidate
    return (Path.cwd() / p).resolve()


def _load_audio_paths_from_manifest(manifest_path: Path, split: str) -> list:
    paths = []
    if not manifest_path.exists():
        raise FileNotFoundError(f"audio_split_manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            rel_or_abs = str(row["path"])
            resolved = _resolve_data_path(rel_or_abs, manifest_path.parent)
            paths.append(str(resolved))
    return paths


def _load_audio_paths_from_list(list_path: Path) -> list:
    paths = []
    if not list_path.exists():
        raise FileNotFoundError(f"audio_list not found: {list_path}")
    with open(list_path, "r") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            resolved = _resolve_data_path(text, list_path.parent)
            paths.append(str(resolved))
    return paths


def get_audio_paths(config: dict) -> list:
    """Get audio file paths from config."""
    data_cfg = config['data']
    paths: list[str] = []

    manifest = data_cfg.get('audio_split_manifest')
    if manifest:
        split = data_cfg.get('audio_split', 'train')
        manifest_path = _resolve_data_path(manifest, Path.cwd())
        paths = _load_audio_paths_from_manifest(manifest_path, split)
        print(f"Loaded {len(paths)} audio files from manifest split '{split}': {manifest_path}")
    else:
        audio_list = data_cfg.get('audio_list')
        if audio_list:
            list_path = _resolve_data_path(audio_list, Path.cwd())
            paths = _load_audio_paths_from_list(list_path)
            print(f"Loaded {len(paths)} audio files from list: {list_path}")
        else:
            audio_dir = Path(data_cfg['audio_dir'])
            extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            discovered = []
            for ext in extensions:
                discovered.extend(audio_dir.glob(f'*{ext}'))
                discovered.extend(audio_dir.glob(f'**/*{ext}'))
            paths = [str(p.resolve()) for p in sorted(set(discovered))]

    # Keep only existing files from explicit list/manifest to avoid runtime crashes.
    paths = [p for p in paths if Path(p).exists()]
    paths = sorted(set(paths))

    max_files = data_cfg.get('max_audio_files')
    if max_files:
        paths = paths[:max_files]
    return paths


def get_image_paths(config: dict) -> list:
    """Get image file paths from config."""
    image_dir = Path(config['data']['image_dir'])
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    paths = []
    for ext in extensions:
        paths.extend(image_dir.glob(f'*{ext}'))
        paths.extend(image_dir.glob(f'**/*{ext}'))
    paths = sorted(set(str(p) for p in paths))
    max_files = config['data'].get('max_image_files')
    if max_files:
        paths = paths[:max_files]
    return paths


# =============================================================================
# COMMAND: PRECOMPUTE (Phase A)
# =============================================================================

def run_precompute(config: dict):
    """Phase A: Build vocab, correspondence matrix, inverse mapping DB."""
    import numpy as np
    from src.vocab import StyleVocabulary
    from src.correspondence import compute_correspondence_matrix, save_correspondence_heatmap
    from src.database import build_inverse_mapping_db
    from src.effects.pedalboard_effects import get_total_param_count

    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    clip, clap = load_embedders(config)
    effect_names = config['effects']['active']

    # --- Phase A-1: Build Style Vocabularies ---
    print("\n" + "=" * 60)
    print("PHASE A-1: BUILD STYLE VOCABULARIES")
    print("=" * 60)

    vocab = StyleVocabulary()
    vocab.build_img_vocab(clip)
    vocab.build_aud_vocab(clap)
    vocab.save(str(output_dir))

    # --- Phase A-2: Correspondence Matrix ---
    print("\n" + "=" * 60)
    print("PHASE A-2: BUILD CORRESPONDENCE MATRIX")
    print("=" * 60)

    sbert_model = config['correspondence']['sbert_model']
    corr = compute_correspondence_matrix(
        vocab.img_vocab.keywords,
        vocab.aud_vocab.keywords,
        sbert_model_name=sbert_model,
    )
    corr.save(str(output_dir / "correspondence_matrix.npz"))
    try:
        save_correspondence_heatmap(
            correspondence=corr,
            output_path=str(output_dir / "correspondence_heatmap.png"),
        )
    except Exception as e:
        print(f"Warning: Failed to save correspondence heatmap: {e}")

    # --- Phase A-3: Inverse Mapping Database ---
    print("\n" + "=" * 60)
    print("PHASE A-3: BUILD INVERSE MAPPING DATABASE")
    print("=" * 60)

    audio_paths = get_audio_paths(config)
    print(f"Found {len(audio_paths)} audio files")

    if not audio_paths:
        print("ERROR: No audio files found. Skipping inverse mapping.")
        return

    inv_cfg = config['inverse_mapping']
    db = build_inverse_mapping_db(
        audio_paths=audio_paths,
        clap_embedder=clap,
        aud_vocab_embeddings=vocab.aud_vocab.embeddings,
        effect_names=effect_names,
        output_path=str(output_dir / "inverse_mapping.h5"),
        num_augmentations_per_audio=inv_cfg['augmentations_per_audio'],
        sample_rate=config['model']['clap']['sample_rate'],
        max_duration=config['model']['clap']['max_duration'],
        temperature=inv_cfg['temperature'],
        seed=inv_cfg['seed'],
        save_augmented_audio=config['data'].get('save_augmented_audio', True),
        augmented_audio_dir=config['data'].get('augmented_audio_dir', "data/augmented/pipeline/audio"),
        min_active_effects=inv_cfg.get('min_active_effects', 1),
        max_active_effects=inv_cfg.get('max_active_effects'),
        effect_sampling_weights=inv_cfg.get('effect_sampling_weights'),
    )

    # Save pipeline config for inference loading
    pipeline_config = {
        'effect_names': effect_names,
        'total_params': get_total_param_count(effect_names),
        'sample_rate': config['model']['clap']['sample_rate'],
        'projection_dim': config['visual_encoder']['projection_dim'],
        'top_k': config['inference']['top_k'],
        'img_vocab_size': vocab.img_vocab.size,
        'aud_vocab_size': vocab.aud_vocab.size,
    }
    with open(output_dir / 'pipeline_config.json', 'w') as f:
        json.dump(pipeline_config, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE A COMPLETE")
    print("=" * 60)


# =============================================================================
# COMMAND: TRAIN (Phase B)
# =============================================================================

def run_train(config: dict):
    """Phase B: Train controller + visual siamese encoder."""
    from src.controller import train_controller, run_controller_post_train_analysis
    from src.effects.pedalboard_effects import get_total_param_count
    from src.inference import train_visual_encoder
    from src.vocab import StyleVocabulary
    from src.models import CLIPEmbedder

    output_dir = Path(config['output']['dir'])
    db_path = output_dir / "inverse_mapping.h5"

    if not db_path.exists():
        print("ERROR: Inverse mapping DB not found. Run 'precompute' first.")
        return

    # Load pipeline config
    with open(output_dir / 'pipeline_config.json', 'r') as f:
        pipeline_cfg = json.load(f)

    ctrl_cfg = config['controller']
    effect_names = config['effects']['active']
    controller_lr = float(ctrl_cfg['learning_rate'])

    print("\n" + "=" * 60)
    print("PHASE B: TRAIN AUDIO CONTROLLER")
    print("=" * 60)

    train_controller(
        db_path=str(db_path),
        output_dir=str(output_dir / "controller"),
        style_vocab_size=pipeline_cfg['aud_vocab_size'],
        total_params=get_total_param_count(effect_names),
        batch_size=ctrl_cfg['batch_size'],
        num_epochs=ctrl_cfg['num_epochs'],
        learning_rate=controller_lr,
        val_split=ctrl_cfg['val_split'],
        hidden_dims=ctrl_cfg.get('hidden_dims'),
        dropout=ctrl_cfg.get('dropout', 0.1),
        device=config.get('device', 'cpu'),
    )

    # Mirror best checkpoint to root for inference loader compatibility.
    best_ckpt = output_dir / "controller" / "controller_best.pt"
    if best_ckpt.exists():
        shutil.copy2(best_ckpt, output_dir / "controller_best.pt")

    # Phase B-1 post-train analysis (pred vs target report + A/B renders)
    analysis_cfg = ctrl_cfg.get('post_train_analysis', {})
    if analysis_cfg.get('enabled', True):
        print("\n" + "=" * 60)
        print("PHASE B-1: CONTROLLER POST-TRAIN ANALYSIS")
        print("=" * 60)
        try:
            def _pick(name, default):
                val = analysis_cfg.get(name, None)
                return default if val is None else val

            manifest_path = analysis_cfg.get('manifest_path')
            if manifest_path is None:
                aug_audio_dir = config.get('data', {}).get('augmented_audio_dir')
                if aug_audio_dir:
                    manifest_path = str(Path(aug_audio_dir) / "manifest.jsonl")

            analysis_report = run_controller_post_train_analysis(
                artifacts_dir=str(output_dir),
                out_dir=_pick('out_dir', None),
                val_split=float(_pick('val_split', ctrl_cfg.get('val_split', 0.2))),
                split_seed=int(_pick('split_seed', 42)),
                batch_size=int(_pick('batch_size', 128)),
                num_renders=int(_pick('num_renders', 5)),
                sample_rate=int(_pick('sample_rate', config['model']['clap'].get('sample_rate', 48000))),
                max_duration=float(_pick('max_duration', config['model']['clap'].get('max_duration', 20.0))),
                device=str(_pick('device', config.get('device', 'cpu'))),
                manifest_path=manifest_path,
                hidden_dims=_pick('hidden_dims', ctrl_cfg.get('hidden_dims', [512, 256, 128])),
                dropout=float(_pick('dropout', ctrl_cfg.get('dropout', 0.1))),
            )
            report_path = Path(analysis_report['val_metrics_summary_json']).parent / "analysis_report.json"
            print(f"Controller analysis report: {report_path}")
            print(f"  Best val loss: {analysis_report['curve_summary']['best_val_loss']:.6f}")
            print(f"  Rendered examples: {analysis_report['num_rendered_examples']}")
        except Exception as e:
            print(f"WARNING: Controller post-train analysis failed: {e}")

    # Phase B-2: Train visual siamese encoder
    ve_cfg = config.get('visual_encoder', {}).get('training', {})
    if ve_cfg.get('enabled', True):
        print("\n" + "=" * 60)
        print("PHASE B-2: TRAIN VISUAL SIAMESE ENCODER")
        print("=" * 60)

        image_paths = get_image_paths(config)
        if not image_paths:
            print("WARNING: No image files found. Skipping visual encoder training.")
        else:
            device = config.get('device', 'cpu')
            clip_cfg = config['model']['clip']
            clip = CLIPEmbedder(
                model_name=clip_cfg['name'],
                pretrained=clip_cfg['pretrained'],
                device=device,
            )

            vocab = StyleVocabulary()
            vocab.load(str(output_dir))
            if vocab.img_vocab is None:
                raise RuntimeError("IMG_VOCAB not found. Run precompute first.")

            train_visual_encoder(
                image_paths=image_paths,
                clip_embedder=clip,
                img_vocab_embeddings=vocab.img_vocab.embeddings,
                save_path=str(output_dir / "visual_encoder.pt"),
                projection_dim=config['visual_encoder'].get('projection_dim', 768),
                dropout=config['visual_encoder'].get('dropout', 0.1),
                batch_size=ve_cfg.get('batch_size', 32),
                num_epochs=ve_cfg.get('num_epochs', 40),
                learning_rate=float(ve_cfg.get('learning_rate', 1e-4)),
                val_split=ve_cfg.get('val_split', 0.2),
                augmentations_per_image=ve_cfg.get('augmentations_per_image', 2),
                effect_types=ve_cfg.get(
                    'effect_types',
                    [
                        "adaptive_blur",
                        "motion_blur",
                        "adaptive_sharpen",
                        "add_noise",
                        "spread",
                        "sepia_tone",
                        "solarize",
                    ],
                ),
                intensities=ve_cfg.get('intensities', ["low", "mid", "high"]),
                save_augmented=ve_cfg.get('save_augmented_images', False),
                augmented_dir=ve_cfg.get('augmented_image_dir', "data/augmented/pipeline/images"),
                style_temperature=ve_cfg.get('style_temperature', 0.07),
                contrastive_margin=ve_cfg.get('contrastive_margin', 0.3),
                loss_weight_align=ve_cfg.get('loss_weight_align', 1.0),
                loss_weight_style=ve_cfg.get('loss_weight_style', 0.5),
                loss_weight_contrastive=ve_cfg.get('loss_weight_contrastive', 0.2),
                seed=ve_cfg.get('seed', 42),
                device=device,
            )

    print("\n" + "=" * 60)
    print("PHASE B COMPLETE")
    print("=" * 60)


# =============================================================================
# COMMAND: INFER (Phase C)
# =============================================================================

def run_infer(config: dict, args):
    """Phase C: Run inference."""
    from src.inference import DeltaV2APipeline

    output_dir = Path(config['output']['dir'])

    clip, clap = load_embedders(config)

    print("\nLoading pipeline...")
    pipeline = DeltaV2APipeline.load(
        artifacts_dir=str(output_dir),
        clip_embedder=clip,
        clap_embedder=clap,
        device=config.get('device', 'cpu'),
    )

    print("\nRunning inference...")
    result = pipeline.infer_from_paths(
        original_image_path=args.original,
        edited_image_path=args.edited,
        input_audio_path=args.audio,
        output_audio_path=args.output,
    )

    print(f"\nTop image styles:")
    for term, score in zip(result.top_k_img_terms, result.top_k_img_scores):
        print(f"  {score:.4f}  {term}")

    print(f"\nPredicted params:")
    for effect, params in result.predicted_params_dict.items():
        print(f"  {effect}: {params}")


# =============================================================================
# COMMAND: ALL
# =============================================================================

def run_all(config: dict):
    """Run precompute + train."""
    run_precompute(config)
    run_train(config)
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE (A + B)")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeltaV2A Pipeline (Phase A-C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'command',
        choices=['precompute', 'train', 'infer', 'all'],
        help="Command to run",
    )
    parser.add_argument('--config', type=str, default='configs/pipeline.yaml')

    # Inference-specific args
    parser.add_argument('--original', type=str, help="Original image path (infer)")
    parser.add_argument('--edited', type=str, help="Edited image path (infer)")
    parser.add_argument('--audio', type=str, help="Input audio path (infer)")
    parser.add_argument('--output', type=str, help="Output audio path (infer)")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    try:
        if args.command == 'precompute':
            run_precompute(config)
        elif args.command == 'train':
            run_train(config)
        elif args.command == 'infer':
            if not all([args.original, args.edited, args.audio]):
                print("ERROR: infer requires --original, --edited, and --audio")
                sys.exit(1)
            run_infer(config, args)
        elif args.command == 'all':
            run_all(config)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
