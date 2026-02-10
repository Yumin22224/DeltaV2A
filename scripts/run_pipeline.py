#!/usr/bin/env python
"""
DeltaV2A Pipeline Runner (Phase A-C)

Commands:
  precompute    - Phase A: Build vocab, correspondence matrix, inverse mapping DB
  train         - Phase B: Train the audio controller
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


def get_audio_paths(config: dict) -> list:
    """Get audio file paths from config."""
    audio_dir = Path(config['data']['audio_dir'])
    extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    paths = []
    for ext in extensions:
        paths.extend(audio_dir.glob(f'*{ext}'))
        paths.extend(audio_dir.glob(f'**/*{ext}'))
    paths = sorted(set(str(p) for p in paths))
    max_files = config['data'].get('max_audio_files')
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
    from src.correspondence import compute_correspondence_matrix
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
    """Phase B: Train the audio controller."""
    from src.controller import train_controller
    from src.effects.pedalboard_effects import get_total_param_count

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
        learning_rate=ctrl_cfg['learning_rate'],
        val_split=ctrl_cfg['val_split'],
        hidden_dims=ctrl_cfg.get('hidden_dims'),
        dropout=ctrl_cfg.get('dropout', 0.1),
        device=config.get('device', 'cpu'),
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
