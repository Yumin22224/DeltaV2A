"""
Inference script for DeltaV2A

Given (I_init, A_init, I_edit), generate A_edit
"""

import sys
import argparse
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import get_audio_transform, get_image_transform
from src.models import (
    PriorEstimator,
    HardPrior,
    SoftPrior,
    VisualDeltaEncoder,
    DeltaMappingModule,
    AudioGenerator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DeltaV2A Inference")
    parser.add_argument(
        "--image_init",
        type=str,
        required=True,
        help="Path to initial image",
    )
    parser.add_argument(
        "--image_edit",
        type=str,
        required=True,
        help="Path to edited image",
    )
    parser.add_argument(
        "--audio_init",
        type=str,
        required=True,
        help="Path to initial audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output audio",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage2_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/stage2",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.5,
        help="Noise level for editing (0-1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def load_models(config, checkpoint_dir, device):
    """Load all trained models"""
    checkpoint_dir = Path(checkpoint_dir)

    # Prior Estimator
    # TODO: Load actual trained prior
    hard_prior = HardPrior(rules=[], num_heads=config.model.num_heads)
    soft_prior = SoftPrior(
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        freeze=True,
    )
    prior_estimator = PriorEstimator(
        hard_prior=hard_prior,
        soft_prior=soft_prior,
        alpha=config.model.prior.alpha,
    ).to(device)

    # Visual Delta Encoder
    visual_encoder = VisualDeltaEncoder(
        delta_dim=config.model.visual.delta_dim,
    ).to(device)

    # Load visual encoder weights
    visual_path = checkpoint_dir / "visual_encoder_final.pt"
    if visual_path.exists():
        visual_encoder.load_state_dict(torch.load(visual_path, map_location=device))
        print(f"Loaded visual encoder from {visual_path}")

    # Delta Mapping Module
    delta_mapping = DeltaMappingModule(
        delta_dim=config.model.visual.delta_dim,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        use_manifold_projection=True,
    ).to(device)

    # Load delta mapping weights
    delta_path = checkpoint_dir / "delta_mapping_final.pt"
    if delta_path.exists():
        delta_mapping.load_state_dict(torch.load(delta_path, map_location=device))
        print(f"Loaded delta mapping from {delta_path}")

    # Load S_proxy statistics
    stats_path = Path(config.pretrained.s_proxy_stats)
    if stats_path.exists():
        delta_mapping.load_proxy_statistics(str(stats_path))

    # Audio Generator
    generator = AudioGenerator(
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        use_lora=True,
    ).to(device)

    # Load generator weights
    gen_path = checkpoint_dir / "generator_final.pt"
    if gen_path.exists():
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        print(f"Loaded generator from {gen_path}")

    # Set to eval mode
    prior_estimator.eval()
    visual_encoder.eval()
    delta_mapping.eval()
    generator.eval()

    return prior_estimator, visual_encoder, delta_mapping, generator


@torch.no_grad()
def inference(
    image_init_path,
    image_edit_path,
    audio_init_path,
    output_path,
    prior_estimator,
    visual_encoder,
    delta_mapping,
    generator,
    audio_transform,
    image_transform,
    noise_level=0.5,
    device="cpu",
):
    """
    Main inference pipeline

    Args:
        image_init_path: Path to initial image
        image_edit_path: Path to edited image
        audio_init_path: Path to initial audio
        output_path: Path to save output
        prior_estimator: Prior estimator model
        visual_encoder: Visual delta encoder model
        delta_mapping: Delta mapping module
        generator: Audio generator model
        audio_transform: Audio preprocessing
        image_transform: Image preprocessing
        noise_level: Editing strength
        device: Device to use
    """
    print("\n=== DeltaV2A Inference ===")
    print(f"Initial image: {image_init_path}")
    print(f"Edited image: {image_edit_path}")
    print(f"Initial audio: {audio_init_path}")
    print(f"Noise level: {noise_level}")

    # Load and preprocess inputs
    print("\n[1/5] Loading inputs...")
    image_init = image_transform(image_init_path).unsqueeze(0).to(device)
    image_edit = image_transform(image_edit_path).unsqueeze(0).to(device)
    audio_dict = audio_transform(audio_init_path, return_waveform=True)
    audio_init_mel = audio_dict["mel"].unsqueeze(0).to(device)
    audio_init_wav = audio_dict["waveform"]

    # Estimate C_anchor
    print("[2/5] Estimating C_anchor...")
    C_anchor = prior_estimator.estimate_prior_only(image_init, audio_init_mel)

    # Encode visual delta
    print("[3/5] Encoding visual delta...")
    delta_V = visual_encoder(image_init, image_edit)

    # Map to control signals
    print("[4/5] Mapping to audio controls...")
    S_final = delta_mapping(delta_V, C_anchor)

    # Print control info
    info = delta_mapping.get_head_info(S_final)
    print("  Control signals:")
    for k, v in info.items():
        print(f"    {k}: {v:.4f}")

    # Generate edited audio
    print("[5/5] Generating edited audio...")
    audio_edit_mel = generator(audio_init_mel, S_final, noise_level=noise_level)

    # TODO: Convert mel to waveform using vocoder
    # For now, save mel spectrogram
    print(f"\nSaving output to {output_path}")
    # torch.save(audio_edit_mel, output_path.replace('.wav', '_mel.pt'))

    # Placeholder: save input audio as output for testing
    torchaudio.save(output_path, audio_init_wav, 16000)
    print("Note: Vocoder not implemented yet, saved input audio")

    print("\n✓ Inference complete!")
    return audio_edit_mel


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create transforms
    audio_transform = get_audio_transform(config.data.audio)
    image_transform = get_image_transform(config.data.image)

    # Load models
    print("\nLoading models...")
    prior_estimator, visual_encoder, delta_mapping, generator = load_models(
        config, args.checkpoint_dir, device
    )

    # Run inference
    inference(
        args.image_init,
        args.image_edit,
        args.audio_init,
        args.output,
        prior_estimator,
        visual_encoder,
        delta_mapping,
        generator,
        audio_transform,
        image_transform,
        noise_level=args.noise_level,
        device=device,
    )


if __name__ == "__main__":
    main()
