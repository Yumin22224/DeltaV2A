"""
Training script for Stage 2: Cross-Modal Mapping

Phase 2-A: Cross-modal mapping without generation (δC=0)
Phase 2-B: End-to-end with generation
Phase 2-C: Subjectivity space learning (δC predictor)

Based on System Specification v2.1
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    CrossModalDataset,
    SubjectivityDataset,
    get_audio_transform,
    get_image_transform,
    create_dataloader,
)
from src.models import (
    PriorEstimator,
    VisualDeltaEncoder,
    DeltaMappingModule,
    SEncoder,
    AudioGenerator,
    DeltaCPredictor,
)
from src.losses import (
    PseudoTargetLoss,
    ManifoldLoss,
    IdentityLoss,
    MonotonicityLoss,
    ConditionalPreservationLoss,
    CoherenceLoss,
    ConsistencyLoss,
    PairwiseRankingLoss,
    DirectionLoss,
    BoundedVarianceLoss,
    PriorRegularizationLoss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage2_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["2a", "2b", "2c"],
        default="2a",
        help="Training phase (2a: no gen, 2b: with gen, 2c: subjectivity)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def train_epoch_2a(
    visual_encoder,
    delta_mapping,
    p_align,
    prior_estimator,
    dataloader,
    optimizer,
    device,
    config,
    loss_pseudo,
    loss_manifold,
    loss_identity,
    loss_mono,
):
    """
    Stage 2-A training epoch (no generation, δC=0)
    """
    visual_encoder.train()
    delta_mapping.train()
    p_align.train()

    total_loss = 0
    total_pseudo = 0
    total_manifold = 0
    total_identity = 0
    total_mono = 0

    pbar = tqdm(dataloader, desc="Phase 2-A")

    for batch in pbar:
        # Move to device
        I_init = batch["image_init"].to(device)
        I_edit = batch["image_edit"].to(device)
        A_init = batch["audio_init_mel"].to(device)

        # C_anchor (δC=0)
        with torch.no_grad():
            C_anchor = prior_estimator(I_init, A_init)

        # Visual delta
        ΔV = visual_encoder(I_init, I_edit)

        # Delta mapping
        S_raw = delta_mapping(ΔV, C_anchor)

        # P_align
        S_final = p_align(S_raw)

        # Compute losses
        loss = 0
        loss_dict = {}

        # 1. Pseudo-target (rule-based)
        if loss_pseudo is not None:
            pseudo_loss = loss_pseudo(I_init, I_edit, S_final)
            loss_dict['pseudo'] = pseudo_loss.item()
            loss += config.training.phase_2a.loss_weights.pseudo * pseudo_loss
            total_pseudo += pseudo_loss.item()

        # 2. Manifold (S_proxy distribution alignment)
        if loss_manifold is not None:
            manifold_loss = loss_manifold(S_final)
            loss_dict['manifold'] = manifold_loss.item()
            loss += config.training.phase_2a.loss_weights.manifold * manifold_loss
            total_manifold += manifold_loss.item()

        # 3. Identity (ΔV=0 → S=0)
        if loss_identity is not None:
            # Sample 10% of batch with zero delta
            identity_loss = loss_identity(visual_encoder, S_final, I_init)
            loss_dict['identity'] = identity_loss.item()
            loss += config.training.phase_2a.loss_weights.identity * identity_loss
            total_identity += identity_loss.item()

        # 4. Monotonicity (||ΔV|| ↑ → Σg_h ↑)
        if loss_mono is not None:
            mono_loss = loss_mono(ΔV, S_final)
            loss_dict['mono'] = mono_loss.item()
            loss += config.training.phase_2a.loss_weights.mono * mono_loss
            total_mono += mono_loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(visual_encoder.parameters()) +
                list(delta_mapping.parameters()) +
                list(p_align.parameters()),
                config.training.gradient_clip,
            )

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss_dict)

    metrics = {
        'total': total_loss / len(dataloader),
        'pseudo': total_pseudo / len(dataloader),
        'manifold': total_manifold / len(dataloader),
        'identity': total_identity / len(dataloader),
        'mono': total_mono / len(dataloader),
    }

    return metrics


def train_epoch_2b(
    visual_encoder,
    delta_mapping,
    p_align,
    generator,
    prior_estimator,
    dataloader,
    optimizer_q,
    optimizer_g,
    device,
    config,
    loss_preserve,
    loss_coherence,
    loss_consistency,
    loss_rank,
):
    """
    Stage 2-B training epoch (end-to-end with generation)
    """
    visual_encoder.train()
    delta_mapping.train()
    p_align.train()
    generator.train()

    total_loss = 0
    total_preserve = 0
    total_coherence = 0
    total_consistency = 0
    total_rank = 0

    pbar = tqdm(dataloader, desc="Phase 2-B")

    for batch in pbar:
        # Move to device
        I_init = batch["image_init"].to(device)
        I_edit = batch["image_edit"].to(device)
        A_init = batch["audio_init_mel"].to(device)

        # C_anchor (δC=0)
        with torch.no_grad():
            C_anchor = prior_estimator(I_init, A_init)

        # Visual delta
        ΔV = visual_encoder(I_init, I_edit)

        # Delta mapping & align
        S_raw = delta_mapping(ΔV, C_anchor)
        S_final = p_align(S_raw)

        # Multi-sample generation (K=4)
        K = config.training.phase_2b.num_samples_per_input
        A_edit_samples = []

        for k in range(K):
            A_edit = generator(A_init, S_final, noise_level=0.5)
            A_edit_samples.append(A_edit)

        # Compute losses
        loss = 0
        loss_dict = {}

        # 1. Conditional Preservation
        if loss_preserve is not None:
            preserve_loss = loss_preserve(A_init, A_edit_samples, S_final)
            loss_dict['preserve'] = preserve_loss.item()
            loss += config.training.phase_2b.loss_weights.preserve * preserve_loss
            total_preserve += preserve_loss.item()

        # 2. Multi-level Coherence
        if loss_coherence is not None:
            coherence_loss = loss_coherence(I_init, I_edit, A_init, A_edit_samples, S_final)
            loss_dict['coherence'] = coherence_loss.item()
            loss += config.training.phase_2b.loss_weights.coherence * coherence_loss
            total_coherence += coherence_loss.item()

        # 3. Multi-sample Consistency
        if loss_consistency is not None:
            consistency_loss = loss_consistency(A_edit_samples, S_final)
            loss_dict['consistency'] = consistency_loss.item()
            loss += config.training.phase_2b.loss_weights.consistency * consistency_loss
            total_consistency += consistency_loss.item()

        # 4. Rank Consistency
        if loss_rank is not None:
            rank_loss = loss_rank(S_final, A_edit_samples)
            loss_dict['rank'] = rank_loss.item()
            loss += config.training.phase_2b.loss_weights.rank * rank_loss
            total_rank += rank_loss.item()

        # Backward
        optimizer_q.zero_grad()
        optimizer_g.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(visual_encoder.parameters()) +
                list(delta_mapping.parameters()) +
                list(p_align.parameters()),
                config.training.gradient_clip,
            )
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                config.training.gradient_clip,
            )

        optimizer_q.step()
        optimizer_g.step()

        total_loss += loss.item()
        pbar.set_postfix(loss_dict)

    metrics = {
        'total': total_loss / len(dataloader),
        'preserve': total_preserve / len(dataloader),
        'coherence': total_coherence / len(dataloader),
        'consistency': total_consistency / len(dataloader),
        'rank': total_rank / len(dataloader),
    }

    return metrics


def train_epoch_2c(
    delta_c_predictor,
    visual_encoder,
    delta_mapping,
    p_align,
    generator,
    s_encoder,
    prior_estimator,
    dataloader,
    optimizer,
    device,
    config,
    loss_direction,
    loss_manifold,
    loss_bounded,
    loss_prior,
):
    """
    Stage 2-C training epoch (subjectivity space learning)
    """
    delta_c_predictor.train()

    # Freeze other components
    visual_encoder.eval()
    delta_mapping.eval()
    p_align.eval()
    s_encoder.eval()

    total_loss = 0
    total_direction = 0
    total_manifold = 0
    total_bounded = 0
    total_prior = 0

    pbar = tqdm(dataloader, desc="Phase 2-C")

    for batch in pbar:
        # Move to device
        I_init = batch["image_init"].to(device)
        I_edit = batch["image_edit"].to(device)
        A_init = batch["audio_init_mel"].to(device)
        A_edit_candidates = batch["audio_edit_candidates"].to(device)  # (B, n_cand, 1, T, F)

        n_candidates = A_edit_candidates.shape[1]

        # Process each valid candidate
        results = []

        for c in range(n_candidates):
            A_edit = A_edit_candidates[:, c]  # (B, 1, T, F)

            # C_prior
            with torch.no_grad():
                C_prior = prior_estimator(I_init, A_init)

            # Predict δC
            δC = delta_c_predictor(I_init, A_init, I_edit)
            C_anchor = C_prior + δC

            # q mapping
            with torch.no_grad():
                ΔV = visual_encoder(I_init, I_edit)

            S_raw = delta_mapping(ΔV, C_anchor)
            S_final = p_align(S_raw)

            # Target from S_encoder
            with torch.no_grad():
                S_target = s_encoder(A_init, A_edit)

            results.append({
                'S_final': S_final,
                'S_target': S_target,
                'C_anchor': C_anchor,
                'C_prior': C_prior,
                'δC': δC,
            })

        # Compute losses
        loss = 0
        loss_dict = {}

        # 1. Direction (S_final → S_target)
        if loss_direction is not None:
            direction_loss = torch.mean(torch.stack([
                loss_direction(r['S_final'], r['S_target'])
                for r in results
            ]))
            loss_dict['direction'] = direction_loss.item()
            loss += config.training.phase_2c.loss_weights.direction * direction_loss
            total_direction += direction_loss.item()

        # 2. Manifold
        if loss_manifold is not None:
            manifold_loss = torch.mean(torch.stack([
                loss_manifold(r['S_final'])
                for r in results
            ]))
            loss_dict['manifold'] = manifold_loss.item()
            loss += config.training.phase_2c.loss_weights.manifold * manifold_loss
            total_manifold += manifold_loss.item()

        # 3. Bounded variance
        if loss_bounded is not None:
            bounded_loss = loss_bounded(results)
            loss_dict['bounded'] = bounded_loss.item()
            loss += config.training.phase_2c.loss_weights.bounded * bounded_loss
            total_bounded += bounded_loss.item()

        # 4. Prior regularization
        if loss_prior is not None:
            prior_loss = torch.mean(torch.stack([
                loss_prior(r['C_anchor'], r['C_prior'])
                for r in results
            ]))
            loss_dict['prior'] = prior_loss.item()
            loss += config.training.phase_2c.loss_weights.prior * prior_loss
            total_prior += prior_loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                delta_c_predictor.parameters(),
                config.training.gradient_clip,
            )

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss_dict)

    metrics = {
        'total': total_loss / len(dataloader),
        'direction': total_direction / len(dataloader),
        'manifold': total_manifold / len(dataloader),
        'bounded': total_bounded / len(dataloader),
        'prior': total_prior / len(dataloader),
    }

    return metrics


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")
    print(OmegaConf.to_yaml(config))

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize wandb (optional)
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            name=f"stage2_{args.phase}",
            config=OmegaConf.to_container(config),
        )

    # Create dataset and dataloader
    audio_transform = get_audio_transform(config.data.audio)
    image_transform = get_image_transform(config.data.image)

    if args.phase in ["2a", "2b"]:
        # Cross-modal dataset
        dataset = CrossModalDataset(
            data_dir=config.data.cross_modal.dataset_path,
            audio_transform=audio_transform,
            image_transform=image_transform,
            split="train",
        )
        batch_size = config.data.cross_modal.batch_size
        epochs = config.training.phase_2a.epochs if args.phase == "2a" else config.training.phase_2b.epochs
        lr = config.training.phase_2a.learning_rate if args.phase == "2a" else config.training.phase_2b.learning_rate

    else:  # 2c
        # Subjectivity dataset
        dataset = SubjectivityDataset(
            data_dir=config.data.subjectivity.dataset_path,
            audio_transform=audio_transform,
            image_transform=image_transform,
            split="train",
            min_validity=config.data.subjectivity.min_validity,
            num_candidates=config.training.phase_2c.num_valid_candidates,
        )
        batch_size = config.data.subjectivity.batch_size
        epochs = config.training.phase_2c.epochs
        lr = config.training.phase_2c.learning_rate

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Load pretrained components
    print("\nLoading pretrained components...")

    # Prior estimator (frozen)
    from src.models import HardPrior, SoftPrior
    # TODO: Load from checkpoint

    prior_estimator = None  # Placeholder
    print("  Prior estimator loaded")

    # Create models based on phase
    if args.phase in ["2a", "2b"]:
        # Visual encoder
        visual_encoder = VisualDeltaEncoder(
            low_level_dim=config.model.visual_encoder.low_level.output_dim,
            high_level_dim=config.model.visual_encoder.high_level.output_dim,
            delta_dim=config.model.visual_encoder.high_level.output_dim,
        ).to(device)

        # Delta mapping (q)
        delta_mapping = DeltaMappingModule(
            delta_dim=config.model.delta_mapping.delta_dim,
            num_heads=config.model.num_heads,
            head_dim=config.model.head_dim,
            hidden_dim=config.model.delta_mapping.hidden_dim,
        ).to(device)

        # P_align
        # TODO: Implement P_align module
        p_align = nn.Identity()  # Placeholder

        print("  Visual encoder and delta mapping created")

    if args.phase == "2b":
        # Generator
        generator = AudioGenerator(
            pretrained_model=config.model.audio_generator.pretrained_model,
            num_heads=config.model.num_heads,
            head_dim=config.model.head_dim,
            use_lora=config.model.audio_generator.use_lora,
            lora_rank=config.model.audio_generator.lora_rank,
        ).to(device)

        print("  Audio generator created")

    if args.phase == "2c":
        # δC predictor
        delta_c_predictor = DeltaCPredictor(
            visual_dim=config.model.delta_c_predictor.visual_dim,
            audio_dim=config.model.delta_c_predictor.audio_dim,
            hidden_dim=config.model.delta_c_predictor.hidden_dim,
            N_v=config.model.delta_c_predictor.N_v,
            num_heads=config.model.num_heads,
        ).to(device)

        # Load frozen components
        # visual_encoder, delta_mapping, p_align, s_encoder
        # TODO: Load from checkpoints

        print("  δC predictor created")

    # Initialize losses based on phase
    if args.phase == "2a":
        loss_pseudo = PseudoTargetLoss(
            rules=config.hard_prior_rules.rules_file,  # TODO: Load rules
        ).to(device)

        loss_manifold = ManifoldLoss(
            # TODO: Load S_proxy statistics
        ).to(device)

        loss_identity = IdentityLoss().to(device)
        loss_mono = MonotonicityLoss().to(device)

    elif args.phase == "2b":
        loss_preserve = ConditionalPreservationLoss(
            beta=config.losses.preserve.beta,
        ).to(device)

        loss_coherence = CoherenceLoss(
            # TODO: Pass hard prior rules
        ).to(device)

        loss_consistency = ConsistencyLoss(
            struct_heads=config.losses.consistency.struct_heads,
            style_heads=config.losses.consistency.style_heads,
        ).to(device)

        loss_rank = PairwiseRankingLoss().to(device)

    elif args.phase == "2c":
        loss_direction = DirectionLoss().to(device)
        loss_manifold = ManifoldLoss().to(device)
        loss_bounded = BoundedVarianceLoss().to(device)
        loss_prior = PriorRegularizationLoss().to(device)

    # Optimizers
    if args.phase == "2a":
        optimizer = torch.optim.AdamW(
            list(visual_encoder.parameters()) +
            list(delta_mapping.parameters()) +
            list(p_align.parameters()),
            lr=lr,
            weight_decay=config.optimizer.weight_decay,
        )

    elif args.phase == "2b":
        optimizer_q = torch.optim.AdamW(
            list(visual_encoder.parameters()) +
            list(delta_mapping.parameters()) +
            list(p_align.parameters()),
            lr=lr,
            weight_decay=config.optimizer.weight_decay,
        )
        optimizer_g = torch.optim.AdamW(
            generator.parameters(),
            lr=lr,
            weight_decay=config.optimizer.weight_decay,
        )

    elif args.phase == "2c":
        optimizer = torch.optim.AdamW(
            delta_c_predictor.parameters(),
            lr=lr,
            weight_decay=config.optimizer.weight_decay,
        )

    # TODO: Resume from checkpoint if provided

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        if args.phase == "2a":
            metrics = train_epoch_2a(
                visual_encoder, delta_mapping, p_align, prior_estimator,
                dataloader, optimizer, device, config,
                loss_pseudo, loss_manifold, loss_identity, loss_mono,
            )

        elif args.phase == "2b":
            metrics = train_epoch_2b(
                visual_encoder, delta_mapping, p_align, generator, prior_estimator,
                dataloader, optimizer_q, optimizer_g, device, config,
                loss_preserve, loss_coherence, loss_consistency, loss_rank,
            )

        elif args.phase == "2c":
            metrics = train_epoch_2c(
                delta_c_predictor, visual_encoder, delta_mapping, p_align,
                generator, s_encoder, prior_estimator,
                dataloader, optimizer, device, config,
                loss_direction, loss_manifold, loss_bounded, loss_prior,
            )

        print(f"Train loss: {metrics['total']:.4f}")
        for key, value in metrics.items():
            if key != 'total':
                print(f"  {key}: {value:.4f}")

        # Log to wandb
        if config.logging.use_wandb:
            log_dict = {"epoch": epoch}
            log_dict.update({f"train/{k}": v for k, v in metrics.items()})
            wandb.log(log_dict)

        # Save checkpoint
        if (epoch + 1) % config.logging.save_every_n_epochs == 0:
            save_dir = Path(config.output.output_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "config": config,
            }

            if args.phase == "2a":
                checkpoint.update({
                    "visual_encoder": visual_encoder.state_dict(),
                    "delta_mapping": delta_mapping.state_dict(),
                    "p_align": p_align.state_dict() if hasattr(p_align, 'state_dict') else None,
                    "optimizer": optimizer.state_dict(),
                })
            elif args.phase == "2b":
                checkpoint.update({
                    "visual_encoder": visual_encoder.state_dict(),
                    "delta_mapping": delta_mapping.state_dict(),
                    "p_align": p_align.state_dict() if hasattr(p_align, 'state_dict') else None,
                    "generator": generator.state_dict(),
                    "optimizer_q": optimizer_q.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                })
            elif args.phase == "2c":
                checkpoint["delta_c_predictor"] = delta_c_predictor.state_dict()
                checkpoint["optimizer"] = optimizer.state_dict()

            save_path = save_dir / f"checkpoint_phase{args.phase}_epoch{epoch+1}.pt"
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")

    # Save final models
    save_dir = Path(config.output.output_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ["2a", "2b"]:
        torch.save(visual_encoder.state_dict(), save_dir / "visual_encoder_final.pt")
        torch.save(delta_mapping.state_dict(), save_dir / "delta_mapping_final.pt")
        if hasattr(p_align, 'state_dict'):
            torch.save(p_align.state_dict(), save_dir / "p_align_final.pt")

    if args.phase == "2b":
        torch.save(generator.state_dict(), save_dir / "generator_final.pt")

    if args.phase == "2c":
        torch.save(delta_c_predictor.state_dict(), save_dir / "delta_c_predictor_final.pt")

    print(f"Saved final models to {save_dir}")

    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
