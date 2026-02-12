"""
DeltaV2A Inference Pipeline (Phase C)

End-to-end: (I, I', A) -> A'

Steps:
    1. Visual style extraction:
       - siamese mode: (I, I') -> z_visual
       - direct mode: Sim(I', IMG_VOCAB) - Sim(I, IMG_VOCAB)
    2. Style Retrieval: top-k with sim/delta scores
    3. Cross-Modal Mapping: identity transfer (shared IMG/AUD vocab axes)
    4. Audio Controller: CLAP(A) + audio_style -> predicted params
    5. Rendering: pedalboard applies params -> A'
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

from .visual_encoder import VisualEncoder
from ..vocab.style_vocab import StyleVocabulary
from ..controller.model import AudioController
from ..effects.pedalboard_effects import PedalboardRenderer, denormalize_params


@dataclass
class InferenceResult:
    """Result from the DeltaV2A inference pipeline."""
    z_visual: np.ndarray
    top_k_img_terms: List[str]
    top_k_img_scores: List[float]
    img_style_scores: np.ndarray
    aud_style_scores: np.ndarray
    predicted_params_normalized: np.ndarray
    predicted_params_dict: Dict[str, Dict[str, float]]
    output_audio: Optional[np.ndarray] = None


class DeltaV2APipeline:
    """Complete DeltaV2A inference pipeline."""

    def __init__(
        self,
        clip_embedder,
        visual_encoder: Optional[VisualEncoder],
        style_vocab: StyleVocabulary,
        controller: AudioController,
        clap_embedder,
        effect_names: List[str],
        sample_rate: int = 48000,
        top_k: int = 5,
        device: str = "cpu",
        use_siamese_visual_encoder: bool = True,
        use_clip_delta_fallback: bool = False,
    ):
        self.clip = clip_embedder
        self.visual_encoder = visual_encoder
        self.style_vocab = style_vocab
        self.controller = controller
        self.clap = clap_embedder
        self.effect_names = effect_names
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.device = device
        self.use_siamese_visual_encoder = use_siamese_visual_encoder
        self.use_clip_delta_fallback = use_clip_delta_fallback
        self.renderer = PedalboardRenderer(sample_rate=sample_rate)

        if self.visual_encoder is not None:
            self.visual_encoder.to(device).eval()
        self.controller.to(device).eval()

    @classmethod
    def load(
        cls,
        artifacts_dir: str,
        clip_embedder,
        clap_embedder,
        device: str = "cpu",
        use_siamese_visual_encoder: Optional[bool] = None,
    ) -> 'DeltaV2APipeline':
        """Load pipeline from saved artifacts directory."""
        import json
        from ..effects.pedalboard_effects import get_total_param_count

        artifacts = Path(artifacts_dir)

        with open(artifacts / 'pipeline_config.json', 'r') as f:
            config = json.load(f)

        effect_names = config['effect_names']
        top_k = config.get('top_k', 5)
        sample_rate = config.get('sample_rate', 48000)
        projection_dim = config.get('projection_dim', 768)
        artifact_mode = bool(config.get('use_siamese_visual_encoder', True))
        if use_siamese_visual_encoder is None:
            use_siamese_visual_encoder = artifact_mode
        elif bool(use_siamese_visual_encoder) != artifact_mode:
            print(
                "Warning: Config mode and artifact mode differ for visual_encoder.enabled; "
                f"using config value ({bool(use_siamese_visual_encoder)})."
            )

        # Load vocabularies
        style_vocab = StyleVocabulary()
        style_vocab.load(str(artifacts))

        # Load visual encoder (only for siamese mode)
        visual_encoder = None
        use_clip_delta_fallback = False
        if use_siamese_visual_encoder:
            visual_encoder = VisualEncoder(clip_embedder=clip_embedder, projection_dim=projection_dim)
            ve_ckpt_path = artifacts / 'visual_encoder.pt'
            use_clip_delta_fallback = True
            if ve_ckpt_path.exists():
                ve_ckpt = torch.load(ve_ckpt_path, map_location=device, weights_only=True)
                visual_encoder.load_state_dict(ve_ckpt['model_state_dict'])
                use_clip_delta_fallback = False
            else:
                print(f"Warning: visual encoder checkpoint not found: {ve_ckpt_path}. Using CLIP-delta fallback.")

        # Load controller
        ctrl_ckpt_path = artifacts / 'controller_best.pt'
        if not ctrl_ckpt_path.exists():
            legacy_path = artifacts / 'controller' / 'controller_best.pt'
            if legacy_path.exists():
                ctrl_ckpt_path = legacy_path
        if not ctrl_ckpt_path.exists():
            raise FileNotFoundError(
                f"Controller checkpoint not found at {artifacts / 'controller_best.pt'} "
                f"or {artifacts / 'controller' / 'controller_best.pt'}"
            )
        ctrl_ckpt = torch.load(ctrl_ckpt_path, map_location=device, weights_only=True)
        ctrl_model_cfg = ctrl_ckpt.get("model_config", {})
        total_params = int(ctrl_model_cfg.get("total_params", get_total_param_count(effect_names)))
        controller = AudioController(
            audio_embed_dim=int(ctrl_model_cfg.get("audio_embed_dim", 512)),
            style_vocab_size=int(ctrl_model_cfg.get("style_vocab_size", style_vocab.aud_vocab.size)),
            total_params=total_params,
            hidden_dims=ctrl_model_cfg.get("hidden_dims"),
            dropout=float(ctrl_model_cfg.get("dropout", 0.1)),
            use_activity_head=bool(ctrl_model_cfg.get("use_activity_head", False)),
            num_effects=int(ctrl_model_cfg.get("num_effects", len(effect_names))),
        )
        controller.load_state_dict(ctrl_ckpt['model_state_dict'])

        return cls(
            clip_embedder=clip_embedder,
            visual_encoder=visual_encoder,
            style_vocab=style_vocab,
            controller=controller,
            clap_embedder=clap_embedder,
            effect_names=effect_names,
            sample_rate=sample_rate,
            top_k=top_k,
            device=device,
            use_siamese_visual_encoder=bool(use_siamese_visual_encoder),
            use_clip_delta_fallback=use_clip_delta_fallback,
        )

    @torch.no_grad()
    def infer(
        self,
        original_image: torch.Tensor,
        edited_image: torch.Tensor,
        input_audio: np.ndarray,
    ) -> InferenceResult:
        """
        Run full inference pipeline.

        Args:
            original_image: (1, 3, H, W) in [0, 1]
            edited_image: (1, 3, H, W) in [0, 1]
            input_audio: (samples,) or (channels, samples)

        Returns:
            InferenceResult
        """
        original_image = original_image.to(self.device)
        edited_image = edited_image.to(self.device)

        clip_orig = self.clip.embed_images(original_image)
        clip_edit = self.clip.embed_images(edited_image)

        # Step 1: Visual Encoding / direct style delta
        if not self.use_siamese_visual_encoder:
            # Direct mode: use vocab-space delta Sim(I') - Sim(I)
            clip_orig = clip_orig / (clip_orig.norm(dim=-1, keepdim=True) + 1e-8)
            clip_edit = clip_edit / (clip_edit.norm(dim=-1, keepdim=True) + 1e-8)
            z_visual = clip_edit - clip_orig
            z_visual = z_visual / (z_visual.norm(dim=-1, keepdim=True) + 1e-8)

            z_visual_np = z_visual[0].cpu().numpy()
            img_embeddings = self.style_vocab.img_vocab.embeddings
            sim_orig = img_embeddings @ clip_orig[0].cpu().numpy()
            sim_edit = img_embeddings @ clip_edit[0].cpu().numpy()
            img_delta = (sim_edit - sim_orig).astype(np.float32)
            top_k_indices = np.argsort(img_delta)[-self.top_k:][::-1]
            top_k_terms = [self.style_vocab.img_vocab.terms[i] for i in top_k_indices]
            top_k_scores = [float(img_delta[i]) for i in top_k_indices]

            img_style_scores = np.maximum(img_delta, 0.0)
            total = float(img_style_scores.sum())
            if total > 0:
                img_style_scores = img_style_scores / total
            else:
                img_style_scores = np.ones_like(img_style_scores, dtype=np.float32) / max(
                    len(img_style_scores), 1
                )
        else:
            if self.use_clip_delta_fallback:
                z_visual = clip_edit - clip_orig
                z_visual = z_visual / (z_visual.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                if self.visual_encoder is None:
                    raise RuntimeError("visual_encoder is required when use_siamese_visual_encoder=True")
                z_visual = self.visual_encoder(original_image, edited_image)
            z_visual_np = z_visual[0].cpu().numpy()

            # Step 2: Style Retrieval
            img_embeddings = self.style_vocab.img_vocab.embeddings
            if z_visual_np.shape[0] != img_embeddings.shape[1]:
                raise ValueError(
                    f"Visual embedding dim ({z_visual_np.shape[0]}) must match IMG_VOCAB dim ({img_embeddings.shape[1]})."
                )
            img_sims = img_embeddings @ z_visual_np
            top_k_indices = np.argsort(img_sims)[-self.top_k:][::-1]

            img_style_scores = np.zeros(self.style_vocab.img_vocab.size, dtype=np.float32)
            for idx in top_k_indices:
                img_style_scores[idx] = max(img_sims[idx], 0.0)

            total = img_style_scores.sum()
            if total > 0:
                img_style_scores /= total

            top_k_terms = [self.style_vocab.img_vocab.terms[i] for i in top_k_indices]
            top_k_scores = [float(img_sims[i]) for i in top_k_indices]

        # Step 3: Cross-Modal Mapping (identity over shared axes)
        if self.style_vocab.img_vocab.size != self.style_vocab.aud_vocab.size:
            raise ValueError(
                "IMG/AUD vocab sizes differ. This pipeline assumes shared axes without "
                "a correspondence matrix."
            )
        aud_style_scores = img_style_scores.astype(np.float32, copy=True)

        # Step 4: Audio Controller
        audio_tensor = torch.from_numpy(input_audio).float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        clap_emb = self.clap.embed_audio(audio_tensor, self.sample_rate).to(self.device)

        style_tensor = torch.from_numpy(aud_style_scores).float().unsqueeze(0).to(self.device)
        params_pred = self.controller(clap_emb, style_tensor)
        params_normalized = params_pred[0].cpu().numpy()
        params_dict = denormalize_params(params_normalized, self.effect_names)

        # Step 5: Rendering
        output_audio = self.renderer.render(input_audio, params_dict)

        return InferenceResult(
            z_visual=z_visual_np,
            top_k_img_terms=top_k_terms,
            top_k_img_scores=top_k_scores,
            img_style_scores=img_style_scores,
            aud_style_scores=aud_style_scores,
            predicted_params_normalized=params_normalized,
            predicted_params_dict=params_dict,
            output_audio=output_audio,
        )

    def infer_from_paths(
        self,
        original_image_path: str,
        edited_image_path: str,
        input_audio_path: str,
        output_audio_path: str = None,
    ) -> InferenceResult:
        """Run inference from file paths."""
        from PIL import Image as PILImage
        import librosa
        import soundfile as sf
        import torchvision.transforms.functional as TF

        orig_img = PILImage.open(original_image_path).convert("RGB")
        edit_img = PILImage.open(edited_image_path).convert("RGB")

        # Keep [0,1] tensors; CLIP normalization happens inside CLIPEmbedder.embed_images().
        orig_tensor = TF.to_tensor(orig_img).unsqueeze(0)
        edit_tensor = TF.to_tensor(edit_img).unsqueeze(0)

        audio, _ = librosa.load(input_audio_path, sr=self.sample_rate, mono=True)
        result = self.infer(orig_tensor, edit_tensor, audio)

        if output_audio_path and result.output_audio is not None:
            out = result.output_audio
            if out.ndim == 1:
                sf.write(output_audio_path, out, self.sample_rate)
            else:
                sf.write(output_audio_path, out.T, self.sample_rate)
            print(f"Saved output audio to {output_audio_path}")

        return result
