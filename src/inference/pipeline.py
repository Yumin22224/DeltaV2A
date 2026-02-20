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
from ..effects.pedalboard_effects import (
    EFFECT_CATALOG,
    PedalboardRenderer,
    denormalize_params,
    normalize_params,
)


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
    predicted_activity_probs: Optional[List[float]] = None
    predicted_activity_mask: Optional[List[bool]] = None
    activity_thresholds: Optional[List[float]] = None
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
        activity_thresholds: Optional[np.ndarray] = None,
        apply_activity_gating: bool = True,
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
        self.apply_activity_gating = bool(apply_activity_gating)
        self.renderer = PedalboardRenderer(sample_rate=sample_rate)
        self.activity_thresholds = activity_thresholds.astype(np.float32) if activity_thresholds is not None else None
        self._bypass_normalized = normalize_params({}, effect_names)
        self._effect_param_slices = []
        start = 0
        for effect_name in effect_names:
            width = EFFECT_CATALOG[effect_name].num_params
            self._effect_param_slices.append(slice(start, start + width))
            start += width

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

        thresholds_arr = None
        thresholds_path_candidates = [
            artifacts / "controller" / "activity_thresholds.json",
            artifacts / "controller_activity_thresholds.json",
        ]
        for cand in thresholds_path_candidates:
            if cand.exists():
                with open(cand, "r") as f:
                    payload = json.load(f)
                thresholds = payload.get("thresholds", {})
                thresholds_arr = np.array(
                    [float(thresholds.get(name, 0.5)) for name in effect_names],
                    dtype=np.float32,
                )
                break

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
            activity_thresholds=thresholds_arr,
            apply_activity_gating=True,
        )

    def _gate_params_by_activity(
        self,
        params_normalized: np.ndarray,
        activity_mask: np.ndarray,
    ) -> np.ndarray:
        if activity_mask.size == 0:
            return params_normalized
        gated = params_normalized.copy()
        for i, sl in enumerate(self._effect_param_slices):
            if i >= activity_mask.shape[0]:
                break
            if not bool(activity_mask[i]):
                gated[sl] = self._bypass_normalized[sl]
        return gated

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
        params_pred, activity_logits = self.controller.forward_with_activity(clap_emb, style_tensor)
        params_normalized = params_pred[0].cpu().numpy()
        pred_activity_probs: Optional[np.ndarray] = None
        pred_activity_mask: Optional[np.ndarray] = None
        threshold_arr: Optional[np.ndarray] = None
        if activity_logits is not None:
            pred_activity_probs = torch.sigmoid(activity_logits)[0].cpu().numpy()
            if self.activity_thresholds is not None and self.activity_thresholds.shape[0] == pred_activity_probs.shape[0]:
                threshold_arr = self.activity_thresholds
            else:
                threshold_arr = np.full(pred_activity_probs.shape[0], 0.5, dtype=np.float32)
            pred_activity_mask = (pred_activity_probs >= threshold_arr).astype(np.bool_)
            if self.apply_activity_gating:
                params_normalized = self._gate_params_by_activity(params_normalized, pred_activity_mask)
        params_dict = denormalize_params(params_normalized, self.effect_names)

        # Step 5: Rendering
        output_audio = self.renderer.render(input_audio, params_dict)

        # Volume normalization: match output RMS to input RMS.
        # Consistent with DB build where A' was normalized before CLAP encoding.
        rms_input = float(np.sqrt(np.mean(input_audio ** 2)))
        rms_output = float(np.sqrt(np.mean(output_audio ** 2)))
        if rms_output > 1e-8 and rms_input > 1e-8:
            output_audio = output_audio * (rms_input / rms_output)
            peak = float(np.abs(output_audio).max())
            if peak > 1.0:
                output_audio = output_audio / peak * 0.95

        return InferenceResult(
            z_visual=z_visual_np,
            top_k_img_terms=top_k_terms,
            top_k_img_scores=top_k_scores,
            img_style_scores=img_style_scores,
            aud_style_scores=aud_style_scores,
            predicted_params_normalized=params_normalized,
            predicted_params_dict=params_dict,
            predicted_activity_probs=(
                pred_activity_probs.tolist() if pred_activity_probs is not None else None
            ),
            predicted_activity_mask=(
                pred_activity_mask.tolist() if pred_activity_mask is not None else None
            ),
            activity_thresholds=(threshold_arr.tolist() if threshold_arr is not None else None),
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
