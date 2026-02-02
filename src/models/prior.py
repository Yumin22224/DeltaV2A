"""
Prior construction module for C_anchor estimation

Implements System Specification v2 Section 3:
- Hard Prior: Type-II rules with Classical CV and Librosa
- Soft Prior: ImageBind-based multimodal embeddings
- Combined Prior: Saliency-weighted combination

References all 12 rules (r1-r12) from literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2


class VisualFeatureExtractor:
    """
    Classical CV-based visual feature extraction

    Extracts features for Type-II rules:
    - brightness, lightness, motion_speed, vertical_position
    - sharp_edges, rounded_smooth, size_large, saturation
    - warm_colors, visual_roughness, blur, sharpness_focus
    """

    def __init__(self):
        self.feature_names = [
            'brightness', 'lightness', 'motion_speed', 'vertical_position',
            'sharp_edges', 'rounded_smooth', 'size_large', 'saturation',
            'warm_colors', 'visual_roughness', 'blur', 'sharpness_focus'
        ]

    def extract(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all visual features

        Args:
            images: (B, 3, H, W) RGB images [0, 1]

        Returns:
            features: Dict of feature_name -> (B,) scalar values
        """
        B = images.shape[0]
        device = images.device

        # Convert to numpy for OpenCV
        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)

        features = {}

        for b in range(B):
            img = images_np[b].transpose(1, 2, 0)  # (H, W, 3)

            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # r1: brightness (V channel mean)
            if b == 0:
                features['brightness'] = []
            features['brightness'].append(hsv[:, :, 2].mean() / 255.0)

            # r2: lightness (L in LAB)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            if b == 0:
                features['lightness'] = []
            features['lightness'].append(lab[:, :, 0].mean() / 255.0)

            # r3: motion_speed (placeholder - needs temporal info)
            if b == 0:
                features['motion_speed'] = []
            features['motion_speed'].append(0.0)  # Requires video

            # r4: vertical_position (center of mass)
            if b == 0:
                features['vertical_position'] = []
            y_coords = np.arange(gray.shape[0])
            y_center = np.average(y_coords, weights=gray.sum(axis=1) + 1e-8)
            features['vertical_position'].append(y_center / gray.shape[0])

            # r5: sharp_edges (Sobel magnitude)
            if b == 0:
                features['sharp_edges'] = []
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(sobelx**2 + sobely**2)
            features['sharp_edges'].append(edge_mag.mean() / 255.0)

            # r6: rounded_smooth (inverse of edges)
            if b == 0:
                features['rounded_smooth'] = []
            features['rounded_smooth'].append(1.0 - features['sharp_edges'][-1])

            # r7: size_large (area of bright regions)
            if b == 0:
                features['size_large'] = []
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            area_ratio = binary.sum() / (gray.shape[0] * gray.shape[1] * 255)
            features['size_large'].append(area_ratio)

            # r8: saturation (S channel mean)
            if b == 0:
                features['saturation'] = []
            features['saturation'].append(hsv[:, :, 1].mean() / 255.0)

            # r9: warm_colors (red-yellow dominance)
            if b == 0:
                features['warm_colors'] = []
            hue = hsv[:, :, 0]
            warm_mask = ((hue < 30) | (hue > 150)).astype(float)
            features['warm_colors'].append(warm_mask.mean())

            # r10: visual_roughness (texture variance)
            if b == 0:
                features['visual_roughness'] = []
            features['visual_roughness'].append(np.var(gray) / (255.0**2))

            # r11: blur (Laplacian variance - lower = more blur)
            if b == 0:
                features['blur'] = []
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = laplacian.var()
            features['blur'].append(1.0 - min(lap_var / 1000.0, 1.0))

            # r12: sharpness_focus (inverse of blur)
            if b == 0:
                features['sharpness_focus'] = []
            features['sharpness_focus'].append(1.0 - features['blur'][-1])

        # Convert lists to tensors
        for key in features:
            features[key] = torch.tensor(features[key], device=device, dtype=torch.float32)

        return features


class AudioDescriptorExtractor:
    """
    Librosa-based audio descriptor extraction

    Extracts descriptors for Type-II rules:
    - pitch_height, loudness, tempo, attack_sharpness
    - sound_smoothness, musical_intensity, sound_roughness
    - reverb_spaciousness, dry_close_sound
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.descriptor_names = [
            'pitch_height', 'loudness', 'tempo', 'attack_sharpness',
            'sound_smoothness', 'musical_intensity', 'sound_roughness',
            'reverb_spaciousness', 'dry_close_sound'
        ]

    def extract(
        self,
        audios: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all audio descriptors

        Args:
            audios: (B, 1, T, F) mel spectrograms (fallback if waveforms not provided)
            waveforms: (B, T_samples) raw waveforms at 16kHz (preferred)

        Returns:
            descriptors: Dict of descriptor_name -> (B,) scalar values in [0, 1]
        """
        try:
            import librosa
        except ImportError:
            print("Warning: librosa not available, using placeholder descriptors")
            return self._placeholder_descriptors(audios)

        B = audios.shape[0]
        device = audios.device

        descriptors = {}

        for b in range(B):
            if waveforms is not None:
                # Use raw waveform directly (much more accurate)
                y = waveforms[b].cpu().numpy()
                if y.ndim > 1:
                    y = y.squeeze(0)
            else:
                # Fallback: reconstruct waveform from mel (rough approximation)
                mel = audios[b, 0].cpu().numpy()  # (T, F)
                y = librosa.feature.inverse.mel_to_audio(
                    mel.T, sr=self.sr, n_fft=1024, hop_length=160
                )

            # r1, r4: pitch_height (mean F0)
            if b == 0:
                descriptors['pitch_height'] = []
            try:
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                       fmax=librosa.note_to_hz('C7'), sr=self.sr)
                f0_mean = np.nanmean(f0) if len(f0) > 0 else 0.0
                descriptors['pitch_height'].append(np.clip(f0_mean / 1000.0, 0.0, 1.0))
            except:
                descriptors['pitch_height'].append(0.0)

            # r2, r7: loudness (RMS)
            if b == 0:
                descriptors['loudness'] = []
            rms = librosa.feature.rms(y=y)[0]
            descriptors['loudness'].append(np.clip(rms.mean(), 0.0, 1.0))

            # r3, r9: tempo
            if b == 0:
                descriptors['tempo'] = []
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
                descriptors['tempo'].append(np.clip(tempo / 200.0, 0.0, 1.0))
            except:
                descriptors['tempo'].append(0.5)

            # r5: attack_sharpness (onset strength)
            if b == 0:
                descriptors['attack_sharpness'] = []
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            descriptors['attack_sharpness'].append(np.clip(onset_env.max() / 10.0, 0.0, 1.0))

            # r6: sound_smoothness (inverse of spectral flux)
            if b == 0:
                descriptors['sound_smoothness'] = []
            spec = np.abs(librosa.stft(y))
            flux = np.mean(np.diff(spec, axis=1)**2)
            descriptors['sound_smoothness'].append(np.clip(1.0 - min(flux / 100.0, 1.0), 0.0, 1.0))

            # r8: musical_intensity (spectral energy)
            if b == 0:
                descriptors['musical_intensity'] = []
            spectral_energy = np.sum(spec**2) / spec.size
            descriptors['musical_intensity'].append(np.clip(spectral_energy / 1000.0, 0.0, 1.0))

            # r10: sound_roughness (spectral irregularity)
            if b == 0:
                descriptors['sound_roughness'] = []
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
            descriptors['sound_roughness'].append(np.clip(spec_contrast.std() / 10.0, 0.0, 1.0))

            # r11: reverb_spaciousness (spectral flatness - higher = more reverb)
            if b == 0:
                descriptors['reverb_spaciousness'] = []
            flatness = librosa.feature.spectral_flatness(y=y)
            descriptors['reverb_spaciousness'].append(np.clip(flatness.mean(), 0.0, 1.0))

            # r12: dry_close_sound (inverse of reverb)
            if b == 0:
                descriptors['dry_close_sound'] = []
            descriptors['dry_close_sound'].append(np.clip(1.0 - descriptors['reverb_spaciousness'][-1], 0.0, 1.0))

        # Convert to tensors
        for key in descriptors:
            descriptors[key] = torch.tensor(descriptors[key], device=device, dtype=torch.float32)

        return descriptors

    def _placeholder_descriptors(self, audios: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Placeholder when librosa not available"""
        B = audios.shape[0]
        device = audios.device

        descriptors = {}
        for name in self.descriptor_names:
            descriptors[name] = torch.rand(B, device=device) * 0.5 + 0.25

        return descriptors


class HardPrior:
    """
    Hard Prior based on Type-II rules (r1-r12)

    Implements literature-based crossmodal correspondences
    with Classical CV and Librosa feature extraction
    """

    def __init__(self, rules: List[Dict], num_heads: int = 6):
        """
        Args:
            rules: List of rule dictionaries from config
            num_heads: Number of control heads (6)
        """
        self.rules = rules
        self.num_heads = num_heads
        self.head_names = ["rhythm", "harmony", "energy", "timbre", "space", "texture"]

        # Feature extractors
        self.visual_extractor = VisualFeatureExtractor()
        self.audio_extractor = AudioDescriptorExtractor()

        # Build weight matrix W_hard
        self.W_hard = self._build_weight_matrix()

    def _build_weight_matrix(self) -> torch.Tensor:
        """
        Build weight matrix W_hard
        W_hard[h] = sum of weights for rules targeting head h
        """
        W = torch.zeros(self.num_heads)

        for rule in self.rules:
            target_head = rule['target_head']
            if target_head in self.head_names:
                h_idx = self.head_names.index(target_head)
                W[h_idx] += rule['weight']

        return W

    def extract_visual_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract visual features using Classical CV"""
        return self.visual_extractor.extract(images)

    def extract_audio_descriptors(self, audios: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract audio descriptors using Librosa"""
        return self.audio_extractor.extract(audios)

    def compute_saliency(
        self,
        images: torch.Tensor,
        N_v: int = 256,
    ) -> torch.Tensor:
        """
        Compute saliency map for visual tokens

        Based on Spec v2.1 Section 3.3:
        - Edge detection (Sobel)
        - Texture (Gabor filter)
        - Spatial pooling to N_v tokens (grid pooling)

        Args:
            images: (B, 3, H, W) RGB images
            N_v: Number of visual tokens

        Returns:
            saliency: (B, N_v) saliency scores
        """
        B, C, H, W = images.shape
        device = images.device

        # Convert to numpy
        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)

        saliency_maps = []

        for b in range(B):
            img = images_np[b].transpose(1, 2, 0)  # (H, W, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # 1. Edge detection (Sobel)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = edges / 255.0

            # 2. Texture (Gabor filter bank)
            texture = self._apply_gabor_filters(gray)

            # 3. Combine (50% edge + 50% texture)
            saliency_map = 0.5 * edges + 0.5 * texture
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

            # 4. Spatial pooling to N_v tokens (grid pooling)
            saliency_tokens = self._spatial_pool_to_tokens(saliency_map, N_v)

            saliency_maps.append(saliency_tokens)

        saliency = torch.tensor(saliency_maps, device=device, dtype=torch.float32)
        return saliency

    def _apply_gabor_filters(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Gabor filter bank for texture analysis

        Args:
            gray: (H, W) grayscale image

        Returns:
            texture_map: (H, W) texture response
        """
        # Gabor filter parameters
        ksize = 21
        sigma = 5.0
        lambd = 10.0
        gamma = 0.5
        psi = 0

        # Multiple orientations for robustness
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        responses = []
        for theta in orientations:
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta,
                lambd,
                gamma,
                psi,
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(np.abs(filtered))

        # Combine responses (max across orientations)
        texture_map = np.max(responses, axis=0)
        texture_map = texture_map / (texture_map.max() + 1e-8)

        return texture_map

    def _spatial_pool_to_tokens(self, saliency_map: np.ndarray, N_v: int) -> List[float]:
        """
        Spatial pooling: pool saliency map to N_v tokens using grid

        Args:
            saliency_map: (H, W) saliency map
            N_v: Number of tokens

        Returns:
            saliency_tokens: List of N_v saliency values
        """
        H, W = saliency_map.shape

        # Grid dimensions
        grid_size = int(np.sqrt(N_v))
        h_step = H // grid_size
        w_step = W // grid_size

        saliency_tokens = []

        for i in range(grid_size):
            for j in range(grid_size):
                # Extract patch
                h_start = i * h_step
                h_end = (i + 1) * h_step if i < grid_size - 1 else H
                w_start = j * w_step
                w_end = (j + 1) * w_step if j < grid_size - 1 else W

                patch = saliency_map[h_start:h_end, w_start:w_end]

                # Mean pooling
                saliency_tokens.append(patch.mean())

        # Pad if needed (when grid_size^2 < N_v)
        while len(saliency_tokens) < N_v:
            saliency_tokens.append(0.0)

        # Truncate if needed (when grid_size^2 > N_v)
        saliency_tokens = saliency_tokens[:N_v]

        return saliency_tokens

    def get_weight_matrix(self) -> torch.Tensor:
        """Get static W_hard weight matrix (all rules, no activation check)"""
        return self.W_hard

    @staticmethod
    def compute_activation_score(
        f_v: float,
        f_a: float,
        correlation: float,
        tau: float = 0.5,
        alpha: float = 3.0,
    ) -> float:
        """
        Compute continuous activation score for a rule per Spec v2.4 Section 3.1.

        Step 1: Feature Centering (High-side only)
            f_v_tilde = max(0, f_v - tau)
            f_a_tilde = max(0, f_a - tau)
        Step 2: Saturation (tanh normalization)
            f_v_hat = tanh(alpha * f_v_tilde)
            f_a_hat = tanh(alpha * f_a_tilde)
        Step 3: Correlation Score
            s_r = f_v_hat * f_a_hat       if positive correlation
            s_r = -f_v_hat * f_a_hat      if negative correlation

        Args:
            f_v: Visual feature value in [0, 1]
            f_a: Audio descriptor value in [0, 1]
            correlation: Expected correlation (+1 or -1)
            tau: Neutral threshold (0.5)
            alpha: Sensitivity parameter (3.0)

        Returns:
            s_r: Activation score (>= 0 after max(0, .))
        """
        # Step 1: High-side centering
        f_v_tilde = max(0.0, f_v - tau)
        f_a_tilde = max(0.0, f_a - tau)

        # Step 2: Saturation
        f_v_hat = np.tanh(alpha * f_v_tilde)
        f_a_hat = np.tanh(alpha * f_a_tilde)

        # Step 3: Correlation score
        if correlation >= 0:  # positive correlation
            s_r = f_v_hat * f_a_hat
        else:  # negative correlation
            s_r = -f_v_hat * f_a_hat

        # Non-negative (negative scores are not meaningful)
        return max(0.0, s_r)

    def compute_activated_weights(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
        tau: float = 0.5,
        alpha: float = 3.0,
        delta: float = 0.05,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Compute W_hard with continuous activation scores based on (I, A) pair.

        Per Spec v2.4 Section 3.1:
        W_hard[h] = Σ_{r: h_target(r)=h} w_r · max(0, s_r(I, A))

        where s_r is a continuous activation score using high-side-only
        centering and tanh saturation. A rule is considered activated
        when s_r > delta (minimum activation threshold).

        Args:
            images: (B, 3, H, W) RGB images (ImageNet-normalized or [0,1])
            audios: (B, 1, T, F) mel spectrograms
            waveforms: (B, T_samples) raw waveforms (optional, preferred)
            tau: Neutral threshold (0.5, features normalized to [0,1])
            alpha: Sensitivity parameter for tanh saturation (3.0)
            delta: Minimum activation threshold (0.05)

        Returns:
            W_hard_activated: (B, 6) per-sample activated weight vector
            activation_details: List[Dict] per-sample rule activation info
        """
        B = images.shape[0]
        device = images.device

        # Undo ImageNet normalization if needed (check if values are outside [0,1])
        images_01 = self._ensure_01_range(images)

        # Extract features
        visual_features = self.visual_extractor.extract(images_01)
        audio_descriptors = self.audio_extractor.extract(audios, waveforms=waveforms)

        # Rule-to-feature mapping
        rule_feature_map = {
            'r1_brightness_pitch': ('brightness', 'pitch_height'),
            'r2_lightness_loudness': ('lightness', 'loudness'),
            'r3_motion_tempo': ('motion_speed', 'tempo'),
            'r4_vertical_pitch': ('vertical_position', 'pitch_height'),
            'r5_angular_attack': ('sharp_edges', 'attack_sharpness'),
            'r6_smooth_sound': ('rounded_smooth', 'sound_smoothness'),
            'r7_size_loudness': ('size_large', 'loudness'),
            'r8_saturation_intensity': ('saturation', 'musical_intensity'),
            'r9_warm_colors_tempo': ('warm_colors', 'tempo'),
            'r10_roughness_roughness': ('visual_roughness', 'sound_roughness'),
            'r11_blur_reverb': ('blur', 'reverb_spaciousness'),
            'r12_sharpness_dry': ('sharpness_focus', 'dry_close_sound'),
        }

        W_hard_batch = torch.zeros(B, self.num_heads, device=device)
        activation_details = []

        for b in range(B):
            sample_details = {'rules': []}

            for rule in self.rules:
                rule_name = rule['name']
                target_head = rule['target_head']
                weight = rule['weight']
                correlation = rule.get('correlation', 1.0)

                h_idx = self.head_names.index(target_head) if target_head in self.head_names else -1
                if h_idx < 0:
                    continue

                # Get feature names for this rule
                if rule_name in rule_feature_map:
                    f_v_name, f_a_name = rule_feature_map[rule_name]
                else:
                    # Try to match by visual_feature/audio_feature from config
                    f_v_name = rule.get('visual_feature', None)
                    f_a_name = rule.get('audio_feature', None)

                if f_v_name is None or f_a_name is None:
                    continue

                # Get feature values
                f_v = visual_features.get(f_v_name, torch.zeros(B, device=device))[b].item()
                f_a = audio_descriptors.get(f_a_name, torch.zeros(B, device=device))[b].item()

                # Continuous activation score per Spec v2.4
                s_r = self.compute_activation_score(
                    f_v, f_a, correlation, tau=tau, alpha=alpha,
                )
                activated = bool(s_r > delta)

                # Continuous weighting: w_r * s_r (not binary w_r)
                contribution = weight * s_r if activated else 0.0
                if activated:
                    W_hard_batch[b, h_idx] += contribution

                sample_details['rules'].append({
                    'name': rule_name,
                    'target_head': target_head,
                    'visual_feature': f_v_name,
                    'audio_feature': f_a_name,
                    'f_v_value': round(f_v, 4),
                    'f_a_value': round(f_a, 4),
                    'correlation': correlation,
                    'weight': weight,
                    'activation_score': round(s_r, 4),
                    'activated': activated,
                    'contribution': round(contribution, 4),
                })

            activation_details.append(sample_details)

        return W_hard_batch, activation_details

    def _ensure_01_range(self, images: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization to get [0,1] range if needed."""
        # Check if values are outside [0,1] (ImageNet-normalized)
        if images.min() < -0.5:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            images = images * imagenet_std + imagenet_mean
            images = images.clamp(0, 1)
        return images


class SoftPrior(nn.Module):
    """
    Soft Prior based on ImageBind

    Implements Spec v2 Section 3.2:
    - Extract v_tokens and a_tokens using ImageBind
    - Compute similarity matrix
    - Head pooling with Q_h queries
    - Coupling C_soft = Sim @ a_heads^T
    """

    def __init__(
        self,
        model_name: str = "imagebind",
        num_heads: int = 6,
        head_dim: int = 512,
        N_v: int = 256,
        N_a: int = 256,
        freeze: bool = True,
    ):
        """
        Args:
            model_name: ImageBind model name
            num_heads: Number of control heads (6)
            head_dim: Token embedding dimension (D=512)
            N_v: Number of visual tokens
            N_a: Number of audio tokens
            freeze: Freeze pretrained weights
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.N_v = N_v
        self.N_a = N_a

        # Load ImageBind (or CLIP as fallback)
        self.vision_model, self.audio_model = self._load_imagebind(model_name)

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            if self.audio_model is not self.vision_model:
                for param in self.audio_model.parameters():
                    param.requires_grad = False

        # Head queries (initialized with CLIP text embeddings)
        self.head_queries = nn.Parameter(
            torch.randn(num_heads, head_dim),
            requires_grad=False,
        )

        # Projection layers for ImageBind embeddings (1280 -> head_dim, 1024 -> head_dim)
        # Initialized with fixed seed for reproducibility across runs
        if getattr(self, '_is_imagebind', False):
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(42)
            self._vision_proj = nn.Linear(1280, head_dim, bias=False)
            nn.init.xavier_uniform_(self._vision_proj.weight)
            self._vision_proj.requires_grad_(False)
            self._audio_proj_ib = nn.Linear(1024, head_dim, bias=False)
            nn.init.xavier_uniform_(self._audio_proj_ib.weight)
            self._audio_proj_ib.requires_grad_(False)
            torch.random.set_rng_state(rng_state)

    def _load_imagebind(self, model_name: str):
        """Load ImageBind or fallback to CLIP using centralized loader"""
        from src.utils.model_loaders import load_imagebind_or_clip

        model, self._is_imagebind = load_imagebind_or_clip(freeze=True)

        if self._is_imagebind:
            # ImageBind handles both vision and audio natively
            return model, model
        else:
            # CLIP fallback: vision only, audio handled separately in extract_tokens
            return model, model

    def initialize_head_queries(self, text_prompts: Dict[str, str]):
        """
        Initialize Q_h with CLIP text embeddings
        """
        try:
            import open_clip
            from src.utils.model_loaders import load_clip

            model = load_clip(freeze=True)
            tokenizer = open_clip.get_tokenizer('ViT-L-14')

            with torch.no_grad():
                for h, (head_name, prompt) in enumerate(text_prompts.items()):
                    if h < self.num_heads:
                        tokens = tokenizer([prompt]).to(next(model.parameters()).device)
                        text_features = model.encode_text(tokens)

                        # Ensure correct dimension
                        if text_features.shape[-1] != self.head_dim:
                            # Project if needed
                            text_features = F.adaptive_avg_pool1d(
                                text_features.unsqueeze(1), self.head_dim
                            ).squeeze(1)

                        self.head_queries[h] = text_features.squeeze().cpu()

            print("Initialized head queries with CLIP text embeddings")

        except Exception as e:
            print(f"Failed to initialize head queries: {e}")

    def extract_tokens(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract v_tokens and a_tokens

        Uses ImageBind for both modalities when available:
        - Vision: extracts 256 spatial patch tokens from ViT trunk
        - Audio: converts waveforms to Kaldi fbank, gets global embedding

        Falls back to CLIP for vision only (audio uses mel-based projection).

        Args:
            images: (B, 3, H, W) images (ImageNet-normalized)
            audios: (B, 1, T, F) mel spectrograms (used for CLIP fallback)
            waveforms: (B, T_samples) raw waveforms at 16kHz (used for ImageBind)

        Returns:
            v_tokens: (B, N_v, D)
            a_tokens: (B, N_a, D)
        """
        B = images.shape[0]

        with torch.no_grad():
            if isinstance(self.vision_model, nn.Identity):
                v_tokens = torch.randn(B, self.N_v, self.head_dim, device=images.device)
                a_tokens = torch.randn(B, self.N_a, self.head_dim, device=audios.device)

            elif getattr(self, '_is_imagebind', False):
                # ImageBind: spatial patch tokens + waveform-based audio
                v_tokens = self._extract_imagebind_vision(images)
                a_tokens = self._extract_imagebind_audio(waveforms, audios)
            else:
                # CLIP fallback: vision tokens from CLIP, audio from mel projection
                v_tokens = self._extract_clip_vision(images)
                a_tokens = self._extract_audio_from_mel(audios)

        return v_tokens, a_tokens

    def _extract_imagebind_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract 256 spatial patch tokens from ImageBind's ViT trunk.

        Bypasses the CLS-selection head to get actual spatial tokens.
        ImageBind ViT-H with (2,14,14) patches on 224x224 = 256 spatial tokens.
        """
        from src.utils.model_loaders import (
            imagebind_preprocess_vision,
            imagebind_extract_patch_tokens,
        )

        B = images.shape[0]

        try:
            # Preprocess: undo ImageNet norm, resize to 224, apply ImageBind norm
            images_ib = imagebind_preprocess_vision(images)

            # Extract 256 spatial patch tokens from trunk (skip CLS head)
            patch_tokens = imagebind_extract_patch_tokens(
                self.vision_model, images_ib, target_n_tokens=self.N_v
            )  # (B, N_v, embed_dim)  embed_dim=1280 for ViT-H

            # Project 1280-dim patch tokens to head_dim
            v_tokens = self._vision_proj(patch_tokens.to(self._vision_proj.weight.device))

        except Exception as e:
            print(f"ImageBind vision extraction failed: {e}")
            v_tokens = torch.randn(B, self.N_v, self.head_dim, device=images.device)

        return v_tokens

    def _extract_imagebind_audio(
        self,
        waveforms: Optional[torch.Tensor],
        audios_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract audio tokens using ImageBind.

        When waveforms are available: converts to Kaldi fbank (128 mel, 204 frames)
        and runs ImageBind's audio encoder for a (B, 1024) global embedding,
        then expands to N_a tokens.

        When only mel spectrograms are available: falls back to mel-based projection.

        Args:
            waveforms: (B, T_samples) raw waveforms at 16kHz, or None
            audios_mel: (B, 1, T, F) mel spectrograms (fallback)
        """
        from src.utils.model_loaders import (
            waveform_to_imagebind_audio,
            imagebind_extract_audio_embedding,
        )

        B = audios_mel.shape[0]
        device = audios_mel.device

        if waveforms is None:
            # No waveforms available, use mel-based fallback
            return self._extract_audio_from_mel(audios_mel)

        try:
            # Convert each waveform to ImageBind format and batch
            audio_inputs = []
            for b in range(B):
                wf = waveforms[b]  # (T_samples,) or (1, T_samples)
                ib_audio = waveform_to_imagebind_audio(wf)  # (num_clips, 1, 128, 204)
                audio_inputs.append(ib_audio)

            audio_batch = torch.stack(audio_inputs, dim=0).to(device)
            # (B, num_clips, 1, 128, 204)

            # Get global audio embedding
            a_features = imagebind_extract_audio_embedding(
                self.vision_model, audio_batch
            )  # (B, 1024)

            # Project 1024-dim embedding to head_dim, then expand to N_a tokens
            a_projected = self._audio_proj_ib(a_features)  # (B, head_dim)
            a_tokens = a_projected.unsqueeze(1).expand(-1, self.N_a, -1)
            # (B, N_a, head_dim)

        except Exception as e:
            print(f"ImageBind audio extraction failed: {e}")
            a_tokens = self._extract_audio_from_mel(audios_mel)

        return a_tokens

    def _extract_clip_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Extract vision tokens using CLIP (fallback)"""
        B = images.shape[0]

        try:
            v_features = self.vision_model.encode_image(images)

            if v_features.dim() == 2:
                v_tokens = v_features.unsqueeze(1).expand(-1, self.N_v, -1)
            else:
                v_tokens = v_features

            # Project to head_dim if needed
            if v_tokens.shape[-1] != self.head_dim:
                v_tokens = F.adaptive_avg_pool1d(
                    v_tokens.transpose(1, 2), self.head_dim
                ).transpose(1, 2)

        except Exception:
            v_tokens = torch.randn(B, self.N_v, self.head_dim, device=images.device)

        return v_tokens

    def _extract_audio_from_mel(self, audios: torch.Tensor) -> torch.Tensor:
        """
        Extract audio tokens from mel spectrogram when ImageBind is unavailable.

        Uses a learned linear projection from mel features to token space,
        initialized lazily on first use.
        """
        B = audios.shape[0]

        # Lazy initialization of mel projection layer
        if not hasattr(self, '_mel_proj'):
            mel_dim = audios.shape[-1]  # F (n_mels)
            self._mel_proj = nn.Linear(mel_dim, self.head_dim).to(audios.device)
            # Initialize with small random weights
            nn.init.xavier_uniform_(self._mel_proj.weight)
            self._mel_proj.requires_grad_(False)

        # audios: (B, 1, T, F)
        mel = audios.squeeze(1)  # (B, T, F)

        # Project each time step to head_dim
        a_tokens_full = self._mel_proj(mel)  # (B, T, head_dim)

        # Subsample/pool to N_a tokens
        T = a_tokens_full.shape[1]
        if T >= self.N_a:
            # Uniform subsample
            indices = torch.linspace(0, T - 1, self.N_a).long().to(audios.device)
            a_tokens = a_tokens_full[:, indices, :]
        else:
            # Pad by repeating
            repeats = (self.N_a + T - 1) // T
            a_tokens = a_tokens_full.repeat(1, repeats, 1)[:, :self.N_a, :]

        return a_tokens

    def compute_coupling(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute C_soft following Spec v2 Section 3.2

        Steps:
        1. Sim = cosine_similarity(v_tokens, a_tokens)
        2. a_heads = Q^T @ a_tokens^T
        3. C_soft = Sim @ a_heads^T
        4. Normalize with softmax
        """
        B, N_v, D = v_tokens.shape
        N_a = a_tokens.shape[1]

        # Step 1: Similarity matrix
        v_norm = F.normalize(v_tokens, dim=-1)
        a_norm = F.normalize(a_tokens, dim=-1)
        sim = torch.bmm(v_norm, a_norm.transpose(1, 2))  # (B, N_v, N_a)

        # Step 2: Head pooling
        Q_norm = F.normalize(self.head_queries, dim=-1).to(v_tokens.device)  # (6, D)
        a_norm_t = a_norm.transpose(1, 2)  # (B, D, N_a)
        # Use einsum for clear batched matmul: Q_norm (6, D) x a_norm_t (B, D, N_a) -> (B, 6, N_a)
        a_heads = torch.einsum('hd,bdn->bhn', Q_norm, a_norm_t)  # (B, 6, N_a)

        # Step 3: Coupling
        C_soft = torch.bmm(sim, a_heads.transpose(1, 2))  # (B, N_v, N_a) x (B, N_a, 6) = (B, N_v, 6)

        # Step 4: Normalize
        C_soft = F.softmax(C_soft, dim=-1)

        return C_soft

    def forward(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute C_soft"""
        v_tokens, a_tokens = self.extract_tokens(images, audios, waveforms=waveforms)
        C_soft = self.compute_coupling(v_tokens, a_tokens)
        return C_soft


class PriorEstimator(nn.Module):
    """
    Combined Prior Estimator

    Implements Spec v2 Section 3.3:
    C_prior[i, :] = (1-alpha)·C_soft[i, :] + alpha·s_i·W_hard^T
    C_anchor = C_prior + δC
    """

    def __init__(
        self,
        hard_prior: HardPrior,
        soft_prior: SoftPrior,
        alpha: float = 0.3,
        entropy_min: float = 0.5,
        sparsity_max: float = 5.0,
    ):
        """
        Args:
            hard_prior: HardPrior instance
            soft_prior: SoftPrior instance
            alpha: Balance (0.3 in spec)
            entropy_min: H(C_prior) > 0.5
            sparsity_max: ||C_prior||_1 < 5.0
        """
        super().__init__()

        self.hard_prior = hard_prior
        self.soft_prior = soft_prior
        self.alpha = alpha
        self.entropy_min = entropy_min
        self.sparsity_max = sparsity_max

    def forward(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
        delta_C: Optional[torch.Tensor] = None,
        waveforms: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        """
        Estimate C_anchor = C_prior + δC

        Per Spec v2.3: W_hard is computed per-sample based on rule activation
        using both visual features and audio descriptors.

        Args:
            images: (B, 3, H, W)
            audios: (B, 1, T, F)
            delta_C: (B, N_v, 6) optional perturbation
            waveforms: (B, T_samples) raw waveforms at 16kHz (optional)
            return_details: If True, return (C_anchor, activation_details)

        Returns:
            C_anchor: (B, N_v, 6)
            activation_details: (optional) List[Dict] per-sample rule activation info
        """
        # Compute C_soft
        C_soft = self.soft_prior(images, audios, waveforms=waveforms)
        B, N_v, num_heads = C_soft.shape

        # Compute saliency
        saliency = self.hard_prior.compute_saliency(images, N_v=N_v)  # (B, N_v)

        # Compute activated W_hard per-sample (Spec v2.3 Section 3.1)
        W_hard_activated, activation_details = self.hard_prior.compute_activated_weights(
            images, audios, waveforms=waveforms
        )  # (B, 6), List[Dict]

        # Combine: C_prior[i, :] = (1-α)·C_soft[i, :] + α·s_i·W_hard^T
        saliency_expanded = saliency.unsqueeze(-1)  # (B, N_v, 1)
        W_hard_expanded = W_hard_activated.unsqueeze(1)  # (B, 1, 6)

        C_hard = saliency_expanded * W_hard_expanded  # (B, N_v, 6)
        C_hard = C_hard / (C_hard.sum(dim=-1, keepdim=True) + 1e-8)

        C_prior = (1 - self.alpha) * C_soft + self.alpha * C_hard

        # Apply δC if provided
        if delta_C is not None:
            C_anchor = C_prior + delta_C
        else:
            C_anchor = C_prior

        # Apply constraints
        C_anchor = self._apply_constraints(C_anchor)

        if return_details:
            return C_anchor, activation_details
        return C_anchor

    def _apply_constraints(self, C: torch.Tensor) -> torch.Tensor:
        """
        Apply entropy and sparsity constraints from Spec v2

        H(C_prior) > H_min = 0.5
        ||C_prior||_1 < S_max = 5.0
        """
        # Non-negative
        C = torch.clamp(C, min=0)

        # Normalize
        C = C / (C.sum(dim=-1, keepdim=True) + 1e-8)

        # TODO: Implement entropy and sparsity projection
        # For now, simple normalization

        return C

    def estimate_prior_only(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate C_prior (δC=0) for Stage 2-A/B"""
        result = self.forward(images, audios, delta_C=None, waveforms=waveforms)
        assert isinstance(result, torch.Tensor)
        return result
