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

    def extract(self, audios: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all audio descriptors

        Args:
            audios: (B, 1, T, F) mel spectrograms

        Returns:
            descriptors: Dict of descriptor_name -> (B,) scalar values
        """
        try:
            import librosa
        except ImportError:
            print("Warning: librosa not available, using placeholder descriptors")
            return self._placeholder_descriptors(audios)

        B = audios.shape[0]
        device = audios.device

        # Convert mel to waveform (simplified - in practice use vocoder)
        # For now, use inverse mel as approximation
        descriptors = {}

        for b in range(B):
            mel = audios[b, 0].cpu().numpy()  # (T, F)

            # Approximate waveform from mel (very rough)
            # In practice, should use Griffin-Lim or neural vocoder
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
                descriptors['pitch_height'].append(f0_mean / 1000.0)
            except:
                descriptors['pitch_height'].append(0.0)

            # r2, r7: loudness (RMS)
            if b == 0:
                descriptors['loudness'] = []
            rms = librosa.feature.rms(y=y)[0]
            descriptors['loudness'].append(rms.mean())

            # r3, r9: tempo
            if b == 0:
                descriptors['tempo'] = []
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
                descriptors['tempo'].append(tempo / 200.0)  # Normalize
            except:
                descriptors['tempo'].append(0.5)

            # r5: attack_sharpness (onset strength)
            if b == 0:
                descriptors['attack_sharpness'] = []
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            descriptors['attack_sharpness'].append(onset_env.max() / 10.0)

            # r6: sound_smoothness (inverse of spectral flux)
            if b == 0:
                descriptors['sound_smoothness'] = []
            spec = np.abs(librosa.stft(y))
            flux = np.mean(np.diff(spec, axis=1)**2)
            descriptors['sound_smoothness'].append(1.0 - min(flux / 100.0, 1.0))

            # r8: musical_intensity (spectral energy)
            if b == 0:
                descriptors['musical_intensity'] = []
            spectral_energy = np.sum(spec**2) / spec.size
            descriptors['musical_intensity'].append(min(spectral_energy / 1000.0, 1.0))

            # r10: sound_roughness (spectral irregularity)
            if b == 0:
                descriptors['sound_roughness'] = []
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
            descriptors['sound_roughness'].append(spec_contrast.std() / 10.0)

            # r11: reverb_spaciousness (spectral flatness - higher = more reverb)
            if b == 0:
                descriptors['reverb_spaciousness'] = []
            flatness = librosa.feature.spectral_flatness(y=y)
            descriptors['reverb_spaciousness'].append(flatness.mean())

            # r12: dry_close_sound (inverse of reverb)
            if b == 0:
                descriptors['dry_close_sound'] = []
            descriptors['dry_close_sound'].append(1.0 - descriptors['reverb_spaciousness'][-1])

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
        """Get W_hard weight matrix"""
        return self.W_hard


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

    def _load_imagebind(self, model_name: str):
        """Load ImageBind or fallback to CLIP"""
        try:
            # Try to load actual ImageBind
            import imagebind
            from imagebind.models import imagebind_model

            model = imagebind_model.imagebind_huge(pretrained=True)
            print("Loaded ImageBind successfully")
            return model, model

        except ImportError:
            # Fallback to CLIP
            print("ImageBind not available, using CLIP as fallback")
            try:
                import open_clip

                vision_model, _, _ = open_clip.create_model_and_transforms(
                    'ViT-L-14', pretrained='openai'
                )

                # Use same model for audio (placeholder)
                return vision_model, vision_model

            except ImportError:
                print("Warning: Neither ImageBind nor CLIP available")
                return nn.Identity(), nn.Identity()

    def initialize_head_queries(self, text_prompts: Dict[str, str]):
        """
        Initialize Q_h with CLIP text embeddings
        """
        try:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='openai'
            )
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract v_tokens and a_tokens

        Args:
            images: (B, 3, H, W)
            audios: (B, 1, T, F) mel spectrograms

        Returns:
            v_tokens: (B, N_v, D)
            a_tokens: (B, N_a, D)
        """
        B = images.shape[0]

        with torch.no_grad():
            # Extract visual tokens
            if isinstance(self.vision_model, nn.Identity):
                v_tokens = torch.randn(B, self.N_v, self.head_dim, device=images.device)
            else:
                try:
                    v_features = self.vision_model.encode_image(images)

                    # Expand to tokens (simplified - actual ImageBind returns patches)
                    if v_features.dim() == 2:
                        v_tokens = v_features.unsqueeze(1).expand(-1, self.N_v, -1)
                    else:
                        v_tokens = v_features
                except:
                    v_tokens = torch.randn(B, self.N_v, self.head_dim, device=images.device)

            # Extract audio tokens (placeholder - actual implementation needs audio encoder)
            a_tokens = torch.randn(B, self.N_a, self.head_dim, device=audios.device)

        return v_tokens, a_tokens

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
        a_heads = torch.matmul(Q_norm, a_norm_t)  # (6, B, N_a)
        a_heads = a_heads.permute(1, 0, 2)  # (B, 6, N_a)

        # Step 3: Coupling
        C_soft = torch.bmm(sim, a_heads.transpose(1, 2))  # (B, N_v, 6)

        # Step 4: Normalize
        C_soft = F.softmax(C_soft, dim=-1)

        return C_soft

    def forward(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
    ) -> torch.Tensor:
        """Compute C_soft"""
        v_tokens, a_tokens = self.extract_tokens(images, audios)
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
    ) -> torch.Tensor:
        """
        Estimate C_anchor = C_prior + δC

        Args:
            images: (B, 3, H, W)
            audios: (B, 1, T, F)
            delta_C: (B, N_v, 6) optional perturbation

        Returns:
            C_anchor: (B, N_v, 6)
        """
        # Compute C_soft
        C_soft = self.soft_prior(images, audios)
        B, N_v, num_heads = C_soft.shape

        # Compute saliency
        saliency = self.hard_prior.compute_saliency(images, N_v=N_v)  # (B, N_v)

        # Get W_hard
        W_hard = self.hard_prior.get_weight_matrix().to(images.device)  # (6,)

        # Combine: C_prior[i, :] = (1-α)·C_soft[i, :] + α·s_i·W_hard^T
        saliency_expanded = saliency.unsqueeze(-1)  # (B, N_v, 1)
        W_hard_expanded = W_hard.unsqueeze(0).unsqueeze(0)  # (1, 1, 6)

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
    ) -> torch.Tensor:
        """Estimate C_prior (δC=0) for Stage 2-A/B"""
        return self.forward(images, audios, delta_C=None)
