export interface StyleScore {
  term: string
  score: number
}

export interface EffectActivation {
  name: string
  active: boolean
  probability: number
  params: Record<string, number>
}

export interface InferResult {
  preview_image: string       // base64 PNG of effect-applied image
  output_audio_b64: string    // base64 WAV
  sample_rate: number
  style_scores: StyleScore[]  // all 24 vocab terms
  top_k_terms: string[]
  top_k_scores: number[]
  effect_activations: EffectActivation[]
}

export type WandEffect =
  | 'adaptive_blur'
  | 'motion_blur'
  | 'adaptive_sharpen'
  | 'add_noise'
  | 'spread'
  | 'sepia_tone'
  | 'solarize'

export const WAND_EFFECTS: WandEffect[] = [
  'adaptive_blur',
  'motion_blur',
  'adaptive_sharpen',
  'add_noise',
  'spread',
  'sepia_tone',
  'solarize',
]

export const WAND_EFFECT_LABELS: Record<WandEffect, string> = {
  adaptive_blur: 'Adaptive Blur',
  motion_blur: 'Motion Blur',
  adaptive_sharpen: 'Sharpen',
  add_noise: 'Add Noise',
  spread: 'Spread',
  sepia_tone: 'Sepia',
  solarize: 'Solarize',
}
