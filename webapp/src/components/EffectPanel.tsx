import type { WandEffect } from '../types'
import { WAND_EFFECTS, WAND_EFFECT_LABELS } from '../types'

interface Props {
  selectedEffect: WandEffect
  intensity: number
  onEffectChange: (effect: WandEffect) => void
  onIntensityChange: (value: number) => void
}

export function EffectPanel({ selectedEffect, intensity, onEffectChange, onIntensityChange }: Props) {
  return (
    <div className="bg-gray-800 rounded-xl p-5 space-y-5">
      <div>
        <p className="text-xs uppercase tracking-widest text-gray-500 mb-3">Image Effect</p>
        <div className="flex flex-wrap gap-2">
          {WAND_EFFECTS.map((eff) => (
            <button
              key={eff}
              onClick={() => onEffectChange(eff)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                eff === selectedEffect
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {WAND_EFFECT_LABELS[eff]}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs uppercase tracking-widest text-gray-500">Intensity</p>
          <span className="text-sm font-mono text-indigo-300">{intensity.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={intensity}
          onChange={(e) => onIntensityChange(Number(e.target.value))}
          className="w-full h-2 rounded-full appearance-none cursor-pointer
                     bg-gray-700 accent-indigo-500"
        />
        <div className="flex justify-between text-xs text-gray-600 mt-1">
          <span>0.0</span>
          <span>1.0</span>
        </div>
      </div>
    </div>
  )
}
