import type { WandEffect, EffectSelection } from '../types'
import { WAND_EFFECTS, WAND_EFFECT_LABELS } from '../types'

interface Props {
  selections: EffectSelection[]
  onSelectionsChange: (selections: EffectSelection[]) => void
}

export function EffectPanel({ selections, onSelectionsChange }: Props) {
  const selMap = new Map(selections.map((s) => [s.name, s.intensity]))

  const toggle = (eff: WandEffect) => {
    if (selMap.has(eff)) {
      onSelectionsChange(selections.filter((s) => s.name !== eff))
    } else {
      onSelectionsChange([...selections, { name: eff, intensity: 0.6 }])
    }
  }

  const setIntensity = (eff: WandEffect, val: number) => {
    const clamped = Math.max(0, Math.min(1, isNaN(val) ? 0 : val))
    onSelectionsChange(selections.map((s) => (s.name === eff ? { ...s, intensity: clamped } : s)))
  }

  return (
    <div className="space-y-6">
      <p className="text-xs uppercase tracking-widest text-gray-400">Image Effects</p>

      <div className="flex flex-wrap gap-2">
        {WAND_EFFECTS.map((eff) => {
          const active = selMap.has(eff)
          return (
            <button
              key={eff}
              onClick={() => toggle(eff)}
              className={`px-3 py-1.5 text-sm border transition-colors ${
                active
                  ? 'border-gray-900 bg-gray-900 text-white'
                  : 'border-gray-300 text-gray-600 hover:border-gray-700 hover:text-gray-900'
              }`}
            >
              {WAND_EFFECT_LABELS[eff]}
            </button>
          )
        })}
      </div>

      {selections.length > 0 && (
        <div className="space-y-4 border-t border-gray-100 pt-4">
          {selections.map(({ name, intensity }) => (
            <div key={name} className="flex items-center gap-4">
              <span className="text-xs text-gray-500 w-28 flex-shrink-0">{WAND_EFFECT_LABELS[name]}</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={intensity}
                onChange={(e) => setIntensity(name, Number(e.target.value))}
                className="flex-1 h-px appearance-none bg-gray-200 cursor-pointer accent-gray-900"
              />
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={intensity.toFixed(2)}
                onChange={(e) => setIntensity(name, parseFloat(e.target.value))}
                className="w-14 text-xs font-mono text-gray-700 border-b border-gray-300 bg-transparent
                           text-right pb-0.5 focus:outline-none focus:border-gray-900
                           [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none
                           [&::-webkit-inner-spin-button]:appearance-none"
              />
            </div>
          ))}
        </div>
      )}

      {selections.length === 0 && (
        <p className="text-xs text-gray-400">Select one or more effects above</p>
      )}
    </div>
  )
}
