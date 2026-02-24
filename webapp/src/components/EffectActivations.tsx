import type { EffectActivation } from '../types'

interface Props {
  activations: EffectActivation[]
}

function formatEffectName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatParamName(key: string): string {
  const map: Record<string, string> = {
    cutoff_hz: 'Cutoff Hz',
    bit_depth: 'Bit Depth',
    room_size: 'Room',
    damping: 'Damping',
    wet_level: 'Wet',
    dry_level: 'Dry',
    drive_db: 'Drive dB',
    rate: 'Rate',
    delay_seconds: 'Delay s',
    feedback: 'Feedback',
    mix: 'Mix',
  }
  return map[key] ?? key.replace(/_/g, ' ')
}

function formatParamValue(key: string, val: number): string {
  if (key === 'cutoff_hz') return `${Math.round(val)} Hz`
  if (key === 'drive_db') return `${val.toFixed(1)} dB`
  if (key === 'delay_seconds') return `${val.toFixed(2)}s`
  if (key === 'rate') return `${val.toFixed(2)}x`
  return val.toFixed(2)
}

export function EffectActivations({ activations }: Props) {
  const activeCount = activations.filter((a) => a.active).length

  return (
    <div className="bg-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-300">Audio Effect Activations</h3>
        <span className="text-xs text-gray-500">
          {activeCount}/{activations.length} active
        </span>
      </div>
      <div className="space-y-2">
        {activations.map(({ name, active, probability, params }) => (
          <div
            key={name}
            className={`rounded-lg px-4 py-3 border transition-colors ${
              active
                ? 'border-indigo-600/60 bg-indigo-950/40'
                : 'border-gray-700 bg-gray-750'
            }`}
          >
            <div className="flex items-center gap-3">
              {/* Active indicator */}
              <div
                className={`w-2 h-2 rounded-full flex-shrink-0 ${
                  active ? 'bg-indigo-400' : 'bg-gray-600'
                }`}
              />

              {/* Effect name */}
              <span
                className={`text-sm font-medium flex-1 ${
                  active ? 'text-indigo-200' : 'text-gray-500'
                }`}
              >
                {formatEffectName(name)}
              </span>

              {/* Probability bar */}
              <div className="flex items-center gap-2">
                <div className="w-16 bg-gray-700 rounded-full h-1.5">
                  <div
                    className={`h-full rounded-full ${active ? 'bg-indigo-500' : 'bg-gray-600'}`}
                    style={{ width: `${probability * 100}%` }}
                  />
                </div>
                <span className={`text-xs font-mono w-8 text-right ${active ? 'text-indigo-300' : 'text-gray-600'}`}>
                  {Math.round(probability * 100)}%
                </span>
              </div>
            </div>

            {/* Parameters (only for active effects) */}
            {active && Object.keys(params).length > 0 && (
              <div className="mt-2 pl-5 flex flex-wrap gap-2">
                {Object.entries(params).map(([k, v]) => (
                  <span key={k} className="text-xs bg-gray-700/60 text-gray-300 rounded px-2 py-0.5">
                    {formatParamName(k)}: {formatParamValue(k, v)}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
