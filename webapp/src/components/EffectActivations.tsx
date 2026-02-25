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
    <div>
      <div className="flex items-baseline justify-between mb-5">
        <p className="text-xs uppercase tracking-widest text-gray-400">Audio Effect Activations</p>
        <span className="text-xs text-gray-400">{activeCount}/{activations.length} active</span>
      </div>

      <div className="space-y-3">
        {activations.map(({ name, active, probability, params }) => (
          <div
            key={name}
            className={`pl-3 border-l-2 py-0.5 ${active ? 'border-gray-900' : 'border-gray-200'}`}
          >
            <div className="flex items-center gap-3">
              <span className={`text-sm flex-1 ${active ? 'text-gray-900 font-medium' : 'text-gray-400'}`}>
                {formatEffectName(name)}
              </span>
              <div className="flex items-center gap-2">
                <div className="w-16 bg-gray-100 h-1">
                  <div
                    className={`h-full ${active ? 'bg-gray-900' : 'bg-gray-300'}`}
                    style={{ width: `${probability * 100}%` }}
                  />
                </div>
                <span className={`text-xs font-mono w-8 text-right ${active ? 'text-gray-700' : 'text-gray-400'}`}>
                  {Math.round(probability * 100)}%
                </span>
              </div>
            </div>

            {active && Object.keys(params).length > 0 && (
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
                {Object.entries(params).map(([k, v]) => (
                  <span key={k} className="text-xs text-gray-500">
                    {formatParamName(k)}&thinsp;{formatParamValue(k, v)}
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
