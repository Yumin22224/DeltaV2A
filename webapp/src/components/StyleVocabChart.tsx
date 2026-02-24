import type { StyleScore } from '../types'

interface Props {
  scores: StyleScore[]
  topK: string[]
}

// Strip "a " prefix and " image" suffix for concise display
function formatTerm(term: string): string {
  return term.replace(/^a\s+/, '').replace(/\s+image$/, '')
}

export function StyleVocabChart({ scores, topK }: Props) {
  const sorted = [...scores].sort((a, b) => b.score - a.score)
  const maxScore = sorted[0]?.score ?? 1
  const topKSet = new Set(topK)

  return (
    <div className="bg-gray-800 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-gray-300 mb-4">Image Style Distribution</h3>
      <div className="space-y-2">
        {sorted.map(({ term, score }, i) => {
          const isTop = topKSet.has(term)
          const pct = maxScore > 0 ? (score / maxScore) * 100 : 0
          return (
            <div key={term} className="flex items-center gap-3">
              <span
                className={`text-xs w-28 flex-shrink-0 truncate ${
                  isTop ? 'text-indigo-300 font-medium' : 'text-gray-500'
                }`}
                title={term}
              >
                {formatTerm(term)}
              </span>
              <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    i < 3 ? 'bg-indigo-500' : i < 6 ? 'bg-indigo-700' : 'bg-gray-600'
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className={`text-xs font-mono w-12 text-right ${isTop ? 'text-indigo-300' : 'text-gray-600'}`}>
                {score.toFixed(4)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
