import type { StyleScore } from '../types'

interface Props {
  scores: StyleScore[]
  topK: string[]
}

function formatTerm(term: string): string {
  return term.replace(/^a\s+/, '').replace(/\s+image$/, '')
}

export function StyleVocabChart({ scores, topK }: Props) {
  const sorted = [...scores].sort((a, b) => b.score - a.score)
  const maxScore = sorted[0]?.score ?? 1
  const topKSet = new Set(topK)

  return (
    <div>
      <p className="text-xs uppercase tracking-widest text-gray-400 mb-5">Image Style Distribution</p>
      <div className="space-y-2">
        {sorted.map(({ term, score }, i) => {
          const isTop = topKSet.has(term)
          const pct = maxScore > 0 ? (score / maxScore) * 100 : 0
          return (
            <div key={term} className="flex items-center gap-3">
              <span
                className={`text-xs w-28 flex-shrink-0 truncate ${
                  isTop ? 'text-gray-900 font-medium' : 'text-gray-400'
                }`}
                title={term}
              >
                {formatTerm(term)}
              </span>
              <div className="flex-1 bg-gray-100 h-1.5">
                <div
                  className={`h-full transition-all duration-300 ${
                    i < 3 ? 'bg-gray-900' : i < 6 ? 'bg-gray-500' : 'bg-gray-300'
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className={`text-xs font-mono w-12 text-right ${isTop ? 'text-gray-900' : 'text-gray-400'}`}>
                {score.toFixed(4)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
