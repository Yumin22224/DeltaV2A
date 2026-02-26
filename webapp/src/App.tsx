import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchPreview, runInfer, checkHealth, getApiUrl, setApiUrl } from './api'
import type { InferResult, EffectSelection } from './types'
import { ImageUpload } from './components/ImageUpload'
import { EffectPanel } from './components/EffectPanel'
import { EffectPreview } from './components/EffectPreview'
import { AudioUpload } from './components/AudioUpload'
import { StyleVocabChart } from './components/StyleVocabChart'
import { EffectActivations } from './components/EffectActivations'
import { AudioPlayer } from './components/AudioPlayer'

export default function App() {
  const [originalFile, setOriginalFile] = useState<File | null>(null)
  const [effectSelections, setEffectSelections] = useState<EffectSelection[]>([])
  const [audioFile, setAudioFile] = useState<File | null>(null)

  const [previewB64, setPreviewB64] = useState<string | null>(null)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const previewAbortRef = useRef<AbortController | null>(null)

  const [isInferring, setIsInferring] = useState(false)
  const [result, setResult] = useState<InferResult | null>(null)
  const [inferError, setInferError] = useState<string | null>(null)

  const [serverReady, setServerReady] = useState<boolean | null>(null)
  const [backendUrl, setBackendUrl] = useState(() => getApiUrl())
  const [urlEditing, setUrlEditing] = useState(false)
  const urlInputRef = useRef<HTMLInputElement>(null)

  const commitUrl = useCallback((url: string) => {
    setApiUrl(url)
    setBackendUrl(url)
    setUrlEditing(false)
    checkHealth().then(setServerReady)
  }, [])

  useEffect(() => {
    checkHealth().then(setServerReady)
    const interval = setInterval(() => checkHealth().then(setServerReady), 10_000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!originalFile || effectSelections.length === 0) {
      setPreviewB64(null)
      return
    }
    setIsPreviewLoading(true)
    const timer = setTimeout(async () => {
      previewAbortRef.current?.abort()
      const ctrl = new AbortController()
      previewAbortRef.current = ctrl
      try {
        const b64 = await fetchPreview(originalFile, effectSelections)
        if (!ctrl.signal.aborted) setPreviewB64(b64)
      } catch (err) {
        if (!ctrl.signal.aborted) console.warn('Preview error:', err)
      } finally {
        if (!ctrl.signal.aborted) setIsPreviewLoading(false)
      }
    }, 350)
    return () => { clearTimeout(timer); setIsPreviewLoading(false) }
  }, [originalFile, effectSelections])

  const handleInfer = useCallback(async () => {
    if (!originalFile || !audioFile || effectSelections.length === 0) return
    setIsInferring(true)
    setInferError(null)
    setResult(null)
    try {
      const data = await runInfer(originalFile, effectSelections, audioFile)
      setResult(data)
      setPreviewB64(data.preview_image)
    } catch (err) {
      setInferError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsInferring(false)
    }
  }, [originalFile, audioFile, effectSelections])

  const canInfer =
    !!originalFile && !!audioFile && effectSelections.length > 0 && !isInferring && serverReady === true

  return (
    <div className="min-h-screen bg-white text-gray-900 font-sans">
      {/* Header */}
      <header className="px-8 py-5 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold tracking-tight">DeltaV2A Demo</h1>
          <p className="text-xs text-gray-400 mt-0.5">Image delta → style label → audio effect prediction</p>
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
            serverReady === null ? 'bg-yellow-400 animate-pulse' : serverReady ? 'bg-green-500' : 'bg-red-500'
          }`} />
          <span>{serverReady === null ? 'Connecting…' : serverReady ? 'Server ready' : 'Server offline'}</span>
          <span className="text-gray-200">|</span>
          {urlEditing ? (
            <input
              ref={urlInputRef}
              type="text"
              defaultValue={backendUrl}
              placeholder="http://localhost:8000"
              autoFocus
              onBlur={e => commitUrl(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') commitUrl((e.target as HTMLInputElement).value)
                if (e.key === 'Escape') { setUrlEditing(false); setBackendUrl(getApiUrl()) }
              }}
              className="font-mono text-xs text-gray-600 border border-gray-300 px-2 py-0.5 w-56 focus:outline-none focus:border-gray-500"
            />
          ) : (
            <button
              onClick={() => setUrlEditing(true)}
              className="font-mono text-gray-400 hover:text-gray-600 truncate max-w-56 text-left"
              title="Click to change backend URL"
            >
              {backendUrl || 'set backend URL'}
            </button>
          )}
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-8 py-10 space-y-12">
        {/* Image row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-400 mb-3">Original</p>
            <ImageUpload value={originalFile} onChange={setOriginalFile} />
          </div>
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-400 mb-3">
              Preview
              {isPreviewLoading && <span className="ml-2 normal-case tracking-normal animate-pulse">— updating</span>}
            </p>
            <EffectPreview originalFile={originalFile} previewB64={previewB64} isLoading={isPreviewLoading} />
          </div>
        </div>

        {/* Effect controls */}
        <div className="border-t border-gray-100 pt-10">
          <EffectPanel
            selections={effectSelections}
            onSelectionsChange={(sels) => { setEffectSelections(sels); setResult(null) }}
          />
        </div>

        {/* Audio */}
        <div className="border-t border-gray-100 pt-10">
          <p className="text-xs uppercase tracking-widest text-gray-400 mb-3">Input Audio</p>
          <AudioUpload value={audioFile} onChange={(f) => { setAudioFile(f); setResult(null) }} />
        </div>

        {/* Run */}
        <div className="border-t border-gray-100 pt-10 flex items-center gap-6">
          <button
            onClick={handleInfer}
            disabled={!canInfer}
            className={`text-sm px-6 py-2.5 transition-colors ${
              canInfer
                ? 'bg-gray-900 text-white hover:bg-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            }`}
          >
            {isInferring ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Running…
              </span>
            ) : '▶  Run Inference'}
          </button>

          <span className="text-sm text-gray-400">
            {!originalFile && 'Upload an image to get started'}
            {originalFile && effectSelections.length === 0 && 'Select an effect above'}
            {originalFile && effectSelections.length > 0 && !audioFile && 'Upload audio to continue'}
            {serverReady === false && <span className="text-red-500">Backend offline — run <code className="font-mono">scripts/serve.py</code></span>}
          </span>
        </div>

        {inferError && (
          <p className="text-sm text-red-600 border-l-2 border-red-400 pl-3">{inferError}</p>
        )}

        {/* Results */}
        {result && (
          <div className="border-t border-gray-100 pt-10 space-y-12">
            <p className="text-xs uppercase tracking-widest text-gray-400">Results</p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              <StyleVocabChart scores={result.style_scores} topK={result.top_k_terms} />
              <EffectActivations activations={result.effect_activations} />
            </div>

            <AudioPlayer inputFile={audioFile} outputB64={result.output_audio_b64} sampleRate={result.sample_rate} />
          </div>
        )}
      </main>
    </div>
  )
}
