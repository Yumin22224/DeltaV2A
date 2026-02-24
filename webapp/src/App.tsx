import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchPreview, runInfer, checkHealth } from './api'
import type { InferResult, WandEffect } from './types'
import { ImageUpload } from './components/ImageUpload'
import { EffectPanel } from './components/EffectPanel'
import { EffectPreview } from './components/EffectPreview'
import { AudioUpload } from './components/AudioUpload'
import { StyleVocabChart } from './components/StyleVocabChart'
import { EffectActivations } from './components/EffectActivations'
import { AudioPlayer } from './components/AudioPlayer'

export default function App() {
  // --- Input state ---
  const [originalFile, setOriginalFile] = useState<File | null>(null)
  const [selectedEffect, setSelectedEffect] = useState<WandEffect>('sepia_tone')
  const [intensity, setIntensity] = useState(0.6)
  const [audioFile, setAudioFile] = useState<File | null>(null)

  // --- Preview state ---
  const [previewB64, setPreviewB64] = useState<string | null>(null)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const previewAbortRef = useRef<AbortController | null>(null)

  // --- Inference state ---
  const [isInferring, setIsInferring] = useState(false)
  const [result, setResult] = useState<InferResult | null>(null)
  const [inferError, setInferError] = useState<string | null>(null)

  // --- Server health ---
  const [serverReady, setServerReady] = useState<boolean | null>(null)

  // Check server health on mount
  useEffect(() => {
    checkHealth().then(setServerReady)
    const interval = setInterval(() => {
      checkHealth().then(setServerReady)
    }, 10_000)
    return () => clearInterval(interval)
  }, [])

  // Debounced preview: fires 350ms after originalFile / selectedEffect / intensity changes
  useEffect(() => {
    if (!originalFile) {
      setPreviewB64(null)
      return
    }

    setIsPreviewLoading(true)
    const timer = setTimeout(async () => {
      // Cancel any in-flight preview request
      previewAbortRef.current?.abort()
      const ctrl = new AbortController()
      previewAbortRef.current = ctrl

      try {
        const b64 = await fetchPreview(originalFile, selectedEffect, intensity)
        if (!ctrl.signal.aborted) setPreviewB64(b64)
      } catch (err) {
        if (!ctrl.signal.aborted) console.warn('Preview error:', err)
      } finally {
        if (!ctrl.signal.aborted) setIsPreviewLoading(false)
      }
    }, 350)

    return () => {
      clearTimeout(timer)
      setIsPreviewLoading(false)
    }
  }, [originalFile, selectedEffect, intensity])

  // Reset preview when image is cleared
  useEffect(() => {
    if (!originalFile) setPreviewB64(null)
  }, [originalFile])

  const handleInfer = useCallback(async () => {
    if (!originalFile || !audioFile) return
    setIsInferring(true)
    setInferError(null)
    setResult(null)

    try {
      const data = await runInfer(originalFile, selectedEffect, intensity, audioFile)
      setResult(data)
      // Also update preview from infer result (avoids duplicate wand call)
      setPreviewB64(data.preview_image)
    } catch (err) {
      setInferError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsInferring(false)
    }
  }, [originalFile, audioFile, selectedEffect, intensity])

  const canInfer = !!originalFile && !!audioFile && !isInferring && serverReady === true

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Header */}
      <header className="border-b border-gray-700 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight">
            Delta<span className="text-indigo-400">V2A</span> Demo
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Image delta → style label → audio effect prediction
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              serverReady === null
                ? 'bg-yellow-500 animate-pulse'
                : serverReady
                ? 'bg-green-500'
                : 'bg-red-500'
            }`}
          />
          <span className="text-gray-400">
            {serverReady === null ? 'Connecting…' : serverReady ? 'Server ready' : 'Server offline'}
          </span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-8">
        {/* --- Image row --- */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500 mb-2">Original Image</p>
            <ImageUpload value={originalFile} onChange={setOriginalFile} />
          </div>
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500 mb-2">
              Effect Preview
              {isPreviewLoading && (
                <span className="ml-2 text-indigo-400 normal-case tracking-normal animate-pulse">
                  updating…
                </span>
              )}
            </p>
            <EffectPreview
              originalFile={originalFile}
              previewB64={previewB64}
              isLoading={isPreviewLoading}
            />
          </div>
        </div>

        {/* --- Effect controls --- */}
        <EffectPanel
          selectedEffect={selectedEffect}
          intensity={intensity}
          onEffectChange={(eff) => {
            setSelectedEffect(eff)
            setResult(null)
          }}
          onIntensityChange={(val) => {
            setIntensity(val)
            setResult(null)
          }}
        />

        {/* --- Audio upload --- */}
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-500 mb-2">Input Audio</p>
          <AudioUpload value={audioFile} onChange={(f) => { setAudioFile(f); setResult(null) }} />
        </div>

        {/* --- Run button --- */}
        <div className="flex items-center gap-4">
          <button
            onClick={handleInfer}
            disabled={!canInfer}
            className={`px-6 py-3 rounded-lg font-semibold text-sm transition-colors ${
              canInfer
                ? 'bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
          >
            {isInferring ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Running…
              </span>
            ) : (
              '▶  Run Inference'
            )}
          </button>
          {!originalFile && (
            <span className="text-sm text-gray-500">Upload an image to get started</span>
          )}
          {originalFile && !audioFile && (
            <span className="text-sm text-gray-500">Upload an audio file to continue</span>
          )}
          {serverReady === false && (
            <span className="text-sm text-red-400">Backend offline — start <code>scripts/serve.py</code></span>
          )}
        </div>

        {inferError && (
          <div className="rounded-lg bg-red-950 border border-red-700 px-4 py-3 text-sm text-red-300">
            <strong>Error:</strong> {inferError}
          </div>
        )}

        {/* --- Results --- */}
        {result && (
          <div className="space-y-6 pt-2 border-t border-gray-700">
            <h2 className="text-lg font-semibold text-gray-200">Results</h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <StyleVocabChart scores={result.style_scores} topK={result.top_k_terms} />
              <EffectActivations activations={result.effect_activations} />
            </div>

            <AudioPlayer
              inputFile={audioFile}
              outputB64={result.output_audio_b64}
              sampleRate={result.sample_rate}
            />
          </div>
        )}
      </main>
    </div>
  )
}
