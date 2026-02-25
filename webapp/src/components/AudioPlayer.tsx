import { useEffect, useMemo, useRef } from 'react'

interface Props {
  inputFile: File | null
  outputB64: string
  sampleRate: number
}

function useAudioUrl(b64: string): string | null {
  return useMemo(() => {
    if (!b64) return null
    const bytes = atob(b64)
    const arr = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i)
    return URL.createObjectURL(new Blob([arr], { type: 'audio/wav' }))
  }, [b64])
}

function TrackPlayer({ label, src, downloadName }: { label: string; src: string; downloadName?: string }) {
  return (
    <div className="flex-1 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-gray-400">{label}</span>
        {downloadName && (
          <a
            href={src}
            download={downloadName}
            className="text-xs text-gray-500 hover:text-gray-900 underline underline-offset-2"
          >
            ↓ Download
          </a>
        )}
      </div>
      <audio controls src={src} className="w-full h-8" style={{ colorScheme: 'light' }} />
    </div>
  )
}

export function AudioPlayer({ inputFile, outputB64, sampleRate: _sampleRate }: Props) {
  const inputUrl = useMemo(
    () => (inputFile ? URL.createObjectURL(inputFile) : null),
    [inputFile],
  )
  const outputUrl = useAudioUrl(outputB64)

  const outputUrlRef = useRef(outputUrl)
  useEffect(() => {
    const prev = outputUrlRef.current
    outputUrlRef.current = outputUrl
    return () => { if (prev && prev !== outputUrl) URL.revokeObjectURL(prev) }
  }, [outputUrl])

  if (!inputUrl && !outputUrl) return null

  return (
    <div>
      <p className="text-xs uppercase tracking-widest text-gray-400 mb-4">Audio</p>
      <div className="flex flex-col sm:flex-row gap-8">
        {inputUrl && <TrackPlayer label="Input" src={inputUrl} />}
        {outputUrl && <TrackPlayer label="Output (processed)" src={outputUrl} downloadName="deltav2a_output.wav" />}
      </div>
    </div>
  )
}
