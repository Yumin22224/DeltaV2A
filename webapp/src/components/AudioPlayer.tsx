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

function TrackPlayer({
  label,
  src,
  downloadName,
}: {
  label: string
  src: string
  downloadName?: string
}) {
  return (
    <div className="flex-1 bg-gray-700/40 rounded-xl p-4 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-gray-500">{label}</span>
        {downloadName && (
          <a
            href={src}
            download={downloadName}
            className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download
          </a>
        )}
      </div>
      <audio
        controls
        src={src}
        className="w-full h-8"
        style={{ colorScheme: 'dark' }}
      />
    </div>
  )
}

export function AudioPlayer({ inputFile, outputB64, sampleRate: _sampleRate }: Props) {
  const inputUrl = useMemo(
    () => (inputFile ? URL.createObjectURL(inputFile) : null),
    [inputFile],
  )
  const outputUrl = useAudioUrl(outputB64)

  // Revoke output URL on unmount / change
  const outputUrlRef = useRef(outputUrl)
  useEffect(() => {
    const prev = outputUrlRef.current
    outputUrlRef.current = outputUrl
    return () => {
      if (prev && prev !== outputUrl) URL.revokeObjectURL(prev)
    }
  }, [outputUrl])

  if (!inputUrl && !outputUrl) return null

  return (
    <div className="bg-gray-800 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-gray-300 mb-4">Audio</h3>
      <div className="flex flex-col sm:flex-row gap-4">
        {inputUrl && (
          <TrackPlayer label="Input" src={inputUrl} />
        )}
        {outputUrl && (
          <TrackPlayer
            label="Output (processed)"
            src={outputUrl}
            downloadName="deltav2a_output.wav"
          />
        )}
      </div>
    </div>
  )
}
