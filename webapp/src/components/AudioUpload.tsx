import { useCallback, useRef } from 'react'

interface Props {
  value: File | null
  onChange: (file: File | null) => void
}

export function AudioUpload({ value, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith('audio/')) onChange(file)
    },
    [onChange],
  )

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0] ?? null
      onChange(file)
    },
    [onChange],
  )

  return (
    <div
      className={`flex items-center gap-4 rounded-xl border-2 border-dashed px-5 py-4 cursor-pointer
                  transition-colors ${
                    value
                      ? 'border-indigo-600/50 bg-indigo-950/30'
                      : 'border-gray-600 hover:border-indigo-500 bg-gray-800'
                  }`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleChange}
      />

      <svg
        className={`w-8 h-8 flex-shrink-0 ${value ? 'text-indigo-400' : 'text-gray-500'}`}
        fill="none" stroke="currentColor" viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
      </svg>

      <div className="flex-1 min-w-0">
        {value ? (
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium text-gray-200 truncate">{value.name}</p>
              <p className="text-xs text-gray-500">{(value.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
            <button
              className="text-gray-500 hover:text-gray-300 px-2 py-1 text-xs"
              onClick={(e) => { e.stopPropagation(); onChange(null) }}
            >
              ✕
            </button>
          </div>
        ) : (
          <div>
            <p className="text-sm text-gray-400">
              Drop audio here or <span className="text-indigo-400">click to upload</span>
            </p>
            <p className="text-xs text-gray-600 mt-0.5">WAV, MP3, FLAC</p>
          </div>
        )}
      </div>
    </div>
  )
}
