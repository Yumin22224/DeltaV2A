import { useCallback, useRef } from 'react'

interface Props {
  value: File | null
  onChange: (file: File | null) => void
}

export function ImageUpload({ value, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith('image/')) onChange(file)
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

  const previewUrl = value ? URL.createObjectURL(value) : null

  return (
    <div
      className="relative group rounded-xl border-2 border-dashed border-gray-600 hover:border-indigo-500
                 transition-colors cursor-pointer bg-gray-800 overflow-hidden"
      style={{ minHeight: '220px' }}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleChange}
      />

      {previewUrl ? (
        <>
          <img
            src={previewUrl}
            alt="original"
            className="w-full h-full object-cover"
            style={{ minHeight: '220px', maxHeight: '320px' }}
            onLoad={() => URL.revokeObjectURL(previewUrl)}
          />
          <button
            className="absolute top-2 right-2 bg-gray-900/70 hover:bg-gray-900 text-gray-300
                       rounded-full w-7 h-7 flex items-center justify-center text-xs z-10"
            onClick={(e) => { e.stopPropagation(); onChange(null) }}
            title="Remove image"
          >
            ✕
          </button>
          <div className="absolute bottom-0 left-0 right-0 bg-gray-900/60 px-3 py-1.5 text-xs text-gray-300 truncate">
            {value?.name}
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center justify-center h-full py-12 gap-3">
          <svg className="w-10 h-10 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01
                 M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-sm text-gray-500">Drop image here or <span className="text-indigo-400">click to upload</span></p>
          <p className="text-xs text-gray-600">JPG, PNG, WEBP</p>
        </div>
      )}
    </div>
  )
}
