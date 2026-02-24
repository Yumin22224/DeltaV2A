interface Props {
  originalFile: File | null
  previewB64: string | null
  isLoading: boolean
}

export function EffectPreview({ originalFile, previewB64, isLoading }: Props) {
  return (
    <div
      className="rounded-xl border-2 border-gray-700 bg-gray-800 overflow-hidden relative"
      style={{ minHeight: '220px' }}
    >
      {!originalFile && (
        <div className="flex items-center justify-center h-full py-12">
          <p className="text-sm text-gray-600">Upload an image to see preview</p>
        </div>
      )}

      {originalFile && !previewB64 && !isLoading && (
        <div className="flex items-center justify-center h-full py-12">
          <p className="text-sm text-gray-600">Select an effect to preview</p>
        </div>
      )}

      {previewB64 && (
        <img
          src={`data:image/png;base64,${previewB64}`}
          alt="effect preview"
          className="w-full h-full object-cover"
          style={{ minHeight: '220px', maxHeight: '320px' }}
        />
      )}

      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/60">
          <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
    </div>
  )
}
