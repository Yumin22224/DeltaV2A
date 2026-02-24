import type { InferResult } from './types'

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

export async function fetchPreview(
  image: File,
  effect: string,
  intensity: number,
): Promise<string> {
  const form = new FormData()
  form.append('image', image)
  form.append('effect', effect)
  form.append('intensity', String(intensity))

  const res = await fetch(`${API_URL}/api/preview-effect`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Preview failed (${res.status}): ${detail}`)
  }
  const data = await res.json()
  return data.image_base64 as string
}

export async function runInfer(
  original: File,
  effect: string,
  intensity: number,
  audio: File,
): Promise<InferResult> {
  const form = new FormData()
  form.append('original', original)
  form.append('effect', effect)
  form.append('intensity', String(intensity))
  form.append('audio', audio)

  const res = await fetch(`${API_URL}/api/infer`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Inference failed (${res.status}): ${detail}`)
  }
  return res.json() as Promise<InferResult>
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/api/health`)
    if (!res.ok) return false
    const data = await res.json()
    return data.pipeline_loaded === true
  } catch {
    return false
  }
}
