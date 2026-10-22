import type { Analysis, Box, Localization } from './types'

export function localizePhoto(file: File, signal: AbortSignal) {
  return fetchJson<{ localization: Localization; analysis: Analysis | null }>('/api/holistic/analyze', {
    method: 'POST', headers: { 'Content-Type': file.type }, body: file, signal,
  })
}

export async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Ошибка запроса (${response.status}).`)
  return body as T
}

export function processPhoto(file: File, box: Box, analyze: boolean, signal: AbortSignal) {
  const query = new URLSearchParams(Object.entries(box).map(([key, value]) => [key, String(value)]))
  if (analyze) {
    return fetchJson<{ analysis: Analysis | null; localization: Localization }>(`/api/holistic/analyze?${query}`, {
      method: 'POST', headers: { 'Content-Type': file.type }, body: file, signal,
    }).then(result => { if (!result.analysis) throw new Error(result.localization.message ?? 'Не удалось найти лицо.'); return result.analysis })
  }
  return fetchJson<Analysis>(`/api/prepare?${query}`, {
    method: 'POST', headers: { 'Content-Type': file.type }, body: file, signal,
  })
}

export async function normalizePhoto(file: File): Promise<{ file: File; width: number; height: number }> {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error('Выберите JPEG, PNG или WebP.')
  if (file.size > 10 * 1024 ** 2) throw new Error('Выберите изображение меньше 10 МБ.')
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
  try {
    if (bitmap.width * bitmap.height > 20_000_000 || Math.min(bitmap.width, bitmap.height) < 32) throw new Error('Допустимый размер — от 32 × 32 пикселей до 20 мегапикселей.')
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width; canvas.height = bitmap.height
    const context = canvas.getContext('2d')!
    context.fillStyle = '#ffffff'; context.fillRect(0, 0, canvas.width, canvas.height)
    context.drawImage(bitmap, 0, 0)
    // Keep source dimensions so photo, landmarks and measurements use one set
    // of coordinates. Larger uploads are recompressed to fit the hosting limit.
    let blob: Blob | null = null
    for (const quality of [0.95, 0.9, 0.82, 0.72, 0.6, 0.48]) {
      blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('Не удалось преобразовать изображение.')), 'image/jpeg', quality))
      if (blob.size <= 4 * 1024 ** 2) break
    }
    if (!blob || blob.size > 4 * 1024 ** 2) throw new Error('Уменьшите разрешение фотографии и попробуйте ещё раз.')
    return { file: new File([blob], file.name.replace(/\.[^.]+$/, '') + '.jpg', { type: 'image/jpeg' }), width: bitmap.width, height: bitmap.height }
  } finally { bitmap.close() }
}
