// Render and capture the same pixels instead of relying on a mobile browser's
// video presentation. Do not mirror, rotate, crop, or stretch the camera frame.
export function startCameraPreview(video: HTMLVideoElement, canvas: HTMLCanvasElement, onFirstFrame: () => void) {
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Не удалось открыть предпросмотр камеры.')
  let stopped = false
  let painted = false
  let videoFrame: number | undefined
  let animationFrame: number | undefined

  const paint = () => {
    if (stopped) return
    const width = video.videoWidth
    const height = video.videoHeight
    if (video.readyState >= 2 && width > 0 && height > 0) {
      if (canvas.width !== width) canvas.width = width
      if (canvas.height !== height) canvas.height = height
      context.drawImage(video, 0, 0, width, height)
      if (!painted) { painted = true; onFirstFrame() }
    }
    if (typeof video.requestVideoFrameCallback === 'function') {
      videoFrame = video.requestVideoFrameCallback(paint)
    } else {
      animationFrame = requestAnimationFrame(paint)
    }
  }
  paint()

  return () => {
    stopped = true
    if (videoFrame !== undefined) video.cancelVideoFrameCallback(videoFrame)
    if (animationFrame !== undefined) cancelAnimationFrame(animationFrame)
    context.clearRect(0, 0, canvas.width, canvas.height)
  }
}
