import { useEffect, useRef, useState } from 'react'
import { Camera, SwitchCamera, X } from 'lucide-react'
import { startCameraPreview } from '../cameraPreview'

type FacingMode = 'user' | 'environment'

export default function CameraModal({ onClose, onCapture }: { onClose: () => void; onCapture: (file: File) => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const preview = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(false)
  const [opening, setOpening] = useState(true)
  const [cameraRequest, setCameraRequest] = useState<{ facingMode: FacingMode }>({ facingMode: 'user' })
  const { facingMode } = cameraRequest
  const [activeFacingMode, setActiveFacingMode] = useState<FacingMode>('user')
  useEffect(() => {
    const current = dialog.current
    current?.showModal()
    return () => current?.close()
  }, [])
  useEffect(() => {
    let active = true
    let stream: MediaStream | undefined
    let stopPreview = () => {}
    const currentVideo = video.current
    const currentPreview = preview.current
    setReady(false)
    setOpening(true)
    setError('')
    const open = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Для камеры нужен localhost или HTTPS.')
        const acquired = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: facingMode === 'environment' ? { exact: 'environment' } : { ideal: 'user' },
            width: { ideal: 1280 },
          },
          audio: false,
        })
        if (!active) { acquired.getTracks().forEach(track => track.stop()); return }
        stream = acquired
        if (!currentVideo || !currentPreview) { acquired.getTracks().forEach(track => track.stop()); return }
        currentVideo.srcObject = acquired
        await currentVideo.play()
        if (active) {
          const selectedMode = acquired.getVideoTracks()[0]?.getSettings().facingMode
          setActiveFacingMode(selectedMode === 'user' || selectedMode === 'environment' ? selectedMode : facingMode)
          stopPreview = startCameraPreview(currentVideo, currentPreview, () => {
            if (active) { setReady(true); setOpening(false) }
          })
        }
      } catch (e) {
        stopPreview()
        stream?.getTracks().forEach(track => track.stop())
        if (!active) return
        setOpening(false)
        if (currentVideo && currentVideo.srcObject === stream) currentVideo.srcObject = null
        const name = e instanceof Error ? e.name : ''
        if (name === 'NotAllowedError' || name === 'SecurityError') {
          setError('Разрешите доступ к камере в настройках браузера или загрузите фотографию.')
        } else if (facingMode === 'environment') {
          setError('Не удалось открыть заднюю камеру. Переключитесь на фронтальную или загрузите фотографию.')
        } else {
          setError('Не удалось открыть фронтальную камеру. Попробуйте заднюю камеру или загрузите фотографию.')
        }
      }
    }
    void open()
    return () => {
      active = false
      stopPreview()
      stream?.getTracks().forEach(track => track.stop())
      if (currentVideo && currentVideo.srcObject === stream) currentVideo.srcObject = null
    }
  }, [cameraRequest, facingMode])
  const switchCamera = () => {
    if (opening) return
    setReady(false)
    setOpening(true)
    setCameraRequest(request => ({ facingMode: (ready ? activeFacingMode : request.facingMode) === 'user' ? 'environment' : 'user' }))
  }
  const capture = () => {
    if (!ready || opening || !preview.current?.width) return
    const canvas = document.createElement('canvas')
    canvas.width = preview.current.width; canvas.height = preview.current.height
    canvas.getContext('2d')!.drawImage(preview.current, 0, 0)
    canvas.toBlob(blob => { if (blob) { onCapture(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' })); onClose() } }, 'image/jpeg', 0.95)
  }
  return <dialog ref={dialog} className="camera-dialog" onCancel={onClose} aria-labelledby="camera-title">
    <div className="section-heading"><div><span className="eyebrow">СНИМОК С КАМЕРЫ</span><h2 id="camera-title">Посмотрите в камеру.</h2></div><button className="icon-button" onClick={onClose} aria-label="Закрыть камеру"><X size={20} /></button></div>
    <p>Смотрите прямо, используйте равномерный свет и поместите лицо целиком в кадр.</p>
    <div className="camera-preview">
      <video ref={video} autoPlay playsInline muted aria-hidden="true" />
      <canvas ref={preview} role="img" aria-label="Предпросмотр камеры без зеркального отражения" />
    </div>
    <p className="camera-status" role="status">{opening ? 'Открываем камеру…' : ready ? activeFacingMode === 'user' ? 'Фронтальная камера' : 'Задняя камера' : 'Камера недоступна'}</p>
    {error && <p role="alert" className="error-text">{error}</p>}
    <div className="camera-actions">
      <button className="button primary" disabled={!ready || opening} onClick={capture}><Camera size={17} /> Сделать снимок</button>
      <button className="button secondary" disabled={opening} onClick={switchCamera}><SwitchCamera size={17} /> Переключить камеру</button>
    </div>
    <small>Камера выключается при закрытии окна. Видеозапись не ведётся.</small>
  </dialog>
}
