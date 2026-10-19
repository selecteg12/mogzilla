import { useEffect, useRef, useState } from 'react'
import { Camera, X } from 'lucide-react'

export default function CameraModal({ onClose, onCapture }: { onClose: () => void; onCapture: (file: File) => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(false)
  useEffect(() => {
    dialog.current?.showModal()
    let active = true
    let stream: MediaStream | undefined
    const open = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Для камеры нужен localhost или HTTPS.')
        const acquired = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 } }, audio: false })
        if (!active) { acquired.getTracks().forEach(track => track.stop()); return }
        stream = acquired
        if (video.current) { video.current.srcObject = acquired; await video.current.play() }
        if (active) setReady(true)
      } catch (e) { if (active) setError(e instanceof Error ? e.message : 'Камера недоступна. Загрузите фотографию.') }
    }
    void open()
    const current = dialog.current
    return () => { active = false; stream?.getTracks().forEach(track => track.stop()); current?.close() }
  }, [])
  const capture = () => {
    if (!video.current?.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.current.videoWidth; canvas.height = video.current.videoHeight
    canvas.getContext('2d')!.drawImage(video.current, 0, 0)
    canvas.toBlob(blob => { if (blob) { onCapture(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' })); onClose() } }, 'image/jpeg', 0.95)
  }
  return <dialog ref={dialog} className="camera-dialog" onCancel={onClose} aria-labelledby="camera-title">
    <div className="section-heading"><div><span className="eyebrow">СНИМОК С КАМЕРЫ</span><h2 id="camera-title">Посмотрите в камеру.</h2></div><button className="icon-button" onClick={onClose} aria-label="Закрыть камеру"><X size={20} /></button></div>
    <p>Смотрите прямо, используйте равномерный свет и поместите лицо целиком в кадр.</p>
    <video ref={video} autoPlay playsInline muted aria-label="Предпросмотр камеры" />
    {error && <p role="alert" className="error-text">{error}</p>}
    <button className="button primary" disabled={!ready} onClick={capture}><Camera size={17} /> Сделать снимок</button>
    <small>Камера выключается при закрытии окна. Видеозапись не ведётся.</small>
  </dialog>
}
