import { useId, useState } from 'react'
import { createPortal } from 'react-dom'
import { CircleHelp } from 'lucide-react'

export default function ParameterHelp({ title, children }: { title: string; children: string }) {
  const id = useId()
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null)
  const show = (element: HTMLElement) => {
    const rect = element.getBoundingClientRect()
    const width = Math.min(310, window.innerWidth - 24)
    setPosition({ left: Math.max(12, Math.min(rect.left - width + rect.width, window.innerWidth - width - 12)), top: rect.bottom + 9 })
  }
  return <span className="parameter-help" onMouseEnter={e => show(e.currentTarget)} onMouseLeave={() => setPosition(null)}>
    <button type="button" className="help-trigger" aria-label={`Пояснение: ${title}`} aria-describedby={position ? id : undefined}
      onFocus={e => show(e.currentTarget)} onBlur={() => setPosition(null)} onClick={e => show(e.currentTarget)} onKeyDown={e => { if (e.key === 'Escape') setPosition(null) }}><CircleHelp size={15} /></button>
    {position && createPortal(<span id={id} role="tooltip" className="parameter-tooltip" style={position}><strong>{title}</strong><span>{children}</span></span>, document.body)}
  </span>
}
