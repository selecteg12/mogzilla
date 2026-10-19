import { useId, useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import type { Analysis, Box, Metric, RatingPart } from '../types'

export type Layers = { landmarks: boolean; contours: boolean; measurements: boolean; symmetry: boolean; original: boolean }

export default function FaceCanvas({ url, width, height, box, onBox, result, layers, comparison, selected, disabled, zoom, mode, ratingParts, activePart, allCriteria, onCriterion, regionLabel }: {
  url: string; width: number; height: number; box: Box | null; onBox: (box: Box) => void;
  result: Analysis | null; layers: Layers; comparison: number; selected: Metric | null; disabled: boolean; zoom: boolean;
  mode: 'photo' | 'points' | 'deviations'; ratingParts: RatingPart[]; activePart: RatingPart | null; allCriteria: boolean; onCriterion: (id: string) => void;
  regionLabel?: string;
}) {
  const svg = useRef<SVGSVGElement>(null)
  const [draft, setDraft] = useState<Box | null>(null)
  const start = useRef<{x: number; y: number} | null>(null)
  const clipId = useId().replace(/:/g, '')
  const point = (event: PointerEvent<SVGSVGElement>) => {
    const matrix = svg.current!.getScreenCTM()!.inverse()
    const value = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix)
    return { x: Math.max(0, Math.min(width, value.x)), y: Math.max(0, Math.min(height, value.y)) }
  }
  const move = (event: PointerEvent<SVGSVGElement>) => {
    if (!start.current) return
    const end = point(event), begin = start.current
    setDraft({ x: Math.min(begin.x, end.x), y: Math.min(begin.y, end.y), width: Math.abs(end.x - begin.x), height: Math.abs(end.y - begin.y) })
  }
  const finish = () => {
    if (draft && draft.width >= 24 && draft.height >= 24) onBox(draft)
    start.current = null; setDraft(null)
  }
  const region = draft ?? box
  const landmarks = result?.landmarks
  const segments: [number, number][] = []
  if (selected) {
    const inner = selected.substitution.inner, outer = selected.substitution.outer
    if (typeof inner === 'number' && typeof outer === 'number') segments.push([inner, outer])
    else if (selected.point_ids.length === 4) {
      segments.push([selected.point_ids[0], selected.point_ids[1]], [selected.point_ids[2], selected.point_ids[3]])
      if (selected.id === 'eye_spacing') segments.push([selected.point_ids[0], selected.point_ids[2]])
    } else if (selected.point_ids.length === 3) segments.push([selected.point_ids[0], selected.point_ids[1]], [selected.point_ids[1], selected.point_ids[2]])
    else if (selected.point_ids.length === 2) segments.push([selected.point_ids[0], selected.point_ids[1]])
  }
  const view = zoom && box && landmarks ? { x: Math.max(0, box.x - box.width * .32), y: Math.max(0, box.y - box.height * .32), width: Math.min(width, box.width * 1.64), height: Math.min(height, box.height * 1.64) } : { x: 0, y: 0, width, height }
  view.width = Math.min(view.width, width - view.x); view.height = Math.min(view.height, height - view.y)
  const unit = Math.max(view.width, view.height) / 550
  const divider = view.x + view.width * comparison / 100
  const visibleParts = allCriteria ? ratingParts : activePart ? [activePart] : []
  const labels: Record<string, { name: string; side: 'left' | 'right'; y: number }> = { tilt_mean: { name: 'Наклон глаз', side: 'left', y: .17 }, aspect_mean: { name: 'Глаза W/H', side: 'right', y: .32 }, spacing: { name: 'Интервал глаз', side: 'left', y: .45 }, midface_pupils: { name: 'Midface', side: 'right', y: .63 }, mouth_nose: { name: 'Рот / нос', side: 'left', y: .80 } }
  return <div className={`photo-view ${disabled || landmarks ? '' : 'can-select'}`}>
    <svg ref={svg} viewBox={`${view.x} ${view.y} ${view.width} ${view.height}`} role="group" aria-label={regionLabel ?? (disabled || landmarks ? 'Фото и результат в координатах исходного снимка.' : 'Обведите целое лицо: оба глаза, нос, рот и подбородок. Анализ начнётся автоматически.')}
      onPointerDown={event => { if (disabled || landmarks) return; event.currentTarget.setPointerCapture(event.pointerId); start.current = point(event); setDraft(null) }}
      onPointerMove={move} onPointerUp={finish} onPointerCancel={() => { start.current = null; setDraft(null) }}>
      <defs><clipPath id={clipId}><rect x={divider} y={view.y} width={view.width * (1 - comparison / 100)} height={view.height} /></clipPath></defs>
      <image href={url} width={width} height={height} />
      {!landmarks && region && <>
        <path d={`M0 0H${width}V${height}H0Z M${region.x} ${region.y}V${region.y + region.height}H${region.x + region.width}V${region.y}Z`} fill="#080a13" fillOpacity=".58" fillRule="evenodd" />
        <rect x={region.x} y={region.y} width={region.width} height={region.height} rx={4 * unit} fill="none" stroke="#afa2ff" strokeWidth={2 * unit} strokeDasharray={`${8 * unit} ${4 * unit}`} />
      </>}
      {landmarks && mode !== 'photo' && <g clipPath={`url(#${clipId})`}>
        <rect width={width} height={height} fill="#0b0e19" fillOpacity={layers.original ? 0.18 : 1} />
        {layers.contours && result.geometry?.paths.map((path, index) => <polyline key={index} points={path.map(id => `${landmarks[id].x},${landmarks[id].y}`).join(' ')} fill="none" stroke="#ae9dff" strokeWidth={unit} opacity=".7" />)}
        {layers.symmetry && result.geometry && <line
          x1={result.geometry.axis.origin[0] - result.geometry.axis.direction[0] * height * 2} y1={result.geometry.axis.origin[1] - result.geometry.axis.direction[1] * height * 2}
          x2={result.geometry.axis.origin[0] + result.geometry.axis.direction[0] * height * 2} y2={result.geometry.axis.origin[1] + result.geometry.axis.direction[1] * height * 2}
          stroke="#71e0cc" strokeWidth={1.5 * unit} strokeDasharray={`${6 * unit} ${5 * unit}`} />}
        {mode === 'deviations' && layers.measurements && !activePart && segments.map(([a, b], index) => <line key={index} x1={landmarks[a].x} y1={landmarks[a].y} x2={landmarks[b].x} y2={landmarks[b].y} stroke="#f1cd82" strokeWidth={2 * unit} />)}
        {mode === 'deviations' && layers.measurements && visibleParts.map(part => <g key={part.id} data-criterion={part.id} opacity={activePart && activePart.id !== part.id ? .24 : 1}>
          {part.annotations.map((line, index) => <line key={index} x1={line.a[0]} y1={line.a[1]} x2={line.b[0]} y2={line.b[1]} stroke={part.deviation.direction === 'within' ? '#87e0bb' : '#ffd18a'} strokeWidth={(line.kind === 'reference' ? 1.3 : 2.5) * unit} strokeDasharray={line.kind === 'reference' ? `${5 * unit} ${4 * unit}` : undefined}><title>{part.title}: {line.label}. {part.summary}</title></line>)}
        </g>)}
        {layers.landmarks && landmarks.map(p => <circle key={p.id} cx={p.x} cy={p.y} r={(mode === 'deviations' && (activePart?.point_ids.includes(p.id) || selected?.point_ids.includes(p.id)) ? 3.5 : 2) * unit} fill={mode === 'deviations' && (activePart?.point_ids.includes(p.id) || selected?.point_ids.includes(p.id)) ? '#ffe2a4' : '#c8bcff'} stroke="#17121e" strokeWidth={.5 * unit}><title>Ориентир {p.id}: ({p.x.toFixed(2)}, {p.y.toFixed(2)}){selected?.point_ids.includes(p.id) ? ` · ${selected.title}` : ''}</title></circle>)}
        {mode === 'deviations' && layers.measurements && visibleParts.map(part => {
          const label = labels[part.id]
          if (!label || !part.annotations.length) return null
          const x = label.side === 'left' ? view.x + 8 * unit : view.x + view.width - 151 * unit
          const y = view.y + view.height * label.y
          const coordinates = part.annotations.filter(line => line.kind === 'measure').flatMap(line => [line.a, line.b])
          const anchor = coordinates.reduce((sum, point) => [sum[0] + point[0] / coordinates.length, sum[1] + point[1] / coordinates.length], [0, 0])
          const color = part.deviation.direction === 'within' ? '#87e0bb' : '#ffd18a'
          return <g key={part.id} className="photo-annotation" role="button" tabIndex={0} aria-label={`На фото: ${part.title}, ${part.score.toFixed(1)} из 10`} onClick={() => onCriterion(part.id)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onCriterion(part.id) } }}>
            <title>{part.description} {part.summary}</title>
            <line x1={label.side === 'left' ? x + 144 * unit : x} y1={y + 20 * unit} x2={anchor[0]} y2={anchor[1]} stroke={color} strokeWidth={unit} opacity=".7" />
            <rect x={x} y={y} width={144 * unit} height={44 * unit} rx={6 * unit} fill="#12131bee" stroke={color} strokeOpacity={activePart?.id === part.id ? 1 : .4} strokeWidth={unit} />
            <text x={x + 8 * unit} y={y + 16 * unit} fill={color} fontSize={11 * unit} fontWeight="600">{label.name} · {part.score.toFixed(1)}/10</text>
            <text x={x + 8 * unit} y={y + 32 * unit} fill="#d4c8df" fontSize={10 * unit}>{part.observed.toFixed(2)}{part.unit === '°' ? '°' : ''} · −{part.lost_contribution.toFixed(2)} итога</text>
          </g>
        })}
      </g>}
      {landmarks && mode !== 'photo' && box && <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke="#8ed8c4" strokeOpacity=".7" strokeDasharray={`${5 * unit} ${5 * unit}`} strokeWidth={unit} pointerEvents="none" />}
      {landmarks && comparison > 0 && <line x1={divider} y1={view.y} x2={divider} y2={view.y + view.height} stroke="#ffffff" strokeOpacity=".8" strokeWidth={unit} />}
    </svg>
    <div className="photo-label">{landmarks ? mode === 'photo' ? 'Исходный снимок' : mode === 'points' ? `${landmarks.length} точек · реальный выход CNN` : 'Измеренные отрезки → баллы' : region ? regionLabel ?? 'Выделенная область лица' : 'Выделите лицо прямоугольником'}</div>
    {landmarks && <div className="compare-label">{zoom ? 'Лицо крупно' : 'Весь снимок'}</div>}
  </div>
}
