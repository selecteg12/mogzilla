import { useId, useState } from 'react'
import { ArrowUpRight, Plus } from 'lucide-react'
import type { Analysis, RegionObservation } from '../types'
import { regionDrawing, regionViewport } from '../regionGeometry.mjs'

type Photo = {url:string; width:number; height:number}
const measured: Record<string,string> = {
  eyes:'Наклон уголков, раскрытие обоих глаз и межглазное расстояние.',
  brows:'Расстояние от каждой брови до глаза, наклон и изгиб обеих бровей.',
  upper:'Высота видимой верхней области относительно среднего сегмента.',
  midface:'Расстояние между зрачками, высота до губы, ширина и высота контура.',
  nose_mouth:'Ширина рта относительно носа и межзрачкового расстояния.',
  lower:'Ширина челюсти, длина нижнего сегмента и пропорции подбородка.',
  symmetry:'Расхождения парных точек после отражения и различия пропорций глаз.',
  skin:'Разброс яркости в трёх отмеченных участках щёк и лба.',
}

function RegionCard({photo, result, region, index, onMetric}: {photo:Photo; result:Analysis; region:RegionObservation; index:number; onMetric:(id:string)=>void}) {
  const [expanded, setExpanded] = useState(false)
  const id = useId()
  const drawing = regionDrawing(result,region)
  const landmarks = result.landmarks ?? []
  const view = regionViewport(landmarks,region.id,photo.width,photo.height)
  const unit = Math.max(view.width,view.height)/430
  const activeIds = new Set(drawing.points.map(p=>p.id))
  const contextPoints = landmarks.filter(p=>!activeIds.has(p.id))
  const skinRadius = Math.max(3,Math.floor(Math.hypot((result.landmarks?.[234]?.x ?? 0)-(result.landmarks?.[454]?.x ?? 0),(result.landmarks?.[234]?.y ?? 0)-(result.landmarks?.[454]?.y ?? 0))*.055))
  return <article className={`zone-card ${expanded ? 'is-expanded' : ''}`} data-region={region.id}>
    <div className="zone-card-top"><span className="micro">{String(index+1).padStart(2,'0')}</span><span>Фрагмент крупнее</span></div>
    <div className="zone-portrait" style={{aspectRatio:`${view.width}/${view.height}`}}>
      <svg viewBox={`${view.x} ${view.y} ${view.width} ${view.height}`} role="img" aria-label={`${region.title}: увеличенный фрагмент, ориентиры и измеренные линии`}>
        <image href={photo.url} width={photo.width} height={photo.height} />
        <rect x={view.x} y={view.y} width={view.width} height={view.height} fill="#101010" opacity=".15"/>
        <g className="zone-context-paths" aria-hidden="true">{result.geometry?.paths.map((path,i)=><polyline key={i} points={path.flatMap(id=>landmarks[id]?[`${landmarks[id].x},${landmarks[id].y}`]:[]).join(' ')} fill="none" strokeWidth={.8*unit}/>)}</g>
        <g className="zone-context-points" aria-hidden="true">{contextPoints.map(p=><circle key={p.id} cx={p.x} cy={p.y} r={1.25*unit}/>)}</g>
        <g className="zone-contours">{drawing.contours.map((path,i)=><polyline key={i} points={path.map(p=>`${p.x},${p.y}`).join(' ')} fill="none" strokeWidth={1.7*unit}/>)}</g>
        <g className="zone-lines">{drawing.lines.map((line,i)=><line key={i} x1={line.a.x} y1={line.a.y} x2={line.b.x} y2={line.b.y} strokeWidth={2.8*unit} data-metric={line.metric}><title>{line.label}</title></line>)}</g>
        {region.id==='symmetry' && result.geometry && <line className="zone-axis" x1={result.geometry.axis.origin[0]-result.geometry.axis.direction[0]*photo.height*2} y1={result.geometry.axis.origin[1]-result.geometry.axis.direction[1]*photo.height*2} x2={result.geometry.axis.origin[0]+result.geometry.axis.direction[0]*photo.height*2} y2={result.geometry.axis.origin[1]+result.geometry.axis.direction[1]*photo.height*2} strokeWidth={1.4*unit} strokeDasharray={`${6*unit} ${5*unit}`}/>}
        {region.id==='skin' && drawing.points.map(p=><circle key={p.id} className="zone-sample" cx={p.x} cy={p.y} r={skinRadius} strokeWidth={unit}/>)}
        <g className="zone-points">{drawing.points.map(p=><circle key={p.id} data-point={p.id} cx={p.x} cy={p.y} r={(region.id==='symmetry'?1.8:3.2)*unit} strokeWidth={.85*unit}><title>Ориентир {p.id}</title></circle>)}</g>
      </svg>
      <div className="zone-hover-note" aria-hidden="true"><span className="micro">what's measured</span><p>{measured[region.id] ?? region.subtitle}</p></div>
    </div>
    <h3 className="zone-title"><button type="button" className="zone-toggle" aria-expanded={expanded} aria-controls={id} onClick={()=>setExpanded(value=>!value)}><span>{region.title}</span><Plus size={20}/></button></h3>
    <div className="zone-explanation" id={id} hidden={!expanded}><p>{measured[region.id]}</p><p>{region.summary}</p><dl>{region.metrics.map(metric=><div key={metric.id}><dt>{metric.title}</dt><dd>{metric.value==null?'—':metric.value.toFixed(metric.unit==='°'?1:3)}{metric.unit==='ratio'?'':metric.unit}</dd></div>)}</dl><div className="zone-formula-links">{region.metrics.filter(metric=>result.geometry?.metrics.some(m=>m.id===metric.id)).map(metric=><button key={metric.id} type="button" className="text-button" onClick={()=>onMetric(metric.id)}>{metric.title} <ArrowUpRight size={13}/></button>)}</div></div>
    <p className="zone-caption">{region.subtitle}</p>
  </article>
}

export default function RegionBreakdown({photo,result,onMetric}: {photo:Photo; result:Analysis; onMetric:(id:string)=>void}) {
  if (!result.landmarks || !result.holistic) return null
  const regions = result.holistic.regions.map(region=>region.id==='brows' ? {...region,metrics:[...region.metrics,...(result.geometry?.metrics.filter(metric=>metric.id.startsWith('brow_arch_') && !region.metrics.some(m=>m.id===metric.id)) ?? [])]} : region)
  return <section className="zone-breakdown" aria-labelledby="zone-title"><div className="zone-heading"><div><span className="micro">under the surface</span><h2 id="zone-title">anatomy of a mog.</h2></div><p>Наведи на фото, чтобы увидеть, что измерено. Нажми на название зоны — откроются значения и пояснения.</p></div><div className="zone-legend"><span><i/>Точки и измерения зоны</span><span><i/>Остальная сетка ориентиров</span></div><p className="zone-context">Разметка описывает пропорции на снимке. Фоновые точки показаны для контекста; это не дополнительные измерения и не карта вклада в рейт.</p><div className="zone-grid">{regions.map((region,index)=><RegionCard key={region.id} photo={photo} result={result} region={region} index={index} onMetric={onMetric}/>)}</div></section>
}
