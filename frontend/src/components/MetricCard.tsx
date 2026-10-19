import { ChevronDown, FunctionSquare } from 'lucide-react'
import type { Metric } from '../types'
import ParameterHelp from './ParameterHelp'

export default function MetricCard({ metric, active, onSelect, error }: { metric: Metric; active: boolean; onSelect: () => void; error?: number }) {
  return <article className={`metric-card ${active ? 'active' : ''}`}>
    <div className="metric-heading-row"><button className="metric-summary" onClick={onSelect} aria-pressed={active} title={metric.description}>
      <span><span className="metric-category">{metric.category}</span><span className="metric-title">{metric.title}</span></span>
      <strong>{metric.value === null ? '—' : metric.value.toFixed(metric.unit === '°' ? 2 : 3)}<small>{metric.unit === '°' ? '°' : ' отн.'}</small></strong>
    </button><ParameterHelp title={metric.title}>{`${metric.description} ${metric.limitation}${error !== undefined ? ` Средняя погрешность на проверочных фотографиях: ${error.toFixed(metric.unit === '°' ? 1 : 3)} ${metric.unit === '°' ? '°' : 'отн.'}. Это не граница ошибки данного снимка.` : ''}`}</ParameterHelp></div>
    <details onToggle={event => { if (event.currentTarget.open && !active) onSelect() }}>
      <summary><FunctionSquare size={14} /> Показать расчёт <ChevronDown size={14} /></summary>
      <div className="calculation"><code>{metric.formula}</code>
        <div className="point-list">{metric.points.map(p => <span key={p.id}>p{p.id} <b>({p.x.toFixed(2)}, {p.y.toFixed(2)})</b></span>)}</div>
        <details><summary>Промежуточные значения</summary><pre>{JSON.stringify(metric.substitution, null, 2)}</pre></details>
        {error !== undefined && <p className="calibration">Средняя погрешность на проверочных фотографиях: {error.toFixed(3)} {metric.unit === '°' ? '°' : 'отн.'}. Погрешность конкретного снимка может быть другой.</p>}
        <p>{metric.limitation}</p>
      </div>
    </details>
  </article>
}
