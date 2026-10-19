import { useEffect, useState } from 'react'
import { ArrowLeft, ArrowUpRight, CircleHelp, LoaderCircle } from 'lucide-react'
import { fetchJson } from '../api'
import type { ReferenceValues } from '../types'
import ParameterHelp from './ParameterHelp'

export default function ModelValuesPage() {
  const [data, setData] = useState<ReferenceValues | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const request = new AbortController()
    void fetchJson<ReferenceValues>('/api/reference-values', { signal: request.signal }).then(setData).catch(e => { if (!request.signal.aborted) setError(e.message) })
    return () => request.abort()
  }, [])
  const reference = data?.reference
  const parts = reference?.rating.parts ?? []
  const updatedAt = data?.gallery.generated_at ? new Date(data.gallery.generated_at) : null
  return <section className="model-values-page" id="model-values">
    <a className="reference-back" href="#workspace"><ArrowLeft size={14} /> Вернуться к анализу фото</a>
    <div className="reference-heading"><div><span className="eyebrow">ОРИЕНТИРЫ ТЕКУЩЕЙ ШКАЛЫ</span><h1>Модельные значения<span>.</span></h1><p>Какие пропорции дают 10 из 10 — и как с ними соотносятся реальные фотографии.</p></div><span className="reference-total">10<span>/ 10</span><small>Все пять критериев<br />в заданных диапазонах</small></span></div>
    <p className="reference-context">Здесь «модельные» означает целевые значения формулы приложения. Это выбранная шкала предпочтений, а не универсальный стандарт внешности.</p>
    {!data && !error && <p role="status" className="reference-loading"><LoaderCircle size={17} className="spin" /> Загружаем значения и проверенные примеры…</p>}
    {error && <div className="error-banner" role="alert"><CircleHelp size={17} /> {error}</div>}
    {reference && <>
      <div className="reference-target-grid"><div className="reference-diagram"><span className="card-kicker">ГЕОМЕТРИЧЕСКАЯ СХЕМА · НЕ ФОТОГРАФИЯ</span>
        <svg viewBox={`80 95 380 330`} role="img" aria-label="Построенная схема пропорций со всеми пятью критериями на 10 из 10">
          <ellipse cx="272" cy="240" rx="173" ry="167" fill="#b499e905" />
          {reference.paths.map((path, i) => <polyline key={i} points={path.map(id => `${reference.points[id].x},${reference.points[id].y}`).join(' ')} fill="none" stroke="#c3abef" strokeWidth={1.4} opacity={i === 0 ? .8 : .55} />)}
          {parts.flatMap(part => part.annotations.filter(a => a.kind === 'measure').map((a, i) => <line key={`${part.id}-${i}`} x1={a.a[0]} y1={a.a[1]} x2={a.b[0]} y2={a.b[1]} stroke="#8fd6bc" strokeWidth={1.2} opacity=".6" />))}
          {reference.points.map(point => <circle key={point.id} cx={point.x} cy={point.y} r={1.5} fill={point.id > 95 ? '#c9f2df' : '#bfa7e7'} />)}
          <path d="M190 180H125V160H95M270 243H420V265H452M270 310H130V328H94" fill="none" stroke="#8fd6bc" strokeOpacity=".4" strokeDasharray="3 4" />
        </svg>
        <p>Схема построена по серединам диапазонов. Числа проверены той же формулой, что используется для загруженных фотографий.</p><span className="diagram-score">5 критериев × 10/10 · без снятия баллов</span>
      </div><div className="reference-table-panel"><h2>Значения для 10/10</h2><p>Любое значение внутри указанного диапазона даёт максимальный балл за этот критерий.</p><div className="reference-table-scroll"><table className="reference-values-table"><thead><tr><th>Параметр</th><th>Диапазон 10/10</th><th>На схеме</th><th>Балл</th></tr></thead><tbody>{parts.map(part => <tr key={part.id}><td><div>{part.title}<ParameterHelp title={part.title}>{part.description}</ParameterHelp></div></td><td>{part.low}–{part.high}{part.unit === '°' ? '°' : ''}</td><td>{part.observed}{part.unit === '°' ? '°' : ''}</td><td><b>10/10</b></td></tr>)}</tbody></table></div><div className="reference-formula"><span>Все критерии в диапазоне → потери = 0</span><code>(10 + 10 + 10 + 10 + 10) / 5 = 10</code></div></div></div>

      <section className="reference-examples"><div className="section-heading"><div><span className="eyebrow">ФОТОГРАФИИ С РЕАЛЬНЫМИ ИЗМЕРЕНИЯМИ</span><h2>Близкие примеры.</h2></div></div>
        {updatedAt && Number.isFinite(updatedAt.getTime()) && <p className="reference-provenance">Обновлено {new Intl.DateTimeFormat('ru-RU', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'Europe/Moscow' }).format(updatedAt)}</p>}
        <p className="section-intro">{data?.gallery.examples.length ? 'Сравните значения на фотографиях с диапазонами выше. В карточках видно, какие пропорции повлияли на балл.' : data?.gallery.message}</p>
        <div className="reference-gallery">{data?.gallery.examples.map(example => <article className="reference-photo-card" key={example.id}><div className="reference-photo"><img src={example.image_url} alt={`${example.title}: реальная фотография из WFLW; геометрический балл ${example.value.toFixed(2)} из 10`} width="384" height="384" loading="lazy" /><span>{example.all_ten ? 'Все критерии — 10/10' : 'Близкий пример · не 10/10'}</span></div><div className="reference-photo-details"><div className="reference-example-title"><h3>{example.title}</h3><strong>{example.value.toFixed(2)}<small>/ 10</small></strong></div><p>{example.dataset}</p><dl>{example.parts.map(part => <div key={part.id} title={part.description}><dt>{part.title}<small>{part.observed.toFixed(3)}{part.unit === '°' ? '°' : ''}</small></dt><dd className={part.deviation.direction === 'within' ? 'in-band' : ''}>{part.score.toFixed(1)}<small>/10</small></dd></div>)}</dl><p className="reference-example-note">Совпало с диапазоном 10/10: {example.parts.filter(part => part.deviation.direction === 'within').length} из 5. Фото не объявляется эталоном человека.</p></div></article>)}</div>
        {!!data?.gallery.examples.length && <p className="reference-provenance">Баллы рассчитаны по исходным снимкам. Здесь показаны увеличенные фрагменты лиц; повторный анализ обрезанного изображения может дать другой результат.</p>}
        {data?.gallery.source_url && <a className="reference-source" href={data.gallery.source_url} target="_blank" rel="noreferrer">WFLW: источник исследовательских фотографий <ArrowUpRight size={13} /></a>}
      </section>
      <div className="reference-footer"><h2>Как сравнивать со своим снимком</h2><p>Фронтальный ракурс, спокойное выражение и открытые глаза уменьшают влияние позы и мимики. Полученный балл описывает измеренные пропорции конкретного кадра. Погрешность точек может менять результат.</p><a className="button primary" href="#workspace">Проанализировать фотографию <ArrowUpRight size={16} /></a></div>
    </>}
  </section>
}
