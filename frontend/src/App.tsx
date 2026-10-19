import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { ArrowRight, Camera, ChevronRight, CircleHelp, Download, Focus, ImagePlus, LoaderCircle, LockKeyhole, ScanFace, ShieldCheck, Sparkles, Upload, X } from 'lucide-react'
import { fetchJson, localizePhoto, normalizePhoto, processPhoto } from './api'
import type { Analysis, Box, Metric, ModelStatus } from './types'
import FaceCanvas from './components/FaceCanvas'
import type { Layers } from './components/FaceCanvas'
import MetricCard from './components/MetricCard'
import AnalysisWalkthrough from './components/AnalysisWalkthrough'
import { useGeometryRating } from './useGeometryRating'

const CameraModal = lazy(() => import('./components/CameraModal'))
const ModelValuesPage = lazy(() => import('./components/ModelValuesPage'))
const SOURCE_LAYERS: Layers = { landmarks: false, contours: false, measurements: false, symmetry: false, original: true }

export default function App() {
  const [referencePage, setReferencePage] = useState(() => window.location.hash === '#model-values')
  const [model, setModel] = useState<ModelStatus | null>(null)
  const [backendError, setBackendError] = useState(false)
  const [photo, setPhoto] = useState<{ file: File; url: string; width: number; height: number } | null>(null)
  const [box, setBox] = useState<Box | null>(null)
  const [result, setResult] = useState<Analysis | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [camera, setCamera] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [selectedId, setSelectedId] = useState('')
  const [criterionId, setCriterionId] = useState('')
  const [hoveredCriterion, setHoveredCriterion] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const controller = useRef<AbortController | null>(null)
  const photoVersion = useRef(0)
  const ratingState = useGeometryRating(result)
  const parts = ratingState.rating?.parts ?? []
  const worst = parts.reduce<(typeof parts)[number] | null>((best, part) => !best || part.lost_contribution > best.lost_contribution ? part : best, null)
  const activePart = hoveredCriterion ? parts.find(part => part.id === hoveredCriterion) ?? null : selectedId ? parts.find(part => part.metric_ids.includes(selectedId)) ?? null : parts.find(part => part.id === criterionId) ?? worst

  useEffect(() => {
    const followHash = () => {
      const hash = window.location.hash
      setReferencePage(hash === '#model-values')
      window.requestAnimationFrame(() => {
        if (hash && hash !== '#model-values') document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'instant' })
        else window.scrollTo({ top: 0, behavior: 'instant' })
      })
    }
    window.addEventListener('hashchange', followHash)
    window.addEventListener('popstate', followHash)
    return () => { window.removeEventListener('hashchange', followHash); window.removeEventListener('popstate', followHash) }
  }, [])

  const refresh = useCallback(async () => {
    try {
      const status = await fetchJson<ModelStatus>('/api/model')
      setModel(status); setBackendError(false)
    } catch { setBackendError(true) }
  }, [])
  useEffect(() => { void refresh(); const timer = window.setInterval(() => { void refresh() }, 15000); return () => window.clearInterval(timer) }, [refresh])
  useEffect(() => { const url = photo?.url; return () => { if (url) URL.revokeObjectURL(url) } }, [photo?.url])
  useEffect(() => () => { controller.current?.abort(); photoVersion.current++ }, [])

  const loadPhoto = async (file: File) => {
    const version = ++photoVersion.current
    controller.current?.abort(); setBusy(true); setError(''); setCriterionId(''); setHoveredCriterion(null)
    const request = new AbortController(); controller.current = request
    try {
      const normalized = await normalizePhoto(file)
      if (version !== photoVersion.current) return
      setPhoto({ ...normalized, url: URL.createObjectURL(normalized.file) })
      setBox(null); setResult(null); setSelectedId('')
      window.setTimeout(() => document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 40)
      const found = await localizePhoto(normalized.file, request.signal)
      if (version !== photoVersion.current || request.signal.aborted) return
      if (found.analysis) {
        setResult(found.analysis)
        const [x, y, width, height] = found.analysis.box
        setBox({ x, y, width, height })
      } else setError('Не удалось надёжно найти лицо. Загрузите чёткое фото анфас или обведите лицо на снимке — анализ начнётся автоматически.')
    } catch (e) { if (version === photoVersion.current) setError(e instanceof Error ? e.message : 'Не удалось прочитать изображение.') }
    finally { if (version === photoVersion.current) setBusy(false) }
  }
  const changeBox = async (region: Box) => {
    if (!photo) return
    controller.current?.abort()
    photoVersion.current++; setBox(region); setSelectedId(''); setCriterionId(''); setHoveredCriterion(null)
    const request = new AbortController(); controller.current = request
    setBusy(true); setError(''); setResult(null)
    try {
      const data = await processPhoto(photo.file, region, true, request.signal)
      if (request.signal.aborted) return
      setResult(data); setSelectedId(''); setCriterionId('')
    } catch (e) { if (!request.signal.aborted) setError(e instanceof Error ? e.message : 'Ошибка анализа. Попробуйте ещё раз.') }
    finally { if (controller.current === request) setBusy(false) }
  }
  const reset = () => { controller.current?.abort(); photoVersion.current++; setPhoto(null); setBox(null); setResult(null); setError(''); setBusy(false) }
  const selectMetric = (id: string) => { setSelectedId(id); setCriterionId(''); setHoveredCriterion(null); document.getElementById('geometry-stage')?.scrollIntoView({ behavior: 'smooth' }) }
  const selectCriterion = (id: string, scroll = false) => { setCriterionId(id); setSelectedId(''); if (scroll) document.getElementById('geometry-stage')?.scrollIntoView({ behavior: 'smooth' }) }
  const exportMeasurements = () => {
    if (!result) return
    const { preprocessed_image: _image, heatmap_image: _heatmap, ...measurements } = result
    const url = URL.createObjectURL(new Blob([JSON.stringify({ ...measurements, rating: ratingState.rating }, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = 'facemetrics-measurements.json'; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const selected: Metric | null = result?.geometry?.metrics.find(item => item.id === (activePart?.metric_ids[0] ?? selectedId)) ?? null
  const metrics = result?.geometry?.metrics ?? []
  const warnings = [...(result?.quality.warnings ?? []), ...(result?.geometry?.warnings ?? [])]

  return <div onClickCapture={event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    const link = (event.target as Element).closest<HTMLAnchorElement>('a[href^="#"]')
    if (!link) return
    const hash = link.hash
    event.preventDefault()
    if (window.location.hash !== hash) window.history.pushState(null, '', hash || '#')
    setReferencePage(hash === '#model-values')
    window.requestAnimationFrame(() => {
      if (hash && hash !== '#model-values') document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'instant' })
      else window.scrollTo({ top: 0, behavior: 'instant' })
    })
  }}>
    <a className="skip-link" href="#workspace">Перейти к анализу</a>
    <header className="site-header"><a className="brand" href="#" aria-label="FaceMetrics AI — главная"><span className="brand-mark"><ScanFace size={22} /></span><span>FaceMetrics<span className="brand-ai">AI</span></span></a>
      <nav aria-label="Главное меню"><a className={!referencePage ? 'nav-active' : ''} href="#workspace">Анализ</a><a className={referencePage ? 'nav-active' : ''} aria-current={referencePage ? 'page' : undefined} href="#model-values">Модельные значения</a><a className="nav-method" href="#photo-tips">Советы для фото</a></nav>
      <div className="header-status"><span className={`status-dot ${model?.ready && !backendError ? 'ready' : ''}`} />{backendError ? 'Сервис недоступен' : model?.ready ? 'Готово к анализу' : 'Временно недоступно'}</div>
    </header>
    <main hidden={referencePage}>
      <section id="workspace" className="workspace section-anchor">
        <div className="section-heading workspace-heading"><div><h1>{photo ? 'Результат анализа' : 'Анализ лица по фото'}</h1><p className="section-intro">Узнайте балл по пропорциям лица и посмотрите, какие показатели повлияли на результат.</p></div></div>
        <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" aria-label="Загрузить фотографию лица" onChange={e => { const file = e.target.files?.[0]; if (file) void loadPhoto(file); e.target.value = '' }} />
        {!photo ? <div className={`upload-area ${dragging ? 'drag-active' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); const file = e.dataTransfer.files[0]; if (file) void loadPhoto(file) }}>
          <div className="upload-icon"><ScanFace size={35} strokeWidth={1.3} /><span className="upload-spark"><Sparkles size={13} /></span></div>
          <h2>Выберите фото анфас</h2><p>Лицо целиком в кадре, глаза открыты, ровное освещение.</p>
          <div className="upload-buttons"><button className="button primary" disabled={busy} onClick={() => input.current?.click()}>{busy ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />} Загрузить фото <ArrowRight size={15} /></button><button className="button secondary" onClick={() => setCamera(true)} disabled={busy}><Camera size={17} /> Сделать снимок</button></div>
          <span className="upload-hint">или перетащите изображение · JPG, PNG, WEBP · до 10 МБ</span>
          <div className="privacy-line"><ShieldCheck size={14} /> Фото обрабатывается в памяти и не сохраняется в базу данных.</div>
        </div> : <div className="analysis-layout">
          <div className="image-panel"><div className="panel-toolbar"><span><ImagePlus size={15} /> {photo.file.name}</span><div><button className="icon-button" onClick={() => input.current?.click()} aria-label="Выбрать другую фотографию" disabled={busy}><Upload size={16} /></button><button className="icon-button" onClick={reset} aria-label="Убрать фотографию"><X size={17} /></button></div></div>
            <FaceCanvas url={photo.url} width={photo.width} height={photo.height} box={box} onBox={changeBox} result={result} layers={SOURCE_LAYERS} comparison={0} selected={null} disabled={busy} zoom={false} mode="photo" ratingParts={[]} activePart={null} allCriteria={false} onCriterion={selectCriterion} />
            {result?.landmarks ? <div className="original-photo-note"><strong>Оригинальная фотография без наложений</strong><p>Поэтапный разбор с точками, измерениями и снятием баллов расположен ниже.</p><a className="text-button" href="#walkthrough">Перейти к разбору ↓</a></div> : <div className="region-controls"><p><Focus size={15} /> {busy ? 'Ищем лицо, находим точки и рассчитываем рейтинг…' : 'Если лицо не найдено автоматически, обведите его на снимке. Анализ начнётся сразу.'}</p></div>}
          </div>
          <aside className="analysis-sidebar">{!result?.landmarks && <><span className="card-kicker"><Focus size={15} /> АВТОМАТИЧЕСКИЙ АНАЛИЗ</span><h3>{busy ? 'Изучаем фотографию…' : 'Ожидаем лицо в кадре.'}</h3><p>Приложение найдёт точки, измерит пропорции и сразу рассчитает рейтинг. Под оригиналом появится разбор всех этапов.</p></>}
            {result?.landmarks && <div className="sidebar-rating"><span className="card-kicker">РЕЙТИНГ ПО ГЕОМЕТРИИ</span><div className="sidebar-score" aria-live="polite"><strong>{ratingState.rating?.value != null ? ratingState.rating.value.toFixed(1) : '—'}</strong><span>/ 10</span></div><span className="review-badge">Автоматический расчёт · субъективная шкала</span><p>Как получен балл и где он снижается — в поэтапном разборе под оригиналом.</p><p className="fine-print">Ракурс, мимика и погрешность точек влияют на результат.</p><a className="text-button" href="#walkthrough">Посмотреть все этапы ↓</a>{ratingState.error && <p className="error-text" role="alert">{ratingState.error}</p>}</div>}
            <p className="local-note"><LockKeyhole size={12} /> Фото используется для расчёта и не сохраняется.</p>
          </aside>
        </div>}
        {error && <div className="error-banner" role="alert"><CircleHelp size={17} /><span>{error}</span><button className="icon-button" onClick={() => setError('')} aria-label="Закрыть сообщение"><X size={16} /></button></div>}
        {backendError && <div className="error-banner" role="status">Не удалось подключиться к сервису. <button className="text-button" onClick={() => void refresh()}>Попробовать снова</button></div>}
      </section>

      {photo && result?.landmarks && <AnalysisWalkthrough key={`${photoVersion.current}-${result.total_milliseconds}`} photo={photo} result={result} ratingState={ratingState} activePart={activePart} selected={selected} onSelect={selectCriterion} onHover={setHoveredCriterion} />}

      {result && <section className="results-section section-anchor" id="results">
        <div className="section-heading"><div><h2>Все показатели</h2></div>{result.landmarks && <button className="button secondary small" onClick={exportMeasurements}><Download size={14} /> Сохранить расчёты</button>}</div>
        {warnings.length > 0 && <div className="warning-box"><CircleHelp size={18} /><div><strong>Что могло повлиять на результат</strong>{warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div></div>}
        {result.geometry && <>
          <div className="metric-grid">{metrics.map(metric => <MetricCard key={metric.id} metric={metric} active={metric.id === selectedId} onSelect={() => selectMetric(metric.id)} error={result.reliability?.geometry_error[metric.id]?.mean_absolute_error} />)}</div></>}
      </section>}

      <section id="photo-tips" className="terminology-section section-anchor"><div><h2>Чтобы результат был точнее</h2><p>Для сравнения используйте снимки с одинаковым ракурсом и выражением лица.</p></div><div className="glossary"><details><summary>Какое фото подойдёт?<ChevronRight size={16} /></summary><p>Смотрите прямо в камеру, держите голову ровно и выберите спокойное выражение лица. Нужны открытые глаза и равномерный свет. Волосы, очки и руки не должны закрывать лицо.</p></details><details><summary>Что означает балл?<ChevronRight size={16} /></summary><p>Балл показывает, насколько измеренные пропорции соответствуют диапазонам этой шкалы. Под фотографией видно влияние каждого показателя на итог. Это субъективная оценка пропорций конкретного снимка.</p></details><details><summary>Почему на разных фото меняется результат?<ChevronRight size={16} /></summary><p>Поворот головы, улыбка, прищур и близкая съёмка меняют видимые пропорции. Размытое фото и резкие тени также мешают точно определить точки.</p></details></div></section>
    </main>
    {referencePage && <main><Suspense fallback={<p className="reference-loading" role="status"><LoaderCircle size={17} className="spin" /> Загружаем модельные значения…</p>}><ModelValuesPage /></Suspense></main>}
    <footer><a className="brand" href="#"><ScanFace size={19} /> FaceMetrics<span className="brand-ai">AI</span></a><p>Пропорции лица и подробный разбор оценки.</p><a href="#model-values">Модельные значения</a></footer>
    {camera && <Suspense fallback={<div className="modal-loading" role="status">Открываем камеру…</div>}><CameraModal onClose={() => setCamera(false)} onCapture={file => void loadPhoto(file)} /></Suspense>}
  </div>
}
