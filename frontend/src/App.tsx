import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUpRight, Camera, ChevronRight, CircleHelp, Download, Focus, LoaderCircle, Plus, Upload, X } from 'lucide-react'
import { fetchJson, localizePhoto, normalizePhoto, processPhoto } from './api'
import type { Analysis, Box, ModelStatus } from './types'
import FaceCanvas from './components/FaceCanvas'
import type { Layers } from './components/FaceCanvas'
import MetricCard from './components/MetricCard'
import HolisticAnalysis from './components/HolisticAnalysis'
import MogzillaHero from './components/MogzillaHero'
import SectionBoundary from './components/SectionBoundary'

const CameraModal = lazy(() => import('./components/CameraModal'))
const MethodPage = lazy(() => import('./components/MethodPage'))
const ExamplesPage = lazy(() => import('./components/ExamplesPage'))
const SOURCE_LAYERS: Layers = { landmarks: false, contours: false, measurements: false, symmetry: false, original: true }
type Page = 'home' | 'analysis' | 'method' | 'examples'
const pageForHash = (): Page => window.location.hash === '#model-values' ? 'method' : window.location.hash === '#examples' ? 'examples' : ['#workspace', '#results', '#walkthrough', '#face-study'].includes(window.location.hash) ? 'analysis' : 'home'

export default function App() {
  const [page, setPage] = useState<Page>(pageForHash)
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
  const input = useRef<HTMLInputElement>(null)
  const controller = useRef<AbortController | null>(null)
  const photoVersion = useRef(0)

  useEffect(() => {
    const followHash = () => setPage(pageForHash())
    window.addEventListener('hashchange', followHash)
    window.addEventListener('popstate', followHash)
    return () => { window.removeEventListener('hashchange', followHash); window.removeEventListener('popstate', followHash) }
  }, [])
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const id = window.location.hash.slice(1)
      if (id && id !== 'model-values' && id !== 'examples') document.getElementById(id)?.scrollIntoView({ behavior: 'instant' })
      else window.scrollTo({ top: 0, behavior: 'instant' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [page])
  const refresh = useCallback(async () => {
    try { setModel(await fetchJson<ModelStatus>('/api/holistic/model')); setBackendError(false) }
    catch { setBackendError(true) }
  }, [])
  useEffect(() => { void refresh(); const timer = window.setInterval(() => { void refresh() }, 15000); return () => window.clearInterval(timer) }, [refresh])
  useEffect(() => { const url = photo?.url; return () => { if (url) URL.revokeObjectURL(url) } }, [photo?.url])
  useEffect(() => () => { controller.current?.abort(); photoVersion.current++ }, [])

  const openAnalysis = () => {
    window.history.pushState(null, '', '#workspace'); setPage('analysis')
    window.requestAnimationFrame(() => document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' }))
  }
  const loadPhoto = async (file: File) => {
    const version = ++photoVersion.current
    controller.current?.abort(); setBusy(true); setError(''); setResult(null); setBox(null); setSelectedId('')
    const request = new AbortController(); controller.current = request
    try {
      const normalized = await normalizePhoto(file)
      if (version !== photoVersion.current) return
      setPhoto({ ...normalized, url: URL.createObjectURL(normalized.file) }); openAnalysis()
      const found = await localizePhoto(normalized.file, request.signal)
      if (version !== photoVersion.current || request.signal.aborted) return
      if (found.analysis) {
        setResult(found.analysis)
        const [x, y, width, height] = found.analysis.box
        setBox({ x, y, width, height })
      } else setError(found.localization.message ?? 'Не удалось надёжно найти лицо. Выберите чёткий портрет или обведите лицо на снимке.')
    } catch (e) { if (version === photoVersion.current) setError(e instanceof Error ? e.message : 'Не удалось прочитать изображение.') }
    finally { if (version === photoVersion.current) setBusy(false) }
  }
  const changeBox = async (region: Box) => {
    if (!photo) return
    controller.current?.abort(); photoVersion.current++; setBox(region); setSelectedId('')
    const request = new AbortController(); controller.current = request
    setBusy(true); setError(''); setResult(null)
    try { const data = await processPhoto(photo.file, region, true, request.signal); if (!request.signal.aborted) setResult(data) }
    catch (e) { if (!request.signal.aborted) setError(e instanceof Error ? e.message : 'Ошибка анализа. Попробуйте ещё раз.') }
    finally { if (controller.current === request) setBusy(false) }
  }
  const reset = () => { controller.current?.abort(); photoVersion.current++; setPhoto(null); setBox(null); setResult(null); setError(''); setSelectedId(''); setBusy(false) }
  const selectMetric = (id: string) => { setSelectedId(id); document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }) }
  const exportMeasurements = () => {
    if (!result) return
    const { preprocessed_image: _image, heatmap_image: _heatmap, visual_model_image: _visual, ...measurements } = result
    const url = URL.createObjectURL(new Blob([JSON.stringify(measurements, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = 'mogzilla-measurements.json'; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const metrics = result?.geometry?.metrics ?? []
  const warnings = [...new Set([...(result?.quality.warnings ?? []), ...(result?.geometry?.warnings ?? [])])]
  const upload = () => input.current?.click()

  return <div className="mogzilla-app" onClickCapture={event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    const link = (event.target as Element).closest<HTMLAnchorElement>('a[href^="#"]')
    if (!link) return
    event.preventDefault()
    const hash = link.hash
    if (window.location.hash !== hash) window.history.pushState(null, '', hash || '#')
    setPage(pageForHash())
    window.requestAnimationFrame(() => {
      if (hash && hash !== '#model-values' && hash !== '#examples') document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'instant' })
      else window.scrollTo({ top: 0, behavior: 'instant' })
    })
  }}>
    <a className="skip-link" href="#workspace">Перейти к анализу</a>
    <header className="site-header"><a className="brand" href="#" aria-label="mogzilla — главная">mogzilla<span className="brand-period">.</span></a><nav aria-label="Главное меню"><a href="#" aria-current={page === 'home' ? 'page' : undefined}>Главная</a><a href="#workspace" aria-current={page === 'analysis' ? 'page' : undefined}>Анализ</a><a href="#examples" aria-current={page === 'examples' ? 'page' : undefined}>Примеры</a><a href="#model-values" aria-current={page === 'method' ? 'page' : undefined}>Метод <span>+</span></a></nav></header>
    <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" aria-label="Загрузить фотографию лица" onChange={e => { const file = e.target.files?.[0]; if (file) void loadPhoto(file); e.target.value = '' }} />
    {(error || backendError) && <div className="app-alerts">{error && <div className="error-banner" role="alert"><CircleHelp size={18} /><span>{error}</span><button className="icon-button" onClick={() => setError('')} aria-label="Закрыть сообщение"><X size={18} /></button></div>}{backendError && <div className="error-banner" role="status"><span>Не удалось подключиться к модели.</span><button className="text-button" onClick={() => void refresh()}>Попробовать снова</button></div>}</div>}

    <main hidden={page !== 'home'} className={`home-main ${dragging ? 'drag-active' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false) }} onDrop={event => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files[0]; if (file) void loadPhoto(file) }}>
      <MogzillaHero busy={busy} onUpload={upload} onCamera={() => setCamera(true)} />
      <section id="photo-tips" className="photo-tips section-anchor"><div><span className="micro">before the mog check</span><h2>better pic.<br />better mog.</h2></div><div className="glossary"><details><summary>Какое фото подойдёт?<ChevronRight size={16} /></summary><p>Смотрите прямо в камеру, держите голову ровно. Нужны открытые глаза и равномерный свет. Волосы, очки и руки не должны закрывать лицо. JPG, PNG или WebP, до 10 МБ.</p></details><details><summary>Что означает балл?<ChevronRight size={16} /></summary><p>Это твой рейт по шкале mogzilla и категория в системе looksmaxing — от sub 3 до true adam. Модель анализирует фотографию и прогнозирует балл в шкале, основанной на человеческих оценках внешности. Категория показывает, в какой mog-тир попал этот снимок. Свет и ракурс могут менять результат.</p></details><details><summary>Почему на разных фото балл отличается?<ChevronRight size={16} /></summary><p>Свет, ракурс, выражение лица, ретушь и кадрирование влияют на прогноз. Сравнивайте несколько портретов с похожими условиями съёмки.</p></details></div></section>
    </main>

    <main hidden={page !== 'analysis'} className="analysis-main">
      <section id="workspace" className="workspace section-anchor">
        <div className="analysis-heading"><h1>{photo ? <>Портрет.<br />Разбор в деталях.</> : <>Начнём<br />с портрета.</>}</h1><div><span className="service-status"><span className={`status-dot ${model?.ready && !backendError ? 'ready' : ''}`} />{backendError ? 'Нет соединения' : model?.ready ? 'Модель готова' : model ? 'Модель недоступна' : 'Подключаем модель…'}</span></div></div>
        {!photo ? <div className={`upload-area ${dragging ? 'drag-active' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); const file = e.dataTransfer.files[0]; if (file) void loadPhoto(file) }}>
          <div className="upload-aside"><span>Фото для анализа</span><p>Расчёт начнётся автоматически.<br/>Фото не сохраняется.</p></div>
          <div className="upload-content"><h2>Перетащи портрет сюда</h2><p>или выбери снимок с устройства.</p>
            <div className="upload-file-strip"><span>Файл не выбран</span><button className="upload-file-button" disabled={busy} onClick={upload}><span>Выбрать фото</span>{busy ? <LoaderCircle className="spin" size={17}/> : <Plus size={18} strokeWidth={1.4}/>}</button></div>
            <div className="upload-options"><span>JPG, PNG, WebP · до 10 МБ</span><button className="text-button" onClick={() => setCamera(true)} disabled={busy}><Camera size={14}/> Сделать снимок</button></div>
          </div>
        </div>
          : result?.holistic ? <SectionBoundary><HolisticAnalysis key={`${photoVersion.current}-${result.total_milliseconds}`} photo={photo} result={result} onMetric={selectMetric} onReplace={upload} onReset={reset} /></SectionBoundary>
          : <div className="pending-analysis"><div className="image-panel"><div className="panel-toolbar"><span>{photo.file.name}</span><div><button className="icon-button" onClick={upload} disabled={busy} aria-label="Выбрать другую фотографию"><Upload size={17} /></button><button className="icon-button" onClick={reset} aria-label="Убрать фотографию"><X size={18} /></button></div></div><FaceCanvas url={photo.url} width={photo.width} height={photo.height} box={box} onBox={changeBox} result={result} layers={SOURCE_LAYERS} comparison={0} selected={null} disabled={busy} zoom={false} mode="photo" ratingParts={[]} activePart={null} allCriteria={false} onCriterion={() => {}} /></div><aside className="pending-copy"><span className="micro">Анализ фотографии</span><h2>{busy ? 'Рассматриваем детали.' : error ? 'Нужен другой кадр.' : 'Выдели лицо.'}</h2><p>{busy ? 'Находим лицо, проверяем качество снимка и рассчитываем рейт.' : 'Можно загрузить другой портрет или обвести лицо на фото: оба глаза, нос, рот и подбородок.'}</p>{busy && <div className="analysis-progress" role="status"><LoaderCircle className="spin" size={18} /> Выполняется анализ…</div>}{!busy && <button className="button secondary" onClick={upload}>Выбрать другое фото <ArrowUpRight size={17} /></button>}<p className="fine-print"><Focus size={13} /> Фото используется для расчёта и не сохраняется.</p></aside></div>}
      </section>
      {result && <section className="results-section section-anchor" id="results"><div className="section-heading"><div><span className="micro">Снимок в цифрах</span><h2>Измерения и расчёты</h2></div>{result.landmarks && <button className="button secondary small" onClick={exportMeasurements}><Download size={15} /> Сохранить расчёты</button>}</div>{warnings.length > 0 && <div className="warning-box"><CircleHelp size={19} /><div><strong>Что могло повлиять на результат</strong>{warnings.map(warning => <p key={warning}>{warning}</p>)}</div></div>}{result.geometry && <details className="all-measurements" open={!!selectedId}><summary>Все измерения ({metrics.length}), формулы и координаты</summary><p>Измерения описывают проекцию на фотографии и зависят от ракурса и точности ориентиров.</p><div className="metric-grid">{metrics.map(metric => <MetricCard key={metric.id} metric={metric} active={metric.id === selectedId} onSelect={() => selectMetric(metric.id)} error={result.reliability?.geometry_error[metric.id]?.mean_absolute_error} />)}</div></details>}</section>}
    </main>

    {page === 'examples' && <main className="examples-main"><SectionBoundary><Suspense fallback={<p className="reference-loading" role="status"><LoaderCircle size={17} className="spin" /> Загружаем примеры…</p>}><ExamplesPage busy={busy} onUpload={upload} /></Suspense></SectionBoundary></main>}
    {page === 'method' && <main className="method-main"><SectionBoundary><Suspense fallback={<p className="reference-loading" role="status"><LoaderCircle size={17} className="spin" /> Загружаем описание метода…</p>}><MethodPage /></Suspense></SectionBoundary></main>}
    <footer className="site-footer"><a href="#" className="footer-wordmark">mogzilla.</a><span>oops.. i mogged you again</span><div><a href="#photo-tips">Советы для фото ↗</a><a href="#model-values">О модели и шкале ↗</a></div></footer>
    {camera && <SectionBoundary><Suspense fallback={<div className="modal-loading" role="status">Открываем камеру…</div>}><CameraModal onClose={() => setCamera(false)} onCapture={file => void loadPhoto(file)} /></Suspense></SectionBoundary>}
  </div>
}
