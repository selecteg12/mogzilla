import { useState } from 'react'
import { ArrowUpRight, Upload, X } from 'lucide-react'
import type { Analysis, Box } from '../types'
import FaceCanvas from './FaceCanvas'
import HolisticSummary from './HolisticSummary'
import RegionBreakdown from './RegionBreakdown'

export default function HolisticAnalysis({ photo, result, onMetric, onReplace, onReset }: {
  photo: { file: File; url: string; width: number; height: number }; result: Analysis;
  onMetric: (id: string) => void; onReplace: () => void; onReset: () => void;
}) {
  const [overlay, setOverlay] = useState(false)
  const rating = result.holistic
  if (!rating) return null
  const box: Box = { x: result.box[0], y: result.box[1], width: result.box[2], height: result.box[3] }
  return <>
    <div className="editorial-analysis" id="walkthrough">
      <div className="image-column"><div className="panel-toolbar"><span>{photo.file.name}</span><div><button className="icon-button" onClick={onReplace} aria-label="Выбрать другую фотографию"><Upload size={17} /></button><button className="icon-button" onClick={onReset} aria-label="Убрать фотографию"><X size={19} /></button></div></div>
        <FaceCanvas {...photo} box={box} onBox={() => {}} result={result} disabled zoom={false} mode={overlay ? 'points' : 'photo'} layers={{ landmarks: true, contours: true, measurements: false, original: true, symmetry: false }} comparison={0} selected={null} ratingParts={[]} activePart={null} allCriteria={false} onCriterion={() => {}} regionLabel={overlay ? 'Все найденные ориентиры лица' : 'Исходная фотография без изменения пропорций'} />
        <div className="photo-controls" aria-label="Отображение фотографии"><button aria-pressed={!overlay} onClick={() => setOverlay(false)}>Оригинал</button><button aria-pressed={overlay} onClick={() => setOverlay(true)}>Все точки <span>+</span></button><span className="photo-local">Фото не сохраняется</span></div>
      </div>
      <aside className="editorial-report"><HolisticSummary result={result} /><a className="report-breakdown-link" href="#face-study">Разобрать по зонам <ArrowUpRight size={17}/></a></aside>
    </div>
    <div id="face-study" className="section-anchor"><RegionBreakdown photo={photo} result={result} onMetric={onMetric}/></div>
    <details className="analysis-method-details"><summary>Как получен результат и что важно учитывать</summary><div className="analysis-notes"><article><h3>Общий рейт</h3><p>Рейт — прогноз по фотографии в шкале mogzilla. При подготовке использованы SCUT-FBP5500, фотографии из MEBeauty и дополнительные портреты с ручными оценками. Итоговый балл переводится в категории looksmaxing — от sub 3 до true adam.</p><p>Отложенные фотографии используются для проверки качества. Источники данных и результаты проверок доступны на странице метода.</p></article><article><h3>Границы анализа</h3><ul><li>Шкала mogzilla ориентирована на эстетику looksmaxing-сообщества. Представления о красоте субъективны.</li><li>Фон, волосы, свет, мимика, ракурс и ретушь могут менять результат.</li><li>Проверка новых людей и снимков ограничена; пересечения с исходными датасетами полностью не исключены.</li><li>Профиль и трёхмерная форма костей не оцениваются.</li></ul><a className="text-button" href="#model-values">О модели и шкале <ArrowUpRight size={14} /></a></article></div></details>
  </>
}
