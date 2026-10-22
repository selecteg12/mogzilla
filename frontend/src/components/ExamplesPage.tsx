import { ArrowLeft, ArrowUpRight } from 'lucide-react'
import examplesData from '../data/examples.json'
import { ratingTier } from '../ratingTier.mjs'

const photos = import.meta.glob<string>('../assets/examples/*.{jpg,png}', { eager: true, query: '?url', import: 'default' })
const groups = [
  { id: 'highest', label: '01 / highest mogs', title: 'maxxed out.', description: 'Примеры наивысших оценок.' },
  { id: 'lowest', label: '02 / lowest rates', title: 'the other end.', description: 'Примеры низких оценок.' },
]

export default function ExamplesPage({ busy, onUpload }: { busy: boolean; onUpload: () => void }) {
  return <>
    <header className="examples-heading"><div><a className="examples-back" href="#"><ArrowLeft size={15} /> На главную</a><span className="micro">mogzilla / the mog scale</span><h1>the mog<br />spectrum.</h1></div><p>Реальные результаты mogzilla на выбранных фотографиях. Балл и mog-тир — от одного края шкалы до другого.</p></header>
    {groups.map(group => <section key={group.id} className="examples-group" aria-labelledby={'examples-' + group.id}>
      <div className="examples-group-heading"><div><span className="micro">{group.label}</span><h2 id={'examples-' + group.id}>{group.title}</h2></div><p>{group.description}</p></div>
      <div className="examples-grid">{examplesData.examples.filter(example => example.group === group.id).map((example, index) => {
        const image = photos['../assets/examples/' + example.image]
        const tier = ratingTier(example.score)
        return <figure className="example-portrait" key={example.id}><span className="micro">portrait / {String(index + 1).padStart(2, '0')}</span><a className="example-image" href={image} target="_blank" rel="noopener noreferrer" aria-label={'Открыть пример: ' + example.score.toFixed(1) + ' из 10, ' + (tier?.label ?? '')}><img src={image} width={example.width} height={example.height} alt={'Фотография с оценкой mogzilla ' + example.score.toFixed(1) + ' из 10'} loading={group.id === 'highest' && index === 0 ? 'eager' : 'lazy'} decoding="async" /><span className="example-open" aria-hidden="true"><ArrowUpRight size={18} /></span></a><figcaption><span className="example-tier">{tier?.label}</span><span className="example-score">{example.score.toFixed(1)}<small>/ 10</small></span></figcaption></figure>
      })}</div>
    </section>)}
    <details className="examples-note"><summary>О примерах</summary><p>Это результаты текущей версии на конкретных снимках. Другое освещение, ракурс и кадрирование могут изменить балл. Подборка включает фотографии из обучения модели; она иллюстрирует шкалу и не является независимым тестом точности. Одинаковые баллы не задают дополнительный порядок между фотографиями.</p></details>
    <div className="examples-action"><span className="micro">your turn. unleash the moggster.</span><button className="editorial-action" onClick={onUpload} disabled={busy} aria-label="узнай свой рейт — загрузить фото"><span>узнай свой рейт</span><ArrowUpRight size={60} strokeWidth={1.2} /></button></div>
  </>
}
