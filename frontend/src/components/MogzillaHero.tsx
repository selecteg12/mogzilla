import { ArrowDown, ArrowUpRight, Camera } from 'lucide-react'
import mascot from '../assets/mogzilla-mascot.webp'

export default function MogzillaHero({ busy, onUpload, onCamera }: { busy: boolean; onUpload: () => void; onCamera: () => void }) {
  return <>
    <div className="mog-poster" aria-label="mogzilla — oops.. i mogged you again">
      <div className="poster-grid" aria-hidden="true" />
      <span className="poster-label">mogzilla / mog check</span>
      <span className="poster-edition">looksmaxxing / mog scale</span>
      <a className="poster-phrase" href="#examples" aria-label="oops.. i mogged you again — примеры высоких и низких оценок">oops..<br />i mogged you<br />again<span>↗</span></a>
      <img className="mog-mascot" src={mascot} width="1254" height="1254" alt="Жёлтый глянцевый маскот с выразительными бровями и синими стеклянными слезами" fetchPriority="high" />
      <span className="poster-caption" aria-hidden="true">good mog</span>
    </div>
    <section className="home-action-row" aria-labelledby="home-title"><h1 className="sr-only" id="home-title">Узнай свой рейт в mogzilla</h1><div className="home-action-topline"><span className="micro">unleash the moggster</span><span className="micro">rate. rank. mog.</span></div><button className="editorial-action" disabled={busy} onClick={onUpload} aria-label="узнай свой рейт — загрузить фото"><span>узнай свой рейт</span><ArrowUpRight size={76} strokeWidth={1.2} /></button><div className="home-action-caption"><span>Загрузи портрет · JPG, PNG, WebP</span><button className="text-button" onClick={onCamera} disabled={busy}><Camera size={14} /> Сделать снимок</button></div></section>
    <div className="home-about"><span className="micro">find your mog tier <ArrowDown size={15} /></span><p>Загрузи фото — получи рейт от sub 3 до true adam.</p></div>
  </>
}
