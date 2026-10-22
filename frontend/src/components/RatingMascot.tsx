import laughing from '../assets/mogzilla-mascot.webp'
import calm from '../assets/mascot/calm.webp'
import respect from '../assets/mascot/respect.webp'
import shocked from '../assets/mascot/shocked.webp'
import { mascotMood } from '../mascotMood.mjs'

const moods = {
  laughing: {image:laughing, description:'Маскот mogzilla очень сильно смеётся', caption:'oops..'},
  calm: {image:calm, description:'Спокойный маскот mogzilla', caption:'mog in progress.'},
  respect: {image:respect, description:'Маскот mogzilla с широкой улыбкой и солнечными очками', caption:'respect the mog.'},
  shocked: {image:shocked, description:'Маскот mogzilla в полном изумлении', caption:'beyond mogging.'},
}

export default function RatingMascot({value}: {value:number}) {
  const mood = mascotMood(value), state = moods[mood]
  return <figure className="rating-mascot" data-mood={mood}><img src={state.image} width="640" height="640" alt={state.description}/><figcaption className="micro">{state.caption}</figcaption></figure>
}
