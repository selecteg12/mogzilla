// Face Mesh indices in source-image coordinates. Both sides are always included.
const EYES = [[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33], [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]]
const BROWS = [[70,63,105,66,107,55,65,52,53,46,70], [336,296,334,293,300,276,283,282,295,285,336]]

export function regionViewport(landmarks, regionId, width, height) {
  const core = {
    eyes:EYES.flat(), brows:[...BROWS.flat(),...EYES.flat()], upper:[10,109,338,168,2],
    midface:[234,454,468,473,0], nose_mouth:[168,98,327,61,291,0,17],
    lower:[172,397,152,0,17], skin:[205,425,151],
  }
  const selected = core[regionId]
  const points = landmarks.filter(p=>(!selected || selected.includes(p.id)) && Number.isFinite(p.x) && Number.isFinite(p.y))
  if (!points.length) return {x:0,y:0,width,height}
  const minX=Math.min(...points.map(p=>p.x)), maxX=Math.max(...points.map(p=>p.x))
  const minY=Math.min(...points.map(p=>p.y)), maxY=Math.max(...points.map(p=>p.y))
  const padding = regionId==='eyes' || regionId==='brows' ? 1.45 : 1.3
  const side = Math.max(48,Math.max(maxX-minX,maxY-minY)*padding)
  const cropWidth = Math.min(width,side), cropHeight = Math.min(height,side)
  return {x:Math.max(0,Math.min(width-cropWidth,(minX+maxX-cropWidth)/2)),y:Math.max(0,Math.min(height-cropHeight,(minY+maxY-cropHeight)/2)),width:cropWidth,height:cropHeight}
}

export function regionDrawing(result, region) {
  const points = new Map((result.landmarks ?? []).map(p => [p.id, p]))
  const metrics = (result.geometry?.metrics ?? []).filter(m => region.metrics.some(r => r.id === m.id))
  const paths = result.geometry?.paths ?? []
  const byRegion = {
    eyes: EYES, brows: [...BROWS, ...EYES], upper: [[109,10,338], [10,168,2]],
    midface: [paths[0] ?? [], ...EYES], nose_mouth: paths.slice(5,9),
    lower: [paths[0] ?? [], ...paths.slice(7,9)], symmetry: paths, skin: [],
  }
  const contours = (byRegion[region.id] ?? []).filter(path => path.every(id => points.has(id)))
  const ids = new Set(region.metrics.flatMap(m => m.point_ids))
  contours.flat().forEach(id => ids.add(id))
  const lines = []
  const add = (a, b, metric) => { if (a && b) lines.push({a, b, metric: metric.id, label: metric.title}) }
  const pair = (a, b, metric) => add(points.get(a), points.get(b), metric)
  const coordinate = value => Array.isArray(value) && value.length === 2 && value.every(Number.isFinite) ? {x:value[0], y:value[1]} : null
  for (const metric of metrics) {
    const s = metric.substitution
    if (typeof s.inner === 'number' && typeof s.outer === 'number') pair(s.inner, s.outer, metric)
    else if (metric.id.startsWith('brow_gap_')) add(coordinate(s.brow_center), coordinate(s.eye_center), metric)
    else if (metric.id.startsWith('eye_aspect_') && metric.id !== 'eye_aspect_difference') {
      const pairs = metric.id.endsWith('_0') ? [[33,133],[160,144],[158,153]] : [[362,263],[385,380],[387,373]]
      pairs.forEach(([a,b]) => pair(a,b,metric))
    } else if (metric.id === 'midface_pupils') {
      pair(468,473,metric)
      const a = points.get(468), b = points.get(473), lip = points.get(0), axis = result.geometry?.axis.direction
      if (a && b && lip && axis) {
        const center = {x:(a.x+b.x)/2, y:(a.y+b.y)/2}
        const distance = (lip.x-center.x)*axis[0] + (lip.y-center.y)*axis[1]
        add(center,{x:center.x+axis[0]*distance,y:center.y+axis[1]*distance},metric)
      }
    } else if (metric.id.startsWith('brow_arch_')) {
      const ends = metric.id.endsWith('_0') ? [70,107] : [336,300]
      pair(ends[0],ends[1],metric)
    } else if (metric.id === 'visible_upper_middle') {
      const axis = result.geometry?.axis
      if (axis) {
        const project = id => {
          const p = points.get(id)
          if (!p) return null
          const [ox,oy] = axis.origin, [dx,dy] = axis.direction
          const t = (p.x-ox)*dx+(p.y-oy)*dy
          return {x:ox+t*dx,y:oy+t*dy}
        }
        add(project(10),project(168),metric); add(project(168),project(2),metric)
      }
    } else if (metric.id === 'asymmetry') {
      const axis = result.geometry?.axis
      if (axis) for (const [left,right] of s.pairs ?? []) {
        const a = points.get(left), b = points.get(right)
        if (!a || !b) continue
        const [ox,oy] = axis.origin, [dx,dy] = axis.direction
        const projection = (a.x-ox)*dx + (a.y-oy)*dy
        add({x:2*(ox+projection*dx)-a.x,y:2*(oy+projection*dy)-a.y},b,metric)
      }
    } else if (metric.id === 'eye_spacing') {
      pair(133,362,metric); pair(33,133,metric); pair(362,263,metric)
    } else if (metric.point_ids.length === 4) {
      pair(metric.point_ids[0],metric.point_ids[1],metric)
      pair(metric.point_ids[2],metric.point_ids[3],metric)
    } else if (metric.point_ids.length === 3) {
      pair(metric.point_ids[0],metric.point_ids[1],metric)
      pair(metric.point_ids[1],metric.point_ids[2],metric)
    }
  }
  return {points:[...ids].flatMap(id => points.has(id) ? [points.get(id)] : []), contours:contours.map(path => path.map(id => points.get(id))), lines}
}
