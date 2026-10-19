export function architectureName(name: string) {
  return name === 'coordinates' ? 'Регрессия координат' : name === 'heatmaps' ? 'Карты ориентиров' : name
}
