import { Component } from 'react'
import type { ReactNode } from 'react'

// Keep the analysis usable if a lazy section cannot load after a rebuild or
// an interrupted connection. A refresh is an explicit user action.
export default class SectionBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    if (this.state.failed) return <div className="error-banner" role="alert"><span>Не удалось загрузить раздел. Обновите страницу и попробуйте снова.</span><button className="text-button" onClick={() => window.location.reload()}>Обновить страницу</button></div>
    return this.props.children
  }
}
