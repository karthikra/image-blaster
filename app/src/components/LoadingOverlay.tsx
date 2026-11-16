import { useEffect, useRef, useState } from 'react'
import { useProgress } from '@react-three/drei'
import { useLoadingStore } from '../store/loading'
import './LoadingOverlay.css'

interface Props {
  title: string
  hasSplat: boolean
  hasPreview: boolean
}

// The room assembles one face at a time as loading advances.
const FACES = [
  { name: 'floor', at: 0 },
  { name: 'back', at: 0.18 },
  { name: 'left', at: 0.36 },
  { name: 'right', at: 0.54 },
  { name: 'roof-left', at: 0.72 },
  { name: 'roof-right', at: 0.86 },
] as const

const DONE_SETTLE_MS = 450
const FADE_MS = 700

function megabytes(bytes: number) {
  return (bytes / 1e6).toFixed(1)
}

function fileName(url: string) {
  const name = url.split('?')[0].split('/').pop() ?? ''
  return name.replace(/^\d+-/, '')
}

export function LoadingOverlay({ title, hasSplat, hasPreview }: Props) {
  const { active, loaded, total, item } = useProgress()
  const previewReady = useLoadingStore((s) => s.previewReady)
  const fullReady = useLoadingStore((s) => s.fullReady)
  const fullLoaded = useLoadingStore((s) => s.fullLoaded)
  const fullTotal = useLoadingStore((s) => s.fullTotal)

  // drei's counters are cumulative for the session; count from this world on.
  const base = useRef({ loaded, total })
  const items = Math.max(0, total - base.current.total)
  const itemsDone = Math.min(items, Math.max(0, loaded - base.current.loaded))

  const splatFraction = fullReady ? 1 : fullTotal > 0 ? fullLoaded / fullTotal : 0
  const itemFraction = items > 0 ? itemsDone / items : 0
  const worldVisible = !hasSplat || previewReady || fullReady
  // Two phases: up to 20% while the preview loads (only the code, environment
  // and preview are in flight), then the full splat and the objects.
  const rawProgress = !hasSplat
    ? itemFraction
    : !worldVisible
      ? 0.2 * itemFraction
      : 0.2 + 0.5 * splatFraction + 0.3 * itemFraction
  // Objects only start once the preview is up, so the item total grows mid-load
  // and the raw fraction can dip. Never show it going backwards.
  const shown = useRef(0)
  shown.current = Math.max(shown.current, rawProgress)
  const progress = shown.current
  const everythingLoaded = (!hasSplat || fullReady) && !active && itemsDone >= items

  const [done, setDone] = useState(false)
  const [gone, setGone] = useState(false)
  useEffect(() => {
    if (!everythingLoaded) {
      setDone(false)
      return
    }
    const settle = window.setTimeout(() => setDone(true), DONE_SETTLE_MS)
    return () => window.clearTimeout(settle)
  }, [everythingLoaded])
  useEffect(() => {
    if (!done) return
    const fade = window.setTimeout(() => setGone(true), FADE_MS)
    return () => window.clearTimeout(fade)
  }, [done])

  if (gone) return null

  const percent = Math.round(Math.min(1, progress) * 100)
  const splatLine = hasSplat
    ? fullReady
      ? 'splats ready'
      : fullTotal > 0
        ? `splats ${megabytes(fullLoaded)} / ${megabytes(fullTotal)} MB`
        : worldVisible ? 'splats starting' : 'loading a quick preview'
    : ''
  const itemLine = items > 0 ? `assets ${itemsDone} / ${items}` : ''
  const current = active && item ? fileName(item) : ''
  const screenState = worldVisible ? 'is-hidden' : ''
  const pillState = !worldVisible || done ? 'is-hidden' : ''

  return (
    <>
      <div className={`ld-screen ${screenState}`} aria-live="polite" aria-busy={!worldVisible}>
        <div className="ld-dust" />
        <div className="ld-stage">
          <div className="ld-room" style={{ ['--p' as string]: progress }}>
            {FACES.map((face) => (
              <div key={face.name} className={`ld-face ld-${face.name} ${progress >= face.at ? 'is-on' : ''}`}>
                {face.name === 'back' && <span className="ld-window" />}
              </div>
            ))}
          </div>
          <div className="ld-floor-glow" />
        </div>
        <div className="ld-copy">
          <p className="ld-eyebrow">Building the world</p>
          <h2 className="ld-title">{title}</h2>
          <div className="ld-meter">
            <span className="ld-percent">{percent}<small>%</small></span>
            <div className="ld-bar"><span style={{ width: `${percent}%` }} /></div>
          </div>
          <p className="ld-detail">
            {[splatLine, itemLine].filter(Boolean).join('  ·  ')}
            {hasPreview && !previewReady && <span className="ld-hint">the room appears first, detail streams in after</span>}
          </p>
          <p className="ld-item">{current || ' '}</p>
        </div>
      </div>
      <div className={`ld-pill ${pillState}`} role="status">
        <span className="ld-mini" aria-hidden="true"><i /><i /><i /></span>
        <span className="ld-pill-text">
          {fullReady ? 'Placing objects' : 'Refining splats'}
          <b>{percent}%</b>
        </span>
        <span className="ld-pill-bar"><span style={{ width: `${percent}%` }} /></span>
      </div>
    </>
  )
}
