import { create } from 'zustand'

// What the splat loader has reported for the world being shown. Objects, the
// collider, audio and the environment go through three's DefaultLoadingManager
// and are read with drei's useProgress; the splats also report bytes, which is
// the only real progress signal for the largest file.
interface LoadingState {
  splatUrl: string
  previewReady: boolean
  fullReady: boolean
  fullLoaded: number
  fullTotal: number
  beginSplat: (url: string) => void
  setPreviewReady: (url: string) => void
  setFullProgress: (url: string, loaded: number, total: number) => void
  setFullReady: (url: string) => void
}

export const useLoadingStore = create<LoadingState>((set, get) => ({
  splatUrl: '',
  previewReady: false,
  fullReady: false,
  fullLoaded: 0,
  fullTotal: 0,
  beginSplat: (url) => set({ splatUrl: url, previewReady: false, fullReady: false, fullLoaded: 0, fullTotal: 0 }),
  setPreviewReady: (url) => {
    if (get().splatUrl === url) set({ previewReady: true })
  },
  setFullProgress: (url, loaded, total) => {
    if (get().splatUrl === url) set({ fullLoaded: loaded, fullTotal: total })
  },
  setFullReady: (url) => {
    if (get().splatUrl === url) set({ fullReady: true, previewReady: true })
  },
}))
