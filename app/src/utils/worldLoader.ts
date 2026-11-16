import worlds from 'virtual:worlds'
import { type World, type WorldEntry } from '../types/world'

export function loadWorlds(): WorldEntry[] {
  return worlds as WorldEntry[]
}

export async function fetchWorlds(): Promise<WorldEntry[]> {
  if (!import.meta.env.DEV) return loadWorlds()

  const response = await fetch('/__worlds', { cache: 'no-store' })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<WorldEntry[]>
}

function localWorldAssetUrl(url: string | undefined): string {
  return url?.startsWith('/worlds/') ? url : ''
}

export function getSplatUrl(world: World): string {
  const urls = world.assets.splats.spz_urls
  return localWorldAssetUrl(urls.full_res)
}

// A small LOD shown while the full splat streams in: Marble's 100k level is
// ~1.3 MB against ~29 MB, so the room appears in a second or two.
export function getSplatPreviewUrl(world: World): string {
  const urls = world.assets.splats.spz_urls
  const preview = localWorldAssetUrl(urls['100k'] ?? urls['150k'])
  return preview && preview !== getSplatUrl(world) ? preview : ''
}
