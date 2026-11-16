import * as THREE from 'three'

// One listener for every object sound effect.
//
// drei's <PositionalAudio> creates its own AudioListener per sound and adds it to
// the camera. With ~4 impact sounds on each of ~20 objects that was 77 listeners,
// each pushing nine AudioParam ramps to the audio thread every frame -- whether
// anything was playing or not. Sounds are now created only while they play.

let listener: THREE.AudioListener | null = null

export function sharedSfxListener(camera: THREE.Camera) {
  if (!listener) listener = new THREE.AudioListener()
  if (listener.parent !== camera) camera.add(listener)
  return listener
}

const buffers = new Map<string, Promise<AudioBuffer>>()
const loader = new THREE.AudioLoader()

export function loadSfx(url: string) {
  let buffer = buffers.get(url)
  if (!buffer) {
    buffer = loader.loadAsync(url)
    buffers.set(url, buffer)
    buffer.catch((error) => {
      buffers.delete(url)
      console.warn(`Could not load sound "${url}".`, error)
    })
  }
  return buffer
}

/** Play `url` once, positioned at `parent`; the sound removes itself when done. */
export async function playSfxOnce(
  url: string,
  parent: THREE.Object3D,
  camera: THREE.Camera,
  refDistance: number,
  active: Set<THREE.PositionalAudio>,
) {
  const buffer = await loadSfx(url)
  const sound = new THREE.PositionalAudio(sharedSfxListener(camera))
  sound.setBuffer(buffer)
  sound.setRefDistance(refDistance)
  sound.setLoop(false)
  sound.onEnded = () => {
    sound.isPlaying = false
    active.delete(sound)
    parent.remove(sound)
    try {
      sound.disconnect()
    } catch {
      // Already disconnected.
    }
  }
  parent.add(sound)
  active.add(sound)
  if (sound.context.state === 'suspended') await sound.context.resume().catch(() => {})
  sound.play()
}

export function stopSfx(active: Set<THREE.PositionalAudio>) {
  for (const sound of [...active]) {
    if (sound.isPlaying) sound.stop()
    sound.onEnded()
  }
}
