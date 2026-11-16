import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useMemo, useRef } from 'react'
import { ThreeEvent, useFrame, useThree } from '@react-three/fiber'
import { ConvexHullCollider, CuboidCollider, RigidBody, TrimeshCollider, type RapierRigidBody } from '@react-three/rapier'
import * as THREE from 'three'
import { ConvexHull } from 'three/examples/jsm/math/ConvexHull.js'
import { ObjectRenderMode, type WorldObjectAsset, type WorldObjectPhysics } from '../../types/world'
import { useAudioStore } from '../../store/audio'
import { loadSfx, playSfxOnce, stopSfx } from '../audio/sfx'
import { useSceneObjectVisual } from './useSceneObjectVisual'

export const OBJECT_SCALE = 0.5
const OBJECT_AUTO_ROTATE_Y_SPEED = 0.35

const COLLIDER_WIREFRAME_COLOR = 0x00aaff
const SFX_REF_DISTANCE = 2
/** Static furniture collides against its own mesh, simplified to this grid. */
const STATIC_COLLIDER_CELL_M = 0.03

/**
 * Vertex clustering: snap every vertex to a grid, merge those sharing a cell and
 * drop the triangles that collapse. A 50k-triangle table becomes a few thousand
 * triangles that still follow its legs and top -- the raw mesh made each physics
 * step test thousands of triangles under every prop resting on it, and froze the
 * page for most of a minute.
 */
function clusterMesh(vertices: Float32Array, indices: Uint32Array, cell: number) {
  const cellOf = new Map<string, number>()
  const sums: number[] = []
  const counts: number[] = []
  const remap = new Uint32Array(vertices.length / 3)
  for (let i = 0; i < remap.length; i++) {
    const x = vertices[i * 3]
    const y = vertices[i * 3 + 1]
    const z = vertices[i * 3 + 2]
    const key = `${Math.round(x / cell)},${Math.round(y / cell)},${Math.round(z / cell)}`
    let id = cellOf.get(key)
    if (id === undefined) {
      id = counts.length
      cellOf.set(key, id)
      sums.push(0, 0, 0)
      counts.push(0)
    }
    sums[id * 3] += x
    sums[id * 3 + 1] += y
    sums[id * 3 + 2] += z
    counts[id] += 1
    remap[i] = id
  }
  const outVertices = new Float32Array(counts.length * 3)
  for (let id = 0; id < counts.length; id++) {
    outVertices[id * 3] = sums[id * 3] / counts[id]
    outVertices[id * 3 + 1] = sums[id * 3 + 1] / counts[id]
    outVertices[id * 3 + 2] = sums[id * 3 + 2] / counts[id]
  }
  const seen = new Set<string>()
  const outIndices: number[] = []
  for (let t = 0; t < indices.length; t += 3) {
    const a = remap[indices[t]]
    const b = remap[indices[t + 1]]
    const c = remap[indices[t + 2]]
    if (a === b || b === c || a === c) continue
    const key = [a, b, c].sort((m, n) => m - n).join(',')
    if (seen.has(key)) continue
    seen.add(key)
    outIndices.push(a, b, c)
  }
  return { vertices: outVertices, indices: new Uint32Array(outIndices) }
}

type PointerHandler = (event: ThreeEvent<PointerEvent>) => boolean
type HoverHandler = (event: ThreeEvent<PointerEvent>, objectId: string, hovering: boolean) => void
type ClickHandler = (worldPos: THREE.Vector3) => void

const _rotation = new THREE.Quaternion()
export const SCENE_OBJECT_INSTANCE_ID_KEY = 'sceneObjectInstanceId'

export interface SceneObjectHandle {
  id: string
  rigidBody: RapierRigidBody | null
  initialPosition: THREE.Vector3
  initialRotation: THREE.Quaternion
  bounds: THREE.Box3
  getFocusPoint: (target: THREE.Vector3) => THREE.Vector3
  playInteractionSfx: () => void
}

interface Props {
  object: WorldObjectAsset
  position: [number, number, number]
  rotation?: [number, number, number]
  scale?: [number, number, number]
  physics?: WorldObjectPhysics
  renderMode: ObjectRenderMode
  autoRotateY?: boolean
  onHover: HoverHandler
  onClick?: ClickHandler
  onPointerDown?: PointerHandler
  onPointerMove?: PointerHandler
  onPointerUp?: PointerHandler
  onPointerCancel?: PointerHandler
}

// Memoised: a hover or grab state change in ObjectGrid used to re-render every
// object in the room.
export const SceneObject = memo(forwardRef<SceneObjectHandle, Props>(function SceneObject(
  {
    object,
    position,
    rotation = [0, 0, 0],
    scale = [1, 1, 1],
    physics = 'rigidbody',
    renderMode,
    autoRotateY = false,
    onHover,
    onClick,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel,
  },
  ref,
) {
  const rigidBodyRef = useRef<RapierRigidBody>(null)
  const visualGroupRef = useRef<THREE.Group>(null)
  const colliderProxyRef = useRef<THREE.Mesh>(null)
  const activeSfx = useRef(new Set<THREE.PositionalAudio>())
  const camera = useThree((state) => state.camera)
  const lastSfxIndexRef = useRef<number | null>(null)
  const muted = useAudioStore((s) => s.muted)
  const isStatic = physics === 'static' || physics === 'ghost'
  const hasCollider = physics === 'rigidbody' || physics === 'static'
  const initialPosition = useMemo(() => new THREE.Vector3(...position), [position])
  const initialRotation = useMemo(() => new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation)), [rotation])
  const { scene, wireframeOverlayScene, offset, size, bounds, collider } = useSceneObjectVisual({
    asset: object,
    renderMode,
  })
  const colliderCenter = useMemo(
    () => new THREE.Vector3(0, (size.y * OBJECT_SCALE) / 2, 0),
    [size],
  )
  const colliderUserData = useMemo(() => ({
    [SCENE_OBJECT_INSTANCE_ID_KEY]: object.id,
  }), [object.id])
  const colliderHalfExtents = useMemo(
    () => new THREE.Vector3(
      Math.max((size.x * OBJECT_SCALE) / 2, 0.01),
      Math.max((size.y * OBJECT_SCALE) / 2, 0.01),
      Math.max((size.z * OBJECT_SCALE) / 2, 0.01),
    ),
    [size],
  )
  // Colliders shaped like the model, not its bounding box. A box around a vase's
  // leaves is 45 cm wide and knocked over whatever stood near it; a box around a
  // table filled the space under it. Props that move get the convex hull of their
  // mesh; furniture that does not move gets the mesh itself.
  const scaleVertices = useCallback((vertices: Float32Array) => {
    const sx = OBJECT_SCALE * scale[0]
    const sy = OBJECT_SCALE * scale[1]
    const sz = OBJECT_SCALE * scale[2]
    const out = new Float32Array(vertices.length)
    for (let i = 0; i < out.length; i += 3) {
      out[i] = vertices[i] * sx
      out[i + 1] = vertices[i + 1] * sy
      out[i + 2] = vertices[i + 2] * sz
    }
    return out
  }, [scale])
  // Raw vertices are only present for GLBs without precomputed colliders.
  const colliderVertices = useMemo(() => scaleVertices(collider.vertices), [collider, scaleVertices])
  const hasMeshCollider = colliderVertices.length >= 12 && collider.indices.length >= 3
  // Rapier's hull builder took tens of seconds on a raw 25k-50k-vertex mesh and
  // froze the page. three's QuickHull does it in milliseconds; Rapier then gets
  // only the hull's own vertices.
  const staticMesh = useMemo(() => {
    if (physics !== 'static') return null
    if (collider.mesh) return { vertices: scaleVertices(collider.mesh.vertices), indices: collider.mesh.indices }
    if (!hasMeshCollider) return null
    const mesh = clusterMesh(colliderVertices, collider.indices, STATIC_COLLIDER_CELL_M)
    return mesh.indices.length >= 3 ? mesh : null
  }, [collider, colliderVertices, hasMeshCollider, physics, scaleVertices])
  const hullPoints = useMemo(() => {
    if (physics !== 'rigidbody') return null
    if (collider.hull && collider.hull.length >= 12) return scaleVertices(collider.hull)
    if (!hasMeshCollider) return null
    const points: THREE.Vector3[] = []
    for (let i = 0; i < colliderVertices.length; i += 3) {
      points.push(new THREE.Vector3(colliderVertices[i], colliderVertices[i + 1], colliderVertices[i + 2]))
    }
    const hull = new ConvexHull().setFromPoints(points)
    const out: number[] = []
    for (const vertex of hull.vertices) out.push(vertex.point.x, vertex.point.y, vertex.point.z)
    return out.length >= 12 ? new Float32Array(out) : null
  }, [collider.hull, colliderVertices, hasMeshCollider, physics, scaleVertices])
  const colliderWireframeMaterial = useMemo(() => new THREE.MeshBasicMaterial({
    color: COLLIDER_WIREFRAME_COLOR,
    wireframe: true,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
    fog: false,
  }), [])

  useFrame((_, delta) => {
    if (!autoRotateY || !visualGroupRef.current) return
    visualGroupRef.current.rotation.y += delta * OBJECT_AUTO_ROTATE_Y_SPEED
  })

  useEffect(() => {
    colliderWireframeMaterial.opacity = 0
    colliderWireframeMaterial.transparent = true
    colliderWireframeMaterial.depthTest = false
    colliderWireframeMaterial.depthWrite = false
    colliderWireframeMaterial.needsUpdate = true
  }, [colliderWireframeMaterial])

  // Fetch and decode the sounds ahead of the first click, without mounting any.
  useEffect(() => {
    object.sfxUrls.forEach((url) => {
      loadSfx(url).catch(() => {})
    })
  }, [object.sfxUrls])

  useEffect(() => {
    if (muted) stopSfx(activeSfx.current)
  }, [muted])

  useEffect(() => {
    const active = activeSfx.current
    return () => stopSfx(active)
  }, [])

  const playRandomSfx = useCallback(() => {
    if (muted || object.sfxUrls.length === 0) return

    const lastIndex = lastSfxIndexRef.current
    let nextIndex = 0
    if (object.sfxUrls.length > 1) {
      nextIndex = Math.floor(Math.random() * (object.sfxUrls.length - 1))
      if (lastIndex !== null && nextIndex >= lastIndex) nextIndex += 1
    }

    const parent = visualGroupRef.current
    if (!parent) return

    lastSfxIndexRef.current = nextIndex
    stopSfx(activeSfx.current)
    playSfxOnce(object.sfxUrls[nextIndex], parent, camera, SFX_REF_DISTANCE, activeSfx.current).catch(() => {})
  }, [camera, muted, object.sfxUrls])

  useEffect(() => {
    const body = rigidBodyRef.current
    if (!body) return
    body.setTranslation({ x: position[0], y: position[1], z: position[2] }, true)
    _rotation.setFromEuler(new THREE.Euler(...rotation))
    body.setRotation({ x: _rotation.x, y: _rotation.y, z: _rotation.z, w: _rotation.w }, true)
    body.setLinvel({ x: 0, y: 0, z: 0 }, true)
    body.setAngvel({ x: 0, y: 0, z: 0 }, true)
    body.wakeUp()
  }, [position, rotation, scale])

  useEffect(() => {
    return () => {
      colliderWireframeMaterial.dispose()
    }
  }, [colliderWireframeMaterial])

  useImperativeHandle(
    ref,
    () => ({
      id: object.id,
      get rigidBody() {
        return rigidBodyRef.current
      },
      initialPosition,
      initialRotation,
      bounds,
      getFocusPoint: (target) => {
        if (colliderProxyRef.current) return colliderProxyRef.current.getWorldPosition(target)
        return target.copy(initialPosition).add(colliderCenter)
      },
      playInteractionSfx: playRandomSfx,
    }),
    [bounds, colliderCenter, initialPosition, initialRotation, object.id, playRandomSfx],
  )

  const visualContent = (
    <group ref={visualGroupRef} scale={[OBJECT_SCALE * scale[0], OBJECT_SCALE * scale[1], OBJECT_SCALE * scale[2]]}>
      <primitive object={scene} position={offset} dispose={null} />
      {renderMode === ObjectRenderMode.ShadedWireframe && (
        <primitive object={wireframeOverlayScene} position={offset} dispose={null} />
      )}
    </group>
  )

  return (
    <RigidBody
      ref={rigidBodyRef}
      type={isStatic ? 'fixed' : 'dynamic'}
      colliders={false}
      position={position}
      rotation={rotation}
      linearDamping={0.45}
      angularDamping={0.35}
      additionalSolverIterations={4}
      ccd
      canSleep
    >
      {hasCollider && hullPoints && (
        <ConvexHullCollider args={[hullPoints]} />
      )}
      {hasCollider && staticMesh && (
        <TrimeshCollider args={[staticMesh.vertices, staticMesh.indices]} />
      )}
      {hasCollider && !hullPoints && !staticMesh && (
        <CuboidCollider
          args={[
            colliderHalfExtents.x * scale[0],
            colliderHalfExtents.y * scale[1],
            colliderHalfExtents.z * scale[2],
          ]}
          position={[colliderCenter.x * scale[0], colliderCenter.y * scale[1], colliderCenter.z * scale[2]]}
        />
      )}
      <mesh
        ref={colliderProxyRef}
        position={[colliderCenter.x * scale[0], colliderCenter.y * scale[1], colliderCenter.z * scale[2]]}
        material={colliderWireframeMaterial}
        renderOrder={10000}
        userData={colliderUserData}
        onPointerOver={(event) => {
          event.stopPropagation()
          onHover(event, object.id, true)
        }}
        onPointerOut={(event) => {
          event.stopPropagation()
          onHover(event, object.id, false)
        }}
        onClick={(event) => {
          event.stopPropagation()
          onClick?.(event.point.clone())
        }}
        onPointerDown={(event) => {
          if (event.button !== 0) return
          event.stopPropagation()
          if (!onPointerDown?.(event)) playRandomSfx()
        }}
        onPointerMove={(event) => {
          onHover(event, object.id, true)
          onPointerMove?.(event)
        }}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
      >
        <boxGeometry args={[
          colliderHalfExtents.x * scale[0] * 2,
          colliderHalfExtents.y * scale[1] * 2,
          colliderHalfExtents.z * scale[2] * 2,
        ]} />
      </mesh>
      {visualContent}
    </RigidBody>
  )
}))
