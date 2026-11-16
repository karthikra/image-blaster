import { useEffect, useMemo } from 'react'
import { useLoader } from '@react-three/fiber'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { clone as cloneSkeleton } from 'three/examples/jsm/utils/SkeletonUtils.js'
import { ObjectRenderMode, type WorldObjectAsset } from '../../types/world'
import { SHADED_COLOR, useAssetMaterials } from './useAssetMaterials'

interface MeshMaterialState {
  mesh: THREE.Mesh
  litMaterials: THREE.Material | THREE.Material[]
}

interface SceneObjectVisualArgs {
  asset: WorldObjectAsset
  renderMode: ObjectRenderMode
}

const ignoreRaycast: THREE.Object3D['raycast'] = () => {}

export interface ColliderGeometry {
  /** Every vertex in the visual group's frame (offset applied, before scale). */
  vertices: Float32Array
  indices: Uint32Array
  /** Precomputed by compress_objects.py, when the GLB carries them. */
  hull?: Float32Array
  mesh?: { vertices: Float32Array; indices: Uint32Array }
}

const HULL_NODE = 'collider-hull'
const MESH_NODE = 'collider-mesh'

/** Lift a precomputed collision node out of the scene so it is never drawn. */
function takeColliderNode(root: THREE.Object3D, name: string) {
  const node = root.getObjectByName(name)
  if (!node) return null
  root.updateMatrixWorld(true)
  node.updateMatrixWorld(true)
  let geometry: THREE.BufferGeometry | null = null
  node.traverse((child) => {
    if (!geometry && (child instanceof THREE.Mesh || child instanceof THREE.Points)) {
      geometry = child.geometry.clone().applyMatrix4(child.matrixWorld)
    }
  })
  node.removeFromParent()
  return geometry as THREE.BufferGeometry | null
}

function offsetPositions(geometry: THREE.BufferGeometry, offset: THREE.Vector3) {
  const position = geometry.getAttribute('position') as THREE.BufferAttribute
  const out = new Float32Array(position.count * 3)
  for (let i = 0; i < position.count; i++) {
    out[i * 3] = position.getX(i) + offset.x
    out[i * 3 + 1] = position.getY(i) + offset.y
    out[i * 3 + 2] = position.getZ(i) + offset.z
  }
  return out
}

const _vertex = new THREE.Vector3()

/** The model's own triangles, for colliders shaped like it rather than its box. */
function colliderGeometry(root: THREE.Object3D, offset: THREE.Vector3): ColliderGeometry {
  root.updateMatrixWorld(true)
  const vertices: number[] = []
  const indices: number[] = []
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return
    const position = child.geometry.getAttribute('position') as THREE.BufferAttribute | undefined
    if (!position) return
    const base = vertices.length / 3
    for (let i = 0; i < position.count; i++) {
      _vertex.fromBufferAttribute(position, i).applyMatrix4(child.matrixWorld).add(offset)
      vertices.push(_vertex.x, _vertex.y, _vertex.z)
    }
    const index = child.geometry.getIndex()
    if (index) {
      for (let i = 0; i < index.count; i++) indices.push(base + index.getX(i))
    } else {
      for (let i = 0; i < position.count; i++) indices.push(base + i)
    }
  })
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) }
}

function cloneMaterial(material: THREE.Material | THREE.Material[]): THREE.Material | THREE.Material[] {
  if (Array.isArray(material)) return material.map((m) => m.clone())
  return material.clone()
}

function asMaterialArray(material: THREE.Material | THREE.Material[]) {
  return Array.isArray(material) ? material : [material]
}

export function useSceneObjectVisual({ asset, renderMode }: SceneObjectVisualArgs) {
  const gltf = useLoader(GLTFLoader, asset.url)
  const { wireframeMaterial, shadedMaterial, wireframeOverlayMaterial } = useAssetMaterials()

  const { scene, wireframeOverlayScene, offset, size, bounds, materialStates, collider } = useMemo(() => {
    const clonedScene = cloneSkeleton(gltf.scene)
    const overlayScene = cloneSkeleton(gltf.scene)
    const hullGeometry = takeColliderNode(clonedScene, HULL_NODE)
    const meshGeometry = takeColliderNode(clonedScene, MESH_NODE)
    takeColliderNode(overlayScene, HULL_NODE)
    takeColliderNode(overlayScene, MESH_NODE)
    const states: MeshMaterialState[] = []

    clonedScene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      child.castShadow = true
      child.receiveShadow = true
      child.raycast = ignoreRaycast

      const litMaterials = cloneMaterial(child.material)
      child.material = litMaterials
      states.push({ mesh: child, litMaterials })
    })

    const box = new THREE.Box3().setFromObject(clonedScene)
    const center = new THREE.Vector3()
    const size = new THREE.Vector3()
    box.getCenter(center)
    box.getSize(size)

    overlayScene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      child.raycast = ignoreRaycast
    })

    const offset = new THREE.Vector3(-center.x, -box.min.y, -center.z)
    return {
      scene: clonedScene,
      wireframeOverlayScene: overlayScene,
      offset,
      collider: {
        ...(hullGeometry || meshGeometry
          ? { vertices: new Float32Array(), indices: new Uint32Array() }
          : colliderGeometry(clonedScene, offset)),
        ...(hullGeometry ? { hull: offsetPositions(hullGeometry, offset) } : {}),
        ...(meshGeometry && meshGeometry.getIndex()
          ? { mesh: { vertices: offsetPositions(meshGeometry, offset), indices: new Uint32Array(meshGeometry.getIndex()!.array) } }
          : {}),
      } as ColliderGeometry,
      size,
      bounds: box.clone(),
      materialStates: states,
    }
  }, [gltf.scene])

  useEffect(() => {
    wireframeOverlayScene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      child.material = wireframeOverlayMaterial
      child.renderOrder = 1
      child.raycast = ignoreRaycast
    })
  }, [wireframeOverlayMaterial, wireframeOverlayScene])

  useEffect(() => {
    if (renderMode === ObjectRenderMode.ShadedWireframe) {
      shadedMaterial.color.copy(SHADED_COLOR)
    }

    for (const state of materialStates) {
      if (renderMode === ObjectRenderMode.Wireframe) {
        state.mesh.material = wireframeMaterial
        continue
      }

      if (renderMode === ObjectRenderMode.ShadedWireframe) {
        state.mesh.material = shadedMaterial
        continue
      }

      state.mesh.material = state.litMaterials
    }
  }, [materialStates, renderMode, shadedMaterial, wireframeMaterial])

  useEffect(() => {
    return () => {
      for (const state of materialStates) {
        for (const material of asMaterialArray(state.litMaterials)) {
          material.dispose()
        }
      }
    }
  }, [materialStates])

  return { scene, wireframeOverlayScene, offset, size, bounds, collider }
}
