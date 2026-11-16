import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { ThreeEvent, useThree } from '@react-three/fiber'
import { type RapierRigidBody, useBeforePhysicsStep, useRapier } from '@react-three/rapier'
import * as THREE from 'three'
import { markObjectInteraction } from '../interaction/pointerGuards'
import { SCENE_OBJECT_INSTANCE_ID_KEY, type SceneObjectHandle } from './SceneObject'

/** Throw speed cap on release. */
const GRAB_LINEAR_SPEED_LIMIT = 5
const GRAB_ANGULAR_SPEED_LIMIT = 10
/** While held, the object is steered by velocity toward the cursor. These cap how
 * fast it may chase it, and what fraction of the gap it closes each physics step
 * (1 would be exact but jittery against contacts). */
const HOLD_LINEAR_SPEED_LIMIT = 10
const HOLD_LINEAR_GAIN = 0.6
/** Fraction of the remaining turn applied per step when R turns a held object. */
const HOLD_ROTATION_EASE = 0.3
/** Released slower than this, an object is being set down, not thrown: it lets go
 * with no velocity, so a vase lowered onto a table stays where it was put. */
const PLACE_SPEED_LIMIT = 0.8
/** R / Shift+R turns a held object about the vertical by this much. */
const ROTATE_STEP = Math.PI / 12
/** Mouse wheel while holding pushes the object away or pulls it in, within these. */
const GRAB_DEPTH_MIN = 0.3
const GRAB_DEPTH_MAX = 10
const GRAB_DEPTH_PER_WHEEL_PIXEL = 0.0008  // ~10% per 120-pixel wheel notch
const UP = new THREE.Vector3(0, 1, 0)
/** Heavier than this (Rapier mass: hull volume in m^3 at density 1, so ~60
 * litres), an object is furniture: grabbing it slides it along the floor with
 * gravity on, instead of lifting it into the air. */
const HEAVY_MASS = 0.06
/** Furniture (kinematic) closes this fraction of the gap to the cursor per step,
 * no faster than KINEMATIC_SPEED. */
const KINEMATIC_GAIN = 0.5
const KINEMATIC_SPEED = 3
/** Furniture stops this far short of what it would hit. */
const SLIDE_SKIN_M = 0.01
/** A turn is refused if a probe this much smaller would hit something. */
const TURN_PROBE_INSET = 0.85

type ObjectRefMap = Map<string, RefObject<SceneObjectHandle | null>>

interface UseObjectGrabArgs {
  objectRefs: RefObject<ObjectRefMap>
  isObjectEligible?: (objectId: string) => boolean
}

interface ActiveGrab {
  objectId: string
  body: RapierRigidBody
  pointerId: number
  depth: number
  pointerNdc: THREE.Vector2
  target: THREE.Vector3
  previousTarget: THREE.Vector3
  releaseVelocity: THREE.Vector3
  /** The object's placed orientation from scene.json -- what "upright" means for it. */
  restRotation: THREE.Quaternion
  /** Turn about the vertical, relative to restRotation. */
  yaw: number
  /** restRotation turned by yaw: the held object is steered to this. */
  targetRotation: THREE.Quaternion
  /** The grabbed point, in the body's own frame. */
  localAnchor: THREE.Vector3
  /** Restored on release; gravity is off while a light object is held. */
  gravityScale: number
  /** Furniture: dragged across the floor at the grabbed point's height. */
  slide: boolean
  slideHeight: number
  /** Kinematic furniture is moved directly, checked against walls and furniture. */
  kinematic: boolean
  probe: SceneObjectHandle['probe']
}

/** The part of `relative` that is a turn about the world vertical (swing-twist). */
function twistAngleAboutUp(relative: THREE.Quaternion) {
  if (Math.abs(relative.y) < 1e-6 && Math.abs(relative.w) < 1e-6) return 0
  return 2 * Math.atan2(relative.y, relative.w)
}

function uprightRotation(rest: THREE.Quaternion, yaw: number, target: THREE.Quaternion) {
  return target.setFromAxisAngle(UP, yaw).multiply(rest)
}

const _raycaster = new THREE.Raycaster()
const _grabDepthVector = new THREE.Vector3()
const _bodyPosition = new THREE.Vector3()
const _bodyRotation = new THREE.Quaternion()
const _inverseBodyRotation = new THREE.Quaternion()
const _localAnchor = new THREE.Vector3()
const _bodyLinearVelocity = new THREE.Vector3()
const _bodyAngularVelocity = new THREE.Vector3()
const _objectCenter = new THREE.Vector3()
const _grabPoint = new THREE.Vector3()
const _leverArm = new THREE.Vector3()
const _pointVelocity = new THREE.Vector3()
const _currentRotation = new THREE.Quaternion()
const _projectedObjectCenter = new THREE.Vector3()
const _zeroVector = { x: 0, y: 0, z: 0 }

interface PointerTarget {
  objectId: string
  handle: SceneObjectHandle
  point: THREE.Vector3
  centerDistanceSq: number
  hitDistance: number
}

function isBodyUsable(body: RapierRigidBody | null | undefined): body is RapierRigidBody {
  if (!body) return false
  try {
    body.translation()
    return true
  } catch {
    return false
  }
}

function vectorLike(vector: THREE.Vector3) {
  return { x: vector.x, y: vector.y, z: vector.z }
}

function quaternionLike(quaternion: THREE.Quaternion) {
  return { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w }
}

function objectIdFromIntersectionObject(object: THREE.Object3D) {
  let current: THREE.Object3D | null = object
  while (current) {
    const objectId = current.userData[SCENE_OBJECT_INSTANCE_ID_KEY]
    if (typeof objectId === 'string') return objectId
    current = current.parent
  }
  return null
}

function centerDistanceSqToPointer(event: ThreeEvent<PointerEvent>, center: THREE.Vector3, camera: THREE.Camera) {
  _projectedObjectCenter.copy(center).project(camera)
  return (
    (_projectedObjectCenter.x - event.pointer.x) ** 2 +
    (_projectedObjectCenter.y - event.pointer.y) ** 2
  )
}

function computePointerNdc(element: HTMLElement, clientX: number, clientY: number) {
  const rect = element.getBoundingClientRect()
  return new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1,
  )
}

function computeLocalAnchor(body: RapierRigidBody, worldPoint: THREE.Vector3) {
  const translation = body.translation()
  const rotation = body.rotation()

  _bodyPosition.set(translation.x, translation.y, translation.z)
  _bodyRotation.set(rotation.x, rotation.y, rotation.z, rotation.w)
  _inverseBodyRotation.copy(_bodyRotation).invert()

  return _localAnchor.copy(worldPoint).sub(_bodyPosition).applyQuaternion(_inverseBodyRotation).clone()
}

/**
 * Steer a held body so its grabbed point chases `target`, by setting velocity and
 * letting the solver resolve contacts.
 *
 * This replaces a fixed joint to a kinematic anchor. A kinematic anchor has
 * infinite mass, so when the cursor went into a table the joint and the contact
 * fought and the joint won: the held object sank into the table, and anything in
 * its path was shoved with unbounded force. Velocity is something contacts can
 * stop, so a held object now meets surfaces and slides along them.
 *
 * Rotation is locked while held (contacts cannot tip it over) and eased toward
 * `targetRotation` directly, which is only ever a turn about the vertical.
 */
function steerHeldBody(grab: ActiveGrab, dt: number) {
  const body = grab.body
  const translation = body.translation()
  const rotation = body.rotation()
  _currentRotation.set(rotation.x, rotation.y, rotation.z, rotation.w)
  if (_currentRotation.angleTo(grab.targetRotation) > 1e-4) {
    _currentRotation.slerp(grab.targetRotation, HOLD_ROTATION_EASE)
    body.setRotation(quaternionLike(_currentRotation), true)
  }

  _leverArm.copy(grab.localAnchor).applyQuaternion(_currentRotation)
  _grabPoint.set(translation.x, translation.y, translation.z).add(_leverArm)
  _pointVelocity.copy(grab.target).sub(_grabPoint).multiplyScalar(HOLD_LINEAR_GAIN / dt)
  if (_pointVelocity.length() > HOLD_LINEAR_SPEED_LIMIT) _pointVelocity.setLength(HOLD_LINEAR_SPEED_LIMIT)
  // Sliding furniture leaves vertical motion to gravity and the floor.
  if (grab.slide) _pointVelocity.y = body.linvel().y
  body.setLinvel(vectorLike(_pointVelocity), true)
  body.setAngvel(_zeroVector, true)
}

function clampBodyVelocity(body: RapierRigidBody) {
  let linear
  let angular
  try {
    linear = body.linvel()
    angular = body.angvel()
  } catch {
    return false
  }
  _bodyLinearVelocity.set(linear.x, linear.y, linear.z)
  if (_bodyLinearVelocity.length() > GRAB_LINEAR_SPEED_LIMIT) {
    _bodyLinearVelocity.setLength(GRAB_LINEAR_SPEED_LIMIT)
    body.setLinvel(vectorLike(_bodyLinearVelocity), true)
  }

  _bodyAngularVelocity.set(angular.x, angular.y, angular.z)
  if (_bodyAngularVelocity.length() > GRAB_ANGULAR_SPEED_LIMIT) {
    _bodyAngularVelocity.setLength(GRAB_ANGULAR_SPEED_LIMIT)
    body.setAngvel(vectorLike(_bodyAngularVelocity), true)
  }
  return true
}

export function useObjectGrab({ objectRefs, isObjectEligible }: UseObjectGrabArgs) {
  const { camera, gl } = useThree()
  const { world, rapier } = useRapier()
  const [activeObjectId, setActiveObjectId] = useState<string | null>(null)
  const activeGrabRef = useRef<ActiveGrab | null>(null)
  const mountedRef = useRef(true)

  const setActiveObjectIdSafe = useCallback((objectId: string | null) => {
    if (mountedRef.current) setActiveObjectId(objectId)
  }, [])

  const updatePointerTarget = useCallback(
    (grab: ActiveGrab) => {
      _raycaster.setFromCamera(grab.pointerNdc, camera)
      const { origin, direction } = _raycaster.ray
      if (grab.slide) {
        // Where the pointer ray meets the plane the grabbed point slides in. A ray
        // at or above the horizon never meets it; keep the last target.
        if (direction.y >= -1e-3) return
        const t = Math.min((grab.slideHeight - origin.y) / direction.y, GRAB_DEPTH_MAX)
        if (t > 0) grab.target.copy(origin).addScaledVector(direction, t)
        return
      }
      grab.target.copy(origin).addScaledVector(direction, grab.depth)
    },
    [camera],
  )

  const endGrab = useCallback(() => {
    const activeGrab = activeGrabRef.current
    if (!activeGrab) return

    if (isBodyUsable(activeGrab.body) && !activeGrab.kinematic) {
      activeGrab.body.setGravityScale(activeGrab.gravityScale, true)
      activeGrab.body.lockRotations(false, true)
      const releaseVelocity = activeGrab.releaseVelocity.clone()
      if (releaseVelocity.length() < PLACE_SPEED_LIMIT) {
        // Set down: upright, at rest, and it simply settles onto what is below.
        activeGrab.body.setLinvel(_zeroVector, true)
        activeGrab.body.setAngvel(_zeroVector, true)
      } else {
        if (releaseVelocity.length() > GRAB_LINEAR_SPEED_LIMIT) {
          releaseVelocity.setLength(GRAB_LINEAR_SPEED_LIMIT)
        }
        activeGrab.body.setLinvel(vectorLike(releaseVelocity), true)
        clampBodyVelocity(activeGrab.body)
      }
      activeGrab.body.wakeUp()
    }

    try {
      if (gl.domElement.hasPointerCapture(activeGrab.pointerId)) {
        gl.domElement.releasePointerCapture(activeGrab.pointerId)
      }
    } catch {
      // Pointer capture may already be gone during canvas or route teardown.
    }

    activeGrabRef.current = null
    setActiveObjectIdSafe(null)
    markObjectInteraction()
  }, [gl.domElement, setActiveObjectIdSafe])

  const beginGrab = useCallback(
    (objectId: string, handle: SceneObjectHandle, pointerId: number, clientX: number, clientY: number, worldPoint: THREE.Vector3) => {
      const body = handle.rigidBody
      if (!isBodyUsable(body)) return

      endGrab()

      const pointerNdc = computePointerNdc(gl.domElement, clientX, clientY)
      _raycaster.setFromCamera(pointerNdc, camera)
      const depth = Math.max(_grabDepthVector.copy(worldPoint).sub(_raycaster.ray.origin).dot(_raycaster.ray.direction), 0.1)

      // Held upright: the object's own placed orientation, keeping only its current
      // heading. A vase knocked on its side comes back upright when picked up, and a
      // palette placed lying flat stays flat. R / Shift+R turn it from there.
      const restRotation = handle.initialRotation.clone()
      const current = body.rotation()
      const relative = new THREE.Quaternion(current.x, current.y, current.z, current.w)
        .multiply(restRotation.clone().invert())
      const yaw = twistAngleAboutUp(relative)
      const targetRotation = uprightRotation(restRotation, yaw, new THREE.Quaternion())

      const kinematic = body.isKinematic()
      if (!kinematic && !clampBodyVelocity(body)) return
      const localAnchor = computeLocalAnchor(body, worldPoint)
      const gravityScale = body.gravityScale()
      const slide = kinematic || body.mass() >= HEAVY_MASS
      if (!kinematic) {
        if (!slide) body.setGravityScale(0, true)
        body.setAngvel(_zeroVector, true)
        body.lockRotations(true, true)
        body.wakeUp()
      }

      try {
        if (!gl.domElement.hasPointerCapture(pointerId)) {
          gl.domElement.setPointerCapture(pointerId)
        }
      } catch {
        if (!kinematic) {
          body.setGravityScale(gravityScale, true)
          body.lockRotations(false, true)
        }
        return
      }

      activeGrabRef.current = {
        objectId,
        body,
        pointerId,
        depth,
        pointerNdc,
        target: worldPoint.clone(),
        previousTarget: worldPoint.clone(),
        releaseVelocity: new THREE.Vector3(),
        restRotation,
        yaw,
        targetRotation,
        localAnchor,
        gravityScale,
        slide,
        slideHeight: worldPoint.y,
        kinematic,
        probe: handle.probe,
      }
      setActiveObjectIdSafe(objectId)
    },
    [camera, endGrab, gl.domElement, setActiveObjectIdSafe],
  )

  const getPointerTarget = useCallback(
    (fallbackObjectId: string, event: ThreeEvent<PointerEvent>): PointerTarget | null => {
      const seenObjectIds = new Set<string>()
      let best: PointerTarget | null = null

      for (const intersection of event.intersections) {
        const objectId = objectIdFromIntersectionObject(intersection.object)
        if (!objectId || seenObjectIds.has(objectId)) continue
        seenObjectIds.add(objectId)
        if (isObjectEligible && !isObjectEligible(objectId)) continue

        const handle = objectRefs.current.get(objectId)?.current
        if (!handle || !isBodyUsable(handle.rigidBody)) continue

        const centerDistanceSq = centerDistanceSqToPointer(event, handle.getFocusPoint(_objectCenter), camera)
        const hitDistance = Number.isFinite(intersection.distance) ? intersection.distance : Number.POSITIVE_INFINITY
        // The nearest grabbable object the ray hits -- what is under the cursor. It
        // used to be whichever object's *centre* projected closest to the cursor,
        // which could pick something behind: a click on the vase took the mug.
        if (
          !best ||
          hitDistance < best.hitDistance ||
          (hitDistance === best.hitDistance && centerDistanceSq < best.centerDistanceSq)
        ) {
          best = {
            objectId,
            handle,
            point: intersection.point.clone(),
            centerDistanceSq,
            hitDistance,
          }
        }
      }

      // Boxes overlap -- the monstera's encloses the vase beside it -- so among the
      // objects whose box the ray entered, take the one whose model it hits first.
      if (best) {
        _raycaster.setFromCamera(event.pointer, camera)
        let modelBest: { target: PointerTarget; distance: number } | null = null
        for (const intersection of event.intersections) {
          const objectId = objectIdFromIntersectionObject(intersection.object)
          if (!objectId || (isObjectEligible && !isObjectEligible(objectId))) continue
          const handle = objectRefs.current.get(objectId)?.current
          if (!handle || !isBodyUsable(handle.rigidBody)) continue
          const distance = handle.raycastModel(_raycaster)
          if (distance === null || (modelBest && distance >= modelBest.distance)) continue
          const point = _raycaster.ray.at(distance, new THREE.Vector3())
          modelBest = {
            distance,
            target: { objectId, handle, point, centerDistanceSq: 0, hitDistance: distance },
          }
        }
        return modelBest?.target ?? best
      }
      if (isObjectEligible && !isObjectEligible(fallbackObjectId)) return null

      const fallbackHandle = objectRefs.current.get(fallbackObjectId)?.current
      if (!fallbackHandle || !isBodyUsable(fallbackHandle.rigidBody)) return null
      return {
        objectId: fallbackObjectId,
        handle: fallbackHandle,
        point: event.point.clone(),
        centerDistanceSq: centerDistanceSqToPointer(event, fallbackHandle.getFocusPoint(_objectCenter), camera),
        hitDistance: event.distance,
      }
    },
    [camera, isObjectEligible, objectRefs],
  )

  const onPointerDown = useCallback(
    (objectId: string, event: ThreeEvent<PointerEvent>) => {
      if (event.button !== 0) return false
      const target = getPointerTarget(objectId, event)
      if (!target) return false

      event.stopPropagation()
      event.nativeEvent.preventDefault()
      markObjectInteraction()
      target.handle.playInteractionSfx()
      beginGrab(target.objectId, target.handle, event.pointerId, event.clientX, event.clientY, target.point)
      return true
    },
    [beginGrab, getPointerTarget],
  )

  const resetObjects = useCallback(() => {
    endGrab()
    for (const objectRef of objectRefs.current.values()) {
      const handle = objectRef.current
      const body = handle?.rigidBody
      if (!handle || !isBodyUsable(body)) continue

      body.setTranslation(vectorLike(handle.initialPosition), true)
      body.setRotation(quaternionLike(handle.initialRotation), true)
      body.setLinvel(_zeroVector, true)
      body.setAngvel(_zeroVector, true)
      body.wakeUp()
    }
  }, [endGrab, objectRefs])

  const probeShape = useCallback((grab: ActiveGrab, inset = 1) => {
    const { halfExtents } = grab.probe
    return new rapier.Cuboid(halfExtents.x * inset, halfExtents.y * inset, halfExtents.z * inset)
  }, [rapier])

  /**
   * How much of `delta` furniture can travel before its probe meets a wall or
   * other furniture, as a fraction 0..1. A shape cast that ignores contacts the
   * motion is leaving, so a desk standing against a wall can slide along it or
   * away from it, but not into it. (Asking "does it overlap anything new" fails:
   * the whole room is one collider, which the desk touches from the start.)
   */
  const travelFraction = useCallback((grab: ActiveGrab, position: THREE.Vector3, rotation: THREE.Quaternion, delta: THREE.Vector3) => {
    const at = grab.probe.center.clone().applyQuaternion(rotation).add(position)
    const hit = world.castShape(
      vectorLike(at), quaternionLike(rotation), vectorLike(delta), probeShape(grab), SLIDE_SKIN_M, 1, false,
      rapier.QueryFilterFlags.EXCLUDE_DYNAMIC | rapier.QueryFilterFlags.EXCLUDE_SENSORS,
      undefined, undefined, grab.body,
    )
    return hit ? Math.max(0, hit.time_of_impact) : 1
  }, [probeShape, rapier, world])

  /** True when a turned pose would put a (slightly smaller) probe into a wall or furniture. */
  const turnBlocked = useCallback((grab: ActiveGrab, position: THREE.Vector3, rotation: THREE.Quaternion) => {
    const at = grab.probe.center.clone().applyQuaternion(rotation).add(position)
    return world.intersectionWithShape(
      vectorLike(at), quaternionLike(rotation), probeShape(grab, TURN_PROBE_INSET),
      rapier.QueryFilterFlags.EXCLUDE_DYNAMIC | rapier.QueryFilterFlags.EXCLUDE_SENSORS,
      undefined, undefined, grab.body,
    ) !== null
  }, [probeShape, rapier, world])

  /** Height of whatever is under (x, z), below `fromY`, ignoring props and the body itself. */
  const supportHeight = useCallback((grab: ActiveGrab, x: number, z: number, fromY: number) => {
    const ray = new rapier.Ray({ x, y: fromY, z }, { x: 0, y: -1, z: 0 })
    const hit = world.castRay(ray, 4, true, rapier.QueryFilterFlags.EXCLUDE_DYNAMIC, undefined, undefined, grab.body)
    if (!hit) return null
    const toi = (hit as unknown as { toi?: number; timeOfImpact?: number })
    return fromY - (toi.timeOfImpact ?? toi.toi ?? 0)
  }, [rapier, world])

  /**
   * Furniture is kinematic: it keeps its exact collision mesh and pushes props, but
   * nothing stops it. So it is moved directly, a step at a time, and a step that
   * would make it overlap a wall or another piece of furniture it was not already
   * touching is refused -- tried whole, then along x alone, then z alone, so it
   * slides along a wall instead of sticking to it. It follows the floor's slope.
   */
  const slideFurniture = useCallback((grab: ActiveGrab, dt: number) => {
    const body = grab.body
    const t = body.translation()
    const r = body.rotation()
    const position = new THREE.Vector3(t.x, t.y, t.z)
    const rotation = new THREE.Quaternion(r.x, r.y, r.z, r.w)

    let nextRotation = rotation
    if (rotation.angleTo(grab.targetRotation) > 1e-4) {
      const turned = rotation.clone().slerp(grab.targetRotation, HOLD_ROTATION_EASE)
      if (!turnBlocked(grab, position, turned)) nextRotation = turned
    }

    _grabPoint.copy(grab.localAnchor).applyQuaternion(nextRotation).add(position)
    const step = grab.target.clone().sub(_grabPoint).setY(0).multiplyScalar(KINEMATIC_GAIN)
    const maxStep = KINEMATIC_SPEED * dt
    if (step.length() > maxStep) step.setLength(maxStep)

    let next = position
    if (step.lengthSq() > 1e-10) {
      // Whole step, else along x alone, else z alone: slides along a wall.
      for (const candidate of [step, new THREE.Vector3(step.x, 0, 0), new THREE.Vector3(0, 0, step.z)]) {
        if (candidate.lengthSq() < 1e-10) continue
        const fraction = travelFraction(grab, position, nextRotation, candidate)
        if (fraction > 0.05) {
          next = position.clone().addScaledVector(candidate, fraction)
          break
        }
      }
    }
    if (next !== position) {
      const probeTop = position.y + grab.probe.center.y
      const from = supportHeight(grab, position.x, position.z, probeTop)
      const to = supportHeight(grab, next.x, next.z, probeTop)
      if (from !== null && to !== null) next.y += to - from
    }
    // A sleeping kinematic body ignores its next pose; furniture sleeps once settled.
    body.wakeUp()
    body.setNextKinematicTranslation(vectorLike(next))
    body.setNextKinematicRotation(quaternionLike(nextRotation))
  }, [supportHeight, travelFraction, turnBlocked])

  useBeforePhysicsStep((physicsWorld) => {
    const activeGrab = activeGrabRef.current
    if (!activeGrab) return
    if (!isBodyUsable(activeGrab.body)) {
      activeGrabRef.current = null
      setActiveObjectIdSafe(null)
      return
    }

    updatePointerTarget(activeGrab)
    const dt = physicsWorld.timestep || 1 / 60
    activeGrab.releaseVelocity.copy(activeGrab.target).sub(activeGrab.previousTarget).divideScalar(dt)
    if (activeGrab.releaseVelocity.length() > GRAB_LINEAR_SPEED_LIMIT) {
      activeGrab.releaseVelocity.setLength(GRAB_LINEAR_SPEED_LIMIT)
    }
    activeGrab.previousTarget.copy(activeGrab.target)
    if (activeGrab.kinematic) slideFurniture(activeGrab, dt)
    else steerHeldBody(activeGrab, dt)
  })

  useEffect(() => {
    mountedRef.current = true

    const onPointerMove = (event: PointerEvent) => {
      const activeGrab = activeGrabRef.current
      if (activeGrab && event.pointerId === activeGrab.pointerId) {
        activeGrab.pointerNdc.copy(computePointerNdc(gl.domElement, event.clientX, event.clientY))
        event.preventDefault()
        return
      }

    }

    const onPointerEnd = (event: PointerEvent) => {
      if (activeGrabRef.current?.pointerId === event.pointerId) {
        event.preventDefault()
        endGrab()
      }
    }

    // While holding: the wheel moves the object along the view ray instead of
    // zooming the camera. Captured on window, so it runs before the camera's own
    // wheel handler on the canvas and can stop it.
    const onWheel = (event: WheelEvent) => {
      const activeGrab = activeGrabRef.current
      if (!activeGrab) return
      event.preventDefault()
      event.stopImmediatePropagation()
      if (activeGrab.slide) return  // furniture stays on the floor
      activeGrab.depth = THREE.MathUtils.clamp(
        activeGrab.depth * Math.exp(-event.deltaY * GRAB_DEPTH_PER_WHEEL_PIXEL),
        GRAB_DEPTH_MIN,
        GRAB_DEPTH_MAX,
      )
    }

    // While holding: R turns the object one step about the vertical, Shift+R back.
    const onKeyDown = (event: KeyboardEvent) => {
      const activeGrab = activeGrabRef.current
      if (!activeGrab || event.code !== 'KeyR') return
      event.preventDefault()
      activeGrab.yaw += event.shiftKey ? -ROTATE_STEP : ROTATE_STEP
      uprightRotation(activeGrab.restRotation, activeGrab.yaw, activeGrab.targetRotation)
    }

    window.addEventListener('pointermove', onPointerMove, { capture: true, passive: false })
    window.addEventListener('pointerup', onPointerEnd, { capture: true, passive: false })
    window.addEventListener('pointercancel', onPointerEnd, { capture: true, passive: false })
    window.addEventListener('wheel', onWheel, { capture: true, passive: false })
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('pointermove', onPointerMove, { capture: true })
      window.removeEventListener('pointerup', onPointerEnd, { capture: true })
      window.removeEventListener('pointercancel', onPointerEnd, { capture: true })
      window.removeEventListener('wheel', onWheel, { capture: true })
      window.removeEventListener('keydown', onKeyDown)
      endGrab()
      mountedRef.current = false
    }
  }, [endGrab, gl.domElement])

  return {
    activeObjectId,
    activeGrabRef,
    cancelGrab: endGrab,
    onPointerDown,
    resetObjects,
  }
}
