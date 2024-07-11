import * as THREE from 'three'
import { useCallback, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Grid, Environment, CameraControls, PerspectiveCamera, useHelper, Stats, useGLTF } from '@react-three/drei'
import { useControls, buttonGroup, folder } from 'leva'
import { easing } from 'maath'
import { suspend } from 'suspend-react'

import { FaceLandmarker } from './components/FaceLandmarker'
import { FaceControls } from './components/FaceControls'

import { Laptop } from './Laptop'

const city = import('@pmndrs/assets/hdri/city.exr')

export default function App() {
  return (
    <>
      <Canvas shadows camera={{ position: [-0.6, 0.1, 0.6], near: 0.01 }}>
        <FaceLandmarker>
          <Scene />
        </FaceLandmarker>
      </Canvas>
      <Stats />
    </>
  )
}

function Scene() {
  const vids = ['https://storage.googleapis.com/abernier-portfolio/metahumans.mp4', 'https://storage.googleapis.com/abernier-portfolio/metahumans2.mp4']

  const gui = useControls({
    camera: { value: 'user', options: ['user', 'cc'] },
    webcam: folder({
      webcam: true,
      autostart: true,
      webcamVideoTextureSrc: {
        value: vids[0],
        options: vids,
        optional: true,
        disabled: true
      },
      video: buttonGroup({
        opts: {
          pause: () => faceControlsApiRef.current?.pause(),
          play: () => faceControlsApiRef.current?.play()
        }
      })
    }),
    smoothTime: { value: 0.45, min: 0.000001, max: 1 },
    offset: true,
    offsetScalar: { value: 60, min: 0, max: 500 },
    eyes: false,
    eyesAsOrigin: true,
    origin: { value: 0, optional: true, disabled: true, min: 0, max: 477, step: 1 },
    depth: { value: 0.15, min: 0, max: 1, optional: true, disabled: true },
    player: folder({
      rotation: [0, 0, 0],
      position: [-0, 0.2, 0]
    })
  })

  const userCamRef = useRef()
  useHelper(gui.camera !== 'user' && userCamRef, THREE.CameraHelper)

  const [userCam, setUserCam] = useState()

  const controls = useThree((state) => state.controls)
  const faceControlsApiRef = useRef()

  const screenMatRef = useRef(null)
  const onVideoFrame = useCallback(
    (e) => {
      controls.detect(e.texture.source.data, e.time)

      screenMatRef.current.map = e.texture
    },
    [controls]
  )

  const [current] = useState(() => new THREE.Object3D())
  useFrame((_, delta) => {
    if (faceControlsApiRef.current) {
      const target = faceControlsApiRef.current.computeTarget()

      // faceControlsApiRef.current.update(delta, target);
      // userCam.position.copy(target.position);
      // userCam.rotation.copy(target.rotation);
      const eps = 1e-9
      easing.damp3(current.position, target.position, gui.smoothTime, delta, undefined, undefined, eps)
      easing.dampE(current.rotation, target.rotation, gui.smoothTime, delta, undefined, undefined, eps)

      userCam.position.copy(current.position)
      userCam.rotation.copy(current.rotation)
    }
  })

  return (
    <>
      <group rotation={gui.rotation} position={gui.position}>
        <FaceControls
          camera={userCam}
          ref={faceControlsApiRef}
          autostart={gui.autostart}
          makeDefault
          webcam={gui.webcam}
          webcamVideoTextureSrc={gui.webcamVideoTextureSrc}
          manualUpdate
          manualDetect
          onVideoFrame={onVideoFrame}
          smoothTime={gui.smoothTime}
          offset={gui.offset}
          offsetScalar={gui.offsetScalar}
          eyes={gui.eyes}
          eyesAsOrigin={gui.eyesAsOrigin}
          depth={gui.depth}
          facemesh={{ origin: gui.origin, position: [0, 0, 0] }}
          debug={gui.camera !== 'user'}
        />
        <PerspectiveCamera
          ref={(cam) => {
            userCamRef.current = cam
            setUserCam(cam)
          }}
          makeDefault={gui.camera === 'user'}
          fov={70}
          near={0.1}
          far={2}
        />
      </group>

      <Laptop flipHorizontal>
        <meshStandardMaterial ref={screenMatRef} side={THREE.DoubleSide} transparent opacity={0.9} />
      </Laptop>
      <Plant position={[-0.25, 0, -0.2]} scale={0.5} />

      {/* <axesHelper /> */}
      <Ground />

      <CameraControls />

      <Environment files={suspend(city).default} />
    </>
  )
}

function Ground() {
  const gridConfig = {
    cellSize: 0.1,
    cellThickness: 0.5,
    cellColor: '#6f6f6f',
    sectionSize: 1,
    sectionThickness: 1,
    // sectionColor: "#f7d76d",
    fadeDistance: 10,
    fadeStrength: 2,
    followCamera: false,
    infiniteGrid: true
  }
  return <Grid args={[10, 10]} {...gridConfig} />
}

const Plant = (props) => {
  const { nodes, materials } = useGLTF('https://storage.googleapis.com/abernier-portfolio/potted_plant.glb')
  return (
    <group {...props} dispose={null}>
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <mesh castShadow receiveShadow geometry={nodes.Object_2.geometry} material={materials.model_u1_v1} />
        <mesh castShadow receiveShadow geometry={nodes.Object_3.geometry} material={materials.model_u1_v1} />
      </group>
    </group>
  )
}
