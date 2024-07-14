import * as THREE from 'three'
import { useCallback, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Grid, Environment, CameraControls, PerspectiveCamera, useHelper, Stats, useGLTF } from '@react-three/drei'
import { useControls, buttonGroup, folder } from 'leva'
import { easing } from 'maath'
import { suspend } from 'suspend-react'

import { FaceLandmarker } from './components/FaceLandmarker'
import { FaceControls } from './components/FaceControls'

import { Screen } from './Screen.jsx'
import { Camera } from './Camera.jsx'

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
  
// Screen dimensions
const screenWidth = screen.width;
const screenHeight = screen.height;
// console.log(`Screen Width: ${screenWidth}px, Screen Height: ${screenHeight}px`);

// px to cm
// const pxToCm = 0.000264583333;
// const pxToM = 0.0001536458
const pxToCm = 0.01536458;
// const w = screenWidth * pxToCm
// const h = screenHeight * pxToCm
const w = 36; // cm
const h = 81; // cm

function Scene() {
  const vids = ['https://storage.googleapis.com/abernier-portfolio/metahumans.mp4', 'https://storage.googleapis.com/abernier-portfolio/metahumans2.mp4']

  const gui = useControls({
    camera: { value: 'cc', options: ['user', 'cc'] },
    screen: false,
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
    offsetScalar: { value: 1, min: 1, max: 500 },
    eyes: true,
    eyesAsOrigin: false,
    origin: { value: 0, optional: true, disabled: false, min: 0, max: 477, step: 1 },
    depth: { value: 0.15, min: 0, max: 1, optional: true, disabled: false},
    player: folder({
      rotation: [0, 0, 0],
      position: [-0, 0, 0]
    })
  })

  const userCamRef = useRef()
  useHelper(gui.camera !== 'user' && userCamRef, THREE.CameraHelper)

  const [userCam, setUserCam] = useState()

  const controls = useThree((state) => state.controls)
  const faceControlsApiRef = useRef()

  const screenMatRef = useRef(null)
  const webcamMatRef = useRef(null)
  const onWebcamVideoFrame = useCallback(
    (e) => {
      controls.detect(e.texture.source.data, e.time)
      webcamMatRef.current.map = e.texture
      // console.log(e.texture.source.data.videoWidth, e.texture.source.data.videoHeight)
    },
    [controls]
  )
  const onScreenVideoFrame = useCallback(
    (e) => {
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
          screen={gui.screen}
          webcam={gui.webcam}
          webcamVideoTextureSrc={gui.webcamVideoTextureSrc}
          manualUpdate
          manualDetect
          onWebcamVideoFrame={onWebcamVideoFrame}
          onScreenVideoFrame={onScreenVideoFrame}
          smoothTime={gui.smoothTime}
          offset={gui.offset}
          offsetScalar={gui.offsetScalar}
          eyes={gui.eyes}
          eyesAsOrigin={gui.eyesAsOrigin}
          depth={gui.depth}
          facemesh={{ origin: gui.origin, position: [0, 0, 0] }}
          // debug={gui.camera !== 'user'}
          debug={true}
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

      <Screen>
        <meshStandardMaterial ref={screenMatRef} side={THREE.DoubleSide} transparent opacity={1}/>
      </Screen>

      <Camera 
        scale={10}
        aspect={1}
        // transform={{ position: [0, h, -h/2], rotation: [180, 0, 0] }}
        // transform={{ position: [0, h, 0], rotation: [180, 0, 0] }}
        transform={{ position: [0, 0, 0], rotation: [180, 0, 0] }}
      >
        {webcamMatRef &&
          <meshStandardMaterial ref={webcamMatRef} side={THREE.DoubleSide} transparent opacity={0.9}/>
        }
        {/* <meshStandardMaterial ref={webcamMatRef} transparent opacity={0.9}/> */}
      </Camera>

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
    infiniteGrid: true,
    position: [0, -h, 0],
  }
  return <Grid args={[10, 10]} {...gridConfig} />
}