import * as THREE from 'three'
import { forwardRef } from 'react'
import { Plane, Box } from '@react-three/drei'
import { useControls, folder } from 'leva'

const { DEG2RAD } = THREE.MathUtils

export const Screen = forwardRef(({ children, flipHorizontal = false, ...props }, fref) => {
  
  // Screen dimensions
  const screenWidth = screen.width;
  const screenHeight = screen.height;
  // console.log(`Screen Width: ${screenWidth}px, Screen Height: ${screenHeight}px`);

  // px to cm
  // const pxToCm = 0.000264583333;
  const pxToCm = 0.01536458;
  const pxToM = 0.0001536458;
  const w = screenWidth * pxToCm
  const h = screenHeight * pxToCm
  const d = 0.01

  const openAngle = 90

  const d2 = d / 10

  return (
    <group {...props}>
      {/* <group position-z={-h / 2}> */}
      <group >
        {/* Lid */}
        <group rotation-x={(90 - openAngle) * DEG2RAD}>
          {/* <Box args={[w, h, d2]} position-y={h / 2} position-z={-d2 / 2}> */}
          <Box args={[w, h, d2]} position-y={-h/2} position-z={-d2 / 2}>
            <meshStandardMaterial color="gray" side={THREE.DoubleSide} />
            <Plane args={[w, h]} position-z={d2 / 2 + 0.0001}>
              <meshStandardMaterial color="black" />
              <Plane args={[w - w / 100, h - h / 100]} position-z={0.0001} scale-x={flipHorizontal ? -1 : 1}>
                {children}
              </Plane>
            </Plane>
          </Box>
        </group>
      </group>
    </group>
  )
})
