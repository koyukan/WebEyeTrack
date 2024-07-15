import * as THREE from 'three'
import { forwardRef } from 'react'
import { Plane, Box } from '@react-three/drei'
import { useControls, folder } from 'leva'

const { DEG2RAD } = THREE.MathUtils

function px2cm(px) {
  var n = 0;
  var cpi = 2.54; // centimeters per inch
  var dpi = 96; // dots per inch
  var ppd = window.devicePixelRatio; // pixels per dot
  return (px * cpi / (dpi * ppd)).toFixed(n);
}

export const Screen = forwardRef(({ children, flipHorizontal = false, ...props }, fref) => {
  
  // Screen dimensions
  const screenWidth = screen.width;
  const screenHeight = screen.height;
  const w = px2cm(screenWidth);
  const h = px2cm(screenHeight);
  const d = 0.1

  const openAngle = 90

  const d2 = d / 10

  return (
    <group {...props}>
      <group >
        <group rotation-x={(90 - openAngle) * DEG2RAD}>
          <Plane args={[w, h]} position-y={-h/2} position-z={-d2 / 2}>
            <meshStandardMaterial color="gray" side={THREE.DoubleSide} />
            {children}
          </Plane>
        </group>
      </group>
    </group>
  )
})
