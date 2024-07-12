import * as THREE from 'three';
import React, { forwardRef, useRef, useEffect } from 'react';
import { Line } from '@react-three/drei';
import { useThree } from '@react-three/fiber';

const { DEG2RAD } = THREE.MathUtils;

export const Camera = forwardRef(({
  scale = 1,
  aspect = 1,
  transform = { position: [0, 0, 0], rotation: [0, 0, 0] },
  ...props
}, fref) => {
 
  const calculateFrustumPoints = () => {
    const near = 0.1 * scale;
    const far = 1 * scale;
    const fov = 75 * DEG2RAD;
    const heightNear = 2 * Math.tan(fov / 2) * near;
    const heightFar = 2 * Math.tan(fov / 2) * far;
    const widthNear = heightNear * aspect;
    const widthFar = heightFar * aspect;

    const points = [
      // Origin
      [0, 0, 0],

      // Near plane
      [widthNear / 2, heightNear / 2, -near],
      [-widthNear / 2, heightNear / 2, -near],
      [-widthNear / 2, -heightNear / 2, -near],
      [widthNear / 2, -heightNear / 2, -near],

      // Far plane
      [widthFar / 2, heightFar / 2, -far],
      [-widthFar / 2, heightFar / 2, -far],
      [-widthFar / 2, -heightFar / 2, -far],
      [widthFar / 2, -heightFar / 2, -far],
    ];

    return points.map(point => new THREE.Vector3(...point));
  };

  const applyTransformation = (points, position, rotation) => {
    const matrix = new THREE.Matrix4();
    matrix.compose(
      new THREE.Vector3(...position),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation.map(r => r * DEG2RAD))),
      new THREE.Vector3(1, 1, 1)
    );

    return points.map(point => point.applyMatrix4(matrix));
  };

  const frustumPoints = applyTransformation(calculateFrustumPoints(), transform.position, transform.rotation);

  return (
    <group {...props}>
      {/* Lines from the camera origin to the near plane */}
      <Line points={[frustumPoints[0], frustumPoints[1]]} color="red" />
      <Line points={[frustumPoints[0], frustumPoints[2]]} color="red" />
      <Line points={[frustumPoints[0], frustumPoints[3]]} color="red" />
      <Line points={[frustumPoints[0], frustumPoints[4]]} color="red" />

      {/* Lines connecting near and far planes */}
      <Line points={[frustumPoints[1], frustumPoints[5]]} color="red" />
      <Line points={[frustumPoints[2], frustumPoints[6]]} color="red" />
      <Line points={[frustumPoints[3], frustumPoints[7]]} color="red" />
      <Line points={[frustumPoints[4], frustumPoints[8]]} color="red" />

      {/* Near plane lines */}
      <Line points={[frustumPoints[1], frustumPoints[2]]} color="red" />
      <Line points={[frustumPoints[2], frustumPoints[3]]} color="red" />
      <Line points={[frustumPoints[3], frustumPoints[4]]} color="red" />
      <Line points={[frustumPoints[4], frustumPoints[1]]} color="red" />

      {/* Far plane lines */}
      <Line points={[frustumPoints[5], frustumPoints[6]]} color="red" />
      <Line points={[frustumPoints[6], frustumPoints[7]]} color="red" />
      <Line points={[frustumPoints[7], frustumPoints[8]]} color="red" />
      <Line points={[frustumPoints[8], frustumPoints[5]]} color="red" />
    </group>
  );
});