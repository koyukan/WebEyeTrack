import * as THREE from 'three';

export function solvePnP(imagePoints, objectPoints, intrinsicMatrix) {
  const numPoints = imagePoints.length;
  if (numPoints !== objectPoints.length) {
    throw new Error('Number of 2D and 3D points must be the same');
  }

  // Initial guess for rotation (identity) and translation (zero)
  let rotationMatrix = new THREE.Matrix3().identity();
  let translationVector = new THREE.Vector3(0, 0, 0);

  // Define the reprojection error function
  function reprojectionError(params) {
    const rvec = new THREE.Vector3(params[0], params[1], params[2]);
    const tvec = new THREE.Vector3(params[3], params[4], params[5]);

    const rotation = new THREE.Matrix3();
    rotation.set(
      Math.cos(rvec.z) * Math.cos(rvec.y), Math.cos(rvec.z) * Math.sin(rvec.y) * Math.sin(rvec.x) - Math.sin(rvec.z) * Math.cos(rvec.x), Math.cos(rvec.z) * Math.sin(rvec.y) * Math.cos(rvec.x) + Math.sin(rvec.z) * Math.sin(rvec.x),
      Math.sin(rvec.z) * Math.cos(rvec.y), Math.sin(rvec.z) * Math.sin(rvec.y) * Math.sin(rvec.x) + Math.cos(rvec.z) * Math.cos(rvec.x), Math.sin(rvec.z) * Math.sin(rvec.y) * Math.cos(rvec.x) - Math.cos(rvec.z) * Math.sin(rvec.x),
      -Math.sin(rvec.y), Math.cos(rvec.y) * Math.sin(rvec.x), Math.cos(rvec.y) * Math.cos(rvec.x)
    );

    let error = 0;

    for (let i = 0; i < numPoints; i++) {
      const objectPoint = objectPoints[i].clone();
      const projectedPoint = objectPoint.applyMatrix3(rotation).add(tvec);

      const u = (intrinsicMatrix.elements[0] * projectedPoint.x + intrinsicMatrix.elements[2] * projectedPoint.z) / projectedPoint.z;
      const v = (intrinsicMatrix.elements[4] * projectedPoint.y + intrinsicMatrix.elements[5] * projectedPoint.z) / projectedPoint.z;

      const imagePoint = imagePoints[i];
      error += (imagePoint.x - u) ** 2 + (imagePoint.y - v) ** 2;
    }

    return error;
  }

  // Compute numerical gradient
  function computeGradient(params, epsilon = 1e-6) {
    const grad = new Array(params.length).fill(0);
    const baseError = reprojectionError(params);

    for (let i = 0; i < params.length; i++) {
      params[i] += epsilon;
      const newError = reprojectionError(params);
      grad[i] = (newError - baseError) / epsilon;
      params[i] -= epsilon;
    }

    return grad;
  }

  // Gradient descent optimization
  function gradientDescent(params, learningRate, numIterations) {
    for (let i = 0; i < numIterations; i++) {
      const gradients = computeGradient(params);
      for (let j = 0; j < params.length; j++) {
        params[j] -= learningRate * gradients[j];
      }
      if (i % 100 === 0) {
        const error = reprojectionError(params);
        // console.log(`Iteration ${i}, Error: ${error}`);
      }
    }
    return params;
  }

  // Initial parameters: [rvec.x, rvec.y, rvec.z, tvec.x, tvec.y, tvec.z]
  const initialParams = [0, 0, 0, 0, 0, 0];

  // Perform the optimization
  const optimizedParams = gradientDescent(initialParams, 0.01, 1000);

  // Extract the optimized rotation and translation
  const optimizedRvec = new THREE.Vector3(optimizedParams[0], optimizedParams[1], optimizedParams[2]);
  const optimizedTvec = new THREE.Vector3(optimizedParams[3], optimizedParams[4], optimizedParams[5]);

  const optimizedRotation = new THREE.Matrix3();
  optimizedRotation.set(
    Math.cos(optimizedRvec.z) * Math.cos(optimizedRvec.y), Math.cos(optimizedRvec.z) * Math.sin(optimizedRvec.y) * Math.sin(optimizedRvec.x) - Math.sin(optimizedRvec.z) * Math.cos(optimizedRvec.x), Math.cos(optimizedRvec.z) * Math.sin(optimizedRvec.y) * Math.cos(optimizedRvec.x) + Math.sin(optimizedRvec.z) * Math.sin(optimizedRvec.x),
    Math.sin(optimizedRvec.z) * Math.cos(optimizedRvec.y), Math.sin(optimizedRvec.z) * Math.sin(optimizedRvec.y) * Math.sin(optimizedRvec.x) + Math.cos(optimizedRvec.z) * Math.cos(optimizedRvec.x), Math.sin(optimizedRvec.z) * Math.sin(optimizedRvec.y) * Math.cos(optimizedRvec.x) - Math.cos(optimizedRvec.z) * Math.sin(optimizedRvec.x),
    -Math.sin(optimizedRvec.y), Math.cos(optimizedRvec.y) * Math.sin(optimizedRvec.x), Math.cos(optimizedRvec.y) * Math.cos(optimizedRvec.x)
  );

  return {
    rotation: optimizedRotation,
    translation: optimizedTvec
  };
}