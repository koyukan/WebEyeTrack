import { Matrix, inverse } from 'ml-matrix';
import { Matrix as MediaPipeMatrix, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { Point } from './types';
import { safeSVD } from './safeSVD';

// ============================================================================
// Eye Patch Extraction and Homography
// ============================================================================

/**
 * Estimates a 3x3 homography matrix from 4 point correspondences.
 */
export function computeHomography(src: Point[], dst: Point[]): number[][] {
  if (src.length !== 4 || dst.length !== 4) {
    throw new Error('Need exactly 4 source and 4 destination points');
  }

  const A: number[][] = [];

  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];

    A.push([-x, -y, -1, 0, 0, 0, x * u, y * u, u]);
    A.push([0, 0, 0, -x, -y, -1, x * v, y * v, v]);
  }

  const A_mat = new Matrix(A);
  const svd = safeSVD(A_mat);

  // Last column of V (right-singular vectors) is the solution to Ah=0
  // const h = svd.V.getColumn(svd.V.columns - 1);
  const V = svd.rightSingularVectors;
  const h = V.getColumn(V.columns - 1);

  const H = [
    h.slice(0, 3),
    h.slice(3, 6),
    h.slice(6, 9),
  ];

  return H;
}

/**
 * Apply a homography matrix to a point.
 */
export function applyHomography(H, pt) {
  const [x, y] = pt;
  const denom = H[2][0] * x + H[2][1] * y + H[2][2];
  const xPrime = (H[0][0] * x + H[0][1] * y + H[0][2]) / denom;
  const yPrime = (H[1][0] * x + H[1][1] * y + H[1][2]) / denom;
  return [xPrime, yPrime];
}

/**
 * Applies homography to warp a source ImageData to a target rectangle.
 */
export function warpImageData(
  srcImage: ImageData,
  H: number[][],
  outWidth: number,
  outHeight: number
): ImageData {
  // Invert the homography for backward mapping
  const Hinv = inverse(new Matrix(H)).to2DArray();

  const output = new ImageData(outWidth, outHeight);
  const src = srcImage.data;
  const dst = output.data;

  const srcW = srcImage.width;
  const srcH = srcImage.height;

  for (let y = 0; y < outHeight; y++) {
    for (let x = 0; x < outWidth; x++) {
      // Map (x, y) in destination → (x', y') in source
      const denom = Hinv[2][0] * x + Hinv[2][1] * y + Hinv[2][2];
      const srcX = (Hinv[0][0] * x + Hinv[0][1] * y + Hinv[0][2]) / denom;
      const srcY = (Hinv[1][0] * x + Hinv[1][1] * y + Hinv[1][2]) / denom;

      const ix = Math.floor(srcX);
      const iy = Math.floor(srcY);

      // Bounds check
      if (ix < 0 || iy < 0 || ix >= srcW || iy >= srcH) {
        continue; // leave pixel transparent
      }

      const srcIdx = (iy * srcW + ix) * 4;
      const dstIdx = (y * outWidth + x) * 4;

      dst[dstIdx] = src[srcIdx];       // R
      dst[dstIdx + 1] = src[srcIdx + 1]; // G
      dst[dstIdx + 2] = src[srcIdx + 2]; // B
      dst[dstIdx + 3] = src[srcIdx + 3]; // A
    }
  }

  return output;
}

export function cropImageData(
  source: ImageData,
  x: number,
  y: number,
  width: number,
  height: number
): ImageData {
  const output = new ImageData(width, height);
  const src = source.data;
  const dst = output.data;
  const srcWidth = source.width;

  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const srcIdx = ((y + j) * srcWidth + (x + i)) * 4;
      const dstIdx = (j * width + i) * 4;

      dst[dstIdx] = src[srcIdx];       // R
      dst[dstIdx + 1] = src[srcIdx + 1]; // G
      dst[dstIdx + 2] = src[srcIdx + 2]; // B
      dst[dstIdx + 3] = src[srcIdx + 3]; // A
    }
  }

  return output;
}

export function obtainEyePatch(
    frame: HTMLVideoElement,
    faceLandmarks: Point[],
    facePaddingCoefs: [number, number] = [0.4, 0.2],
    faceCropSize: number = 512,
    dstImgSize: [number, number] = [512, 128]
): ImageData {

    // Step 1: Convert HTMLVideoElement to HTMLImageElement
    const videoCanvas = document.createElement('canvas');
    videoCanvas.width = frame.videoWidth;
    videoCanvas.height = frame.videoHeight;
    const ctx = videoCanvas.getContext('2d')!;
    ctx.drawImage(frame, 0, 0);

    // Step 2: Extract ImageData from canvas
    const imageData = ctx.getImageData(0, 0, videoCanvas.width, videoCanvas.height);
   
    // Step 3: Prepare src and dst
    const center = faceLandmarks[4];
    const leftTop = faceLandmarks[103];
    const leftBottom = faceLandmarks[150];
    const rightTop = faceLandmarks[332];
    const rightBottom = faceLandmarks[379];

    let srcPts: Point[] = [leftTop, leftBottom, rightBottom, rightTop];

    // Apply radial padding
    srcPts = srcPts.map(([x, y]) => {
        const dx = x - center[0];
        const dy = y - center[1];
        return [
            x + dx * facePaddingCoefs[0],
            y + dy * facePaddingCoefs[1],
        ];
    });

    const dstPts: Point[] = [
        [0, 0],
        [0, faceCropSize],
        [faceCropSize, faceCropSize],
        [faceCropSize, 0],
    ];

    // Compute homography matrix
    const H = computeHomography(srcPts, dstPts);

    // Step 5: Warp the image
    const warped = warpImageData(imageData, H, faceCropSize, faceCropSize);

    // Step 6: Apply the homography matrix to the facial landmarks
    const warpedLandmarks = faceLandmarks.map(pt => applyHomography(H, pt));

    // Step 7: Generate the crop of the eyes
    const top_eyes_patch = warpedLandmarks[151];
    const bottom_eyes_patch = warpedLandmarks[195];
    const eye_patch = cropImageData(
        warped,
        0,
        Math.round(top_eyes_patch[1]),
        warped.width,
        Math.round(bottom_eyes_patch[1] - top_eyes_patch[1]) 
    );

    // Step 8: Obtain new homography matrix to apply the resize
    const eyePatchSrcPts: Point[] = [
        [0, 0],
        [0, eye_patch.height],
        [eye_patch.width, eye_patch.height],
        [eye_patch.width, 0],
    ];
    const eyePatchDstPts: Point[] = [
        [0, 0],
        [0, dstImgSize[1]],
        [dstImgSize[0], dstImgSize[1]],
        [dstImgSize[0], 0],
    ];
    const eyePatchH = computeHomography(eyePatchSrcPts, eyePatchDstPts);

    // Step 9: Resize the eye patch to the desired output size
    const resizedEyePatch = warpImageData(
        eye_patch,
        eyePatchH,
        dstImgSize[0],
        dstImgSize[1]
    );

    return resizedEyePatch;
}

// ============================================================================
// Face Origin and Head Vector
// ============================================================================

export function computeFaceOrigin3D(
    frame: HTMLVideoElement,
    faceLandmarks: Point[],
    transformationMatrix: MediaPipeMatrix,
): number[] {
  return [0, 0, 0]; // Placeholder for face origin computation
}

function matrixToEuler(matrix: Matrix): [number, number, number] {
  // Extract Euler angles from the rotation matrix
  const pitch = Math.asin(-matrix.get(2, 0));
  const yaw = Math.atan2(matrix.get(2, 1), matrix.get(2, 2));
  const roll = Math.atan2(matrix.get(1, 0), matrix.get(0, 0));
  return [pitch, yaw, roll];
}

function pyrToVector(pitch: number, yaw: number, roll: number): number[] {
  // Convert spherical coordinates to Cartesian coordinates
  const x = Math.cos(pitch) * Math.sin(yaw);
  const y = Math.sin(pitch);
  const z = -Math.cos(pitch) * Math.cos(yaw);
  const vector = new Matrix([[x, y, z]]);

  // Apply roll rotation around the z-axis
  const [cos_r, sin_r] = [Math.cos(roll), Math.sin(roll)];
  const roll_matrix = new Matrix([
    [cos_r, -sin_r, 0],
    [sin_r, cos_r, 0],
    [0, 0, 1],
  ]);

  const rotated_vector = roll_matrix.mmul(vector.transpose()).transpose();
  return rotated_vector.to1DArray();
}

export function getHeadVector(
    tfMatrix: MediaPipeMatrix,
): number[] {

  // Convert MediaPipe matrix (4x4) to a 3x3 rotation matrix
  const rotationMatrix = new Matrix([
    [tfMatrix.data[0], tfMatrix.data[1], tfMatrix.data[2]],
    [tfMatrix.data[4], tfMatrix.data[5], tfMatrix.data[6]],
    [tfMatrix.data[8], tfMatrix.data[9], tfMatrix.data[10]],
  ]);

  // Convert the matrix to euler angles and change the order/direction
  const [pitch, yaw, roll] = matrixToEuler(rotationMatrix);
  const [h_pitch, h_yaw, h_roll] = [-yaw, pitch, roll];

  // Construct a unit vector
  const vector = pyrToVector(h_pitch, h_yaw, h_roll);
  return vector;
}

// ============================================================================
// Gaze State
// ============================================================================

const LEFT_EYE_EAR_LANDMARKS = [362, 385, 387, 263, 373, 380]
const RIGHT_EYE_EAR_LANDMARKS = [133, 158, 160, 33, 144, 153]

export function computeEAR(eyeLandmarks: NormalizedLandmark[], side: 'left' | 'right'): number {
  const EYE_EAR_LANDMARKS = side === 'left' ? LEFT_EYE_EAR_LANDMARKS : RIGHT_EYE_EAR_LANDMARKS;
  const [p1, p2, p3, p4, p5, p6] = EYE_EAR_LANDMARKS.map(idx => eyeLandmarks[idx]);

  const a = Math.sqrt(Math.pow(p2[0] - p6[0], 2) + Math.pow(p2[1] - p6[1], 2));
  const b = Math.sqrt(Math.pow(p3[0] - p5[0], 2) + Math.pow(p3[1] - p5[1], 2));
  const c = Math.sqrt(Math.pow(p1[0] - p4[0], 2) + Math.pow(p1[1] - p4[1], 2));

  return (a + b) / (2.0 * c);
}