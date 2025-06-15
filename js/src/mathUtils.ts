import { multiply, inv, matrix } from 'mathjs';
import { Matrix, SVD } from 'ml-matrix';
import { Point } from './types';

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
  const svd = new SVD(A_mat, {autoTranspose: false});

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

export function obtainEyePatch(
    frame: HTMLVideoElement,
    faceLandmarks: Point[],
    facePaddingCoefs: [number, number] = [0.4, 0.2],
    faceCropSize: number = 512,
    dstImgSize: [number, number] = [512, 128]
): ImageData {
    const center = faceLandmarks[4];
    const leftTop = faceLandmarks[103];
    const leftBottom = faceLandmarks[150];
    const rightTop = faceLandmarks[332];
    const rightBottom = faceLandmarks[379];

    // const leftMost = Math.round(Math.min(leftTop[0], leftBottom[0]));
    const leftMost = 0 // frame.videoWidth - Math.round(leftTop[0])
    // const rightMost = Math.round(Math.max(rightTop[0], rightBottom[0]));
    // const rightMost = frame.videoWidth - Math.round(Math.max(leftTop[0], leftBottom[0]));
    const rightMost = Math.round(frame.videoWidth/2);

    // let srcPts: Point[] = [leftTop, leftBottom, rightBottom, rightTop];
    let srcPts: Point[] = [
        [leftMost, 0],
        [leftMost, frame.videoHeight],
        [rightMost, frame.videoHeight],
        [rightMost, 0],
    ];

    // Apply radial padding
    // srcPts = srcPts.map(([x, y]) => {
    //     const dx = x - center[0];
    //     const dy = y - center[1];
    //     return [
    //         x + dx * facePaddingCoefs[0],
    //         y + dy * facePaddingCoefs[1],
    //     ];
    // });

    const dstPts: Point[] = [
        [0, 0],
        [0, faceCropSize],
        [faceCropSize, faceCropSize],
        [faceCropSize, 0],
    ];

    console.log("srcPts", srcPts)
    console.log("dstPts", dstPts)

    // Step 1: Convert HTMLVideoElement to HTMLImageElement
    const videoCanvas = document.createElement('canvas');
    videoCanvas.width = frame.videoWidth;
    videoCanvas.height = frame.videoHeight;
    const ctx = videoCanvas.getContext('2d')!;
    ctx.drawImage(frame, 0, 0);

    // Step 2: Extract ImageData from canvas
    const imageData = ctx.getImageData(0, 0, videoCanvas.width, videoCanvas.height);

    return imageData;
}
