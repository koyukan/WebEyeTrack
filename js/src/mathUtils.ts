import { multiply, inv, matrix } from 'mathjs';
import { Matrix, SVD } from 'ml-matrix';

import { Point } from './types';

function solveHomogeneousSystem(A: number[][]): number[] {
    const matA = new Matrix(A);
    const svd = new SVD(matA);

    // Last column of V (right singular vectors) corresponds to the smallest singular value
    const h = svd.V.getColumn(svd.V.columns - 1);
    return h;
}

/**
 * Estimates a homography matrix from 4 point correspondences.
 */
function computeHomography(src: Point[], dst: Point[]): number[][] {
    const A: number[][] = [];
    for (let i = 0; i < 4; i++) {
        const [x, y] = src[i];
        const [u, v] = dst[i];

        A.push([-x, -y, -1, 0, 0, 0, x * u, y * u, u]);
        A.push([0, 0, 0, -x, -y, -1, x * v, y * v, v]);
    }

    const A_mat = matrix(A);
    // Use SVD or Gaussian elimination to solve Ah=0
    // For brevity, assume library gives us `h` directly here:
    const h = solveHomogeneousSystem(A); // You'd implement this or use a library

    const H = [
        h.slice(0, 3),
        h.slice(3, 6),
        h.slice(6, 9),
    ];

    return H;
}

/**
 * Applies a 3x3 homography matrix to a point.
 */
function applyHomography(H: number[][], pt: Point): Point {
    const [x, y] = pt;
    const denom = H[2][0] * x + H[2][1] * y + H[2][2];
    const xPrime = (H[0][0] * x + H[0][1] * y + H[0][2]) / denom;
    const yPrime = (H[1][0] * x + H[1][1] * y + H[1][2]) / denom;
    return [xPrime, yPrime];
}

/**
 * Performs a perspective warp of the image using canvas.
 */
function warpImage(
    frame: HTMLVideoElement,
    srcPts: Point[],
    dstSize: [number, number]
): HTMLCanvasElement {
    const [width, height] = dstSize;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;

    // Map source quad to destination rectangle using `setTransform`
    // This uses a simple affine approximation – for full perspective warp, WebGL is needed
    ctx.save();
    // draw transformed image using a library or WebGL in real case
    // fallback: draw unwarped region
    ctx.drawImage(frame, 0, 0, width, height);
    ctx.restore();

    return canvas;
}

export function obtainEyePatch(
    // frame: HTMLImageElement | HTMLCanvasElement,
    frame: HTMLVideoElement,
    faceLandmarks: Point[],
    facePaddingCoefs: [number, number] = [0.4, 0.2],
    faceCropSize: number = 512,
    dstImgSize: [number, number] = [512, 128]
): HTMLCanvasElement {
    const center = faceLandmarks[4];
    const leftTop = faceLandmarks[103];
    const leftBottom = faceLandmarks[150];
    const rightTop = faceLandmarks[332];
    const rightBottom = faceLandmarks[379];

    let srcPts = [leftTop, leftBottom, rightBottom, rightTop];

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

    // Homography matrix
    const H = computeHomography(srcPts, dstPts);

    // Warp face crop (approximation)
    const warpedCanvas = warpImage(frame, srcPts, [faceCropSize, faceCropSize]);

    // Warp landmarks
    const warpedLandmarks = faceLandmarks.map(pt => applyHomography(H, pt));

    // Get eye patch
    const topEyeY = Math.floor(warpedLandmarks[151][1]);
    const bottomEyeY = Math.floor(warpedLandmarks[195][1]);

    const eyeCanvas = document.createElement('canvas');
    eyeCanvas.width = dstImgSize[0];
    eyeCanvas.height = dstImgSize[1];
    const eyeCtx = eyeCanvas.getContext('2d')!;
    eyeCtx.drawImage(
        warpedCanvas,
        0, topEyeY,
        faceCropSize, bottomEyeY - topEyeY,
        0, 0,
        dstImgSize[0], dstImgSize[1]
    );

    return eyeCanvas;
}