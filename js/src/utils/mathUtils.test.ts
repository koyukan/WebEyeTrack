import { ImageData as CanvasImageData } from 'canvas';

// Shim canvas ImageData to match DOM ImageData interface for testing
// Cannot extend in ES5, so we wrap the constructor
const OriginalImageData = CanvasImageData;
global.ImageData = function(dataOrWidth: Uint8ClampedArray | number, widthOrHeight?: number, height?: number) {
  let instance: CanvasImageData;
  if (typeof dataOrWidth === 'number') {
    instance = new OriginalImageData(dataOrWidth, widthOrHeight as number);
  } else {
    instance = new OriginalImageData(dataOrWidth, widthOrHeight as number, height);
  }
  // Add missing colorSpace property from DOM ImageData
  Object.defineProperty(instance, 'colorSpace', {
    value: 'srgb',
    writable: false,
    enumerable: true
  });
  return instance;
} as any;

import { computeHomography, applyHomography, warpImageData } from './mathUtils';
import { Point } from "../types";

function pointsAlmostEqual(p1: Point, p2: Point, epsilon: number = 1e-3) {
  return (
    Math.abs(p1[0] - p2[0]) < epsilon &&
    Math.abs(p1[1] - p2[1]) < epsilon
  );
}

test('computeHomography should map source to destination', () => {
  const src: Point[] = [
    [100, 100],
    [100, 400],
    [400, 400],
    [400, 100],
  ];

  const dst: Point[] = [
    [0, 0],
    [0, 300],
    [300, 300],
    [300, 0],
  ];

  const H = computeHomography(src, dst);

  for (let i = 0; i < src.length; i++) {
    const mapped = applyHomography(H, src[i]);
    // console.log(src[i], mapped, dst[i]);
    expect(pointsAlmostEqual(mapped, dst[i])).toBe(true);
  }
});

function createMockImageData(width: number, height: number, color = [255, 0, 0, 255]): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data.set(color, i);
  }
  return new ImageData(data, width, height);
}

test('warpImageData produces ImageData with desired output size', () => {
  const srcWidth = 100;
  const srcHeight = 100;
  const outputWidth = 200;
  const outputHeight = 100;

  const srcImage = createMockImageData(srcWidth, srcHeight);

  const srcPoints = [
    [0, 0],
    [0, srcHeight],
    [srcWidth, srcHeight],
    [srcWidth, 0],
  ];

  const dstPoints = [
    [0, 0],
    [0, outputHeight],
    [outputWidth, outputHeight],
    [outputWidth, 0],
  ];

  const H = computeHomography(srcPoints, dstPoints);
  const warped = warpImageData(srcImage, H, outputWidth, outputHeight);
  // Verify ImageData properties instead of instanceof (shim doesn't support instanceof)
  expect(warped).toHaveProperty('width');
  expect(warped).toHaveProperty('height');
  expect(warped).toHaveProperty('data');
  expect(warped.width).toBe(outputWidth);
  expect(warped.height).toBe(outputHeight);
});

