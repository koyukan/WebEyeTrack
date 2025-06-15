import { computeHomography, applyHomography } from './mathUtils';

function pointsAlmostEqual(p1, p2, epsilon = 1e-3) {
  return (
    Math.abs(p1[0] - p2[0]) < epsilon &&
    Math.abs(p1[1] - p2[1]) < epsilon
  );
}

test('computeHomography should map source to destination', () => {
  const src = [
    [100, 100],
    [100, 400],
    [400, 400],
    [400, 100],
  ];

  const dst = [
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
