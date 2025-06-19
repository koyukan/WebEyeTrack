export function convertVideoFrameToImageData(frame: HTMLVideoElement): ImageData {
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