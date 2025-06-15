import React, { useEffect, useRef } from 'react';
import { WebcamClient, FaceLandmarkerClient, WebEyeTrack } from 'webeyetrack';

export default function App() {
  const videoRef = useRef(null);
  const eyePatchRef = useRef(null);
  const canvasRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const webEyeTrack = new WebEyeTrack();
  let canvasDimensionFlag = false;

  useEffect(() => {
    async function startWebcamAndLandmarker() {
      if (videoRef.current && canvasRef.current) {
        const webcamClient = new WebcamClient(videoRef.current.id);
        const faceLandmarker = new FaceLandmarkerClient(videoRef.current, canvasRef.current);

        // faceLandmarkerRef.current = faceLandmarker;
        await faceLandmarker.initialize();

        // Start the webcam
        webcamClient.startWebcam(async (frame) => {

          // Update the canvas dimensions to match the video dimensions
          if (!canvasDimensionFlag) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
            canvasDimensionFlag = true;
          }

          let results = await faceLandmarker.processFrame(frame);

          // Further process the results with the WebEyeTrack library
          const gaze_result = webEyeTrack.step(results, frame);

          // Show the eye patch based on the gaze result
          // if (gaze_result && eyePatchRef.current instanceof HTMLCanvasElement) {
          if (gaze_result && eyePatchRef.current) {
            const ctx = eyePatchRef.current.getContext('2d');
            eyePatchRef.current.width = gaze_result.eyePatch.width;
            eyePatchRef.current.height = gaze_result.eyePatch.height;

            if (ctx) {
              // Draw the eye patch canvas onto the display canvas
              ctx.clearRect(0, 0, eyePatchRef.current.width, eyePatchRef.current.height);
              ctx.putImageData(gaze_result.eyePatch, 0, 0);
            }
          }
        });

        // Cleanup: stop the webcam and clear references when the component unmounts
        return () => {
          webcamClient.stopWebcam();
          faceLandmarkerRef.current = null;
        };
      }
    }

    startWebcamAndLandmarker();
  }, []); // Empty dependency array to run only on mount/unmount

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100 relative">
      <video
        id='webcam'
        ref={videoRef}
        autoPlay
        playsInline
        className="absolute z-10 top-0 left-0 h-1/5"
      />
      <canvas
        ref={eyePatchRef}
        className="absolute z-10 top-0 right-0 h-1/5"
      />
      <canvas
        ref={canvasRef}
        className="absolute z-20 top-0 left-0 h-1/5"
      />
    </div>
  );
}
