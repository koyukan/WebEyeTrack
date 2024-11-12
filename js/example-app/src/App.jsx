import React, { useEffect, useRef } from 'react';
import { WebcamClient, FaceLandmarkerClient } from 'webeyetrack';

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
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

          await faceLandmarker.processFrame(frame);
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
        className="absolute z-10 w-full h-auto w-1/2 max-h-full"
      />
      <canvas
        ref={canvasRef}
        // height={600}
        // width={600}
        className="absolute z-20 w-full h-auto w-1/2 max-h-full"
      />
    </div>
  );
}
