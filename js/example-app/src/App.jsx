import React, { useEffect, useRef } from 'react';
// import WebcamClient from './WebcamClient'; // Import your WebcamClient class
// import FaceLandmarkerClient from './FaceLandmarkerClient'; // Import the FaceLandmarkerClient class
import { WebcamClient, FaceLandmarkerClient } from 'webeyetrack';
// import { WebcamClient } from 'webeyetrack';

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceLandmarkerRef = useRef(null);

  useEffect(() => {
    async function startWebcamAndLandmarker() {
      if (videoRef.current && canvasRef.current) {
        const webcamClient = new WebcamClient(videoRef.current.id);
        const faceLandmarker = new FaceLandmarkerClient(videoRef.current, canvasRef.current);

        // faceLandmarkerRef.current = faceLandmarker;
        await faceLandmarker.initialize();

        // Start the webcam
        webcamClient.startWebcam(async () => {
          if (faceLandmarkerRef.current) {
            await faceLandmarkerRef.current.processFrame();
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
        className="absolute z-10 w-full h-auto max-w-full max-h-full"
      />
      <canvas
        ref={canvasRef}
        className="absolute z-20 w-full h-auto max-w-full max-h-full"
      />
    </div>
  );
}
