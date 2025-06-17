import React, { useState, useEffect, useRef } from 'react';
import { WebcamClient, WebEyeTrack } from 'webeyetrack';

import GazeDot from './GazeDot.jsx';
import DebugOverlay from './DebugOverlay.tsx';

export default function App() {
  const [gaze, setGaze] = useState({ x: 0, y: 0, gazeState: 'closed'});
  const [debugData, setDebugData] = useState({});
  const hasInitializedRef = useRef(false);
  const videoRef = useRef(null);
  const eyePatchRef = useRef(null);
  const canvasRef = useRef(null);
  let canvasDimensionFlag = false;

  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    async function startWebcamAndLandmarker() {
      if (videoRef.current && canvasRef.current) {
        const webcamClient = new WebcamClient(videoRef.current.id);
        const webEyeTrack = new WebEyeTrack(videoRef.current, canvasRef.current);
        await webEyeTrack.initialize();

        // Start the webcam
        webcamClient.startWebcam(async (frame) => {

          // Update the canvas dimensions to match the video dimensions
          if (!canvasDimensionFlag) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
            canvasDimensionFlag = true;
          }

          // Further process the results with the WebEyeTrack library
          const gaze_result = await webEyeTrack.step(frame);

          // Set the gaze coordinates
          if (gaze_result) {
            setGaze({
              x: (gaze_result.normPog[0]+0.5) * window.innerWidth,
              y: (gaze_result.normPog[1]+0.5) * window.innerHeight,
              gazeState: gaze_result.gazeState,
            });

            // Update debug data
            setDebugData({
              gazeState: gaze_result.gazeState,
              normPog: gaze_result.normPog,
              headVector: gaze_result.headVector,
              faceOrigin3D: gaze_result.faceOrigin3D,
            });
          }

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
        };
      }
    }

    startWebcamAndLandmarker();
  }, []); // Empty dependency array to run only on mount/unmount

  return (
    <>
      <div className="absolute left-0 right-0 w-full h-full z-100 pointer-events-none">
        <GazeDot x={gaze.x} y={gaze.y} gazeState={gaze.gazeState}/>
      </div>
      <div className="flex items-center justify-center h-screen w-full bg-black relative">
        <video
          id='webcam'
          ref={videoRef}
          autoPlay
          playsInline
          className="absolute z-10 top-0 left-0 h-1/5"
        />
        <canvas
          ref={eyePatchRef}
          className="absolute z-10 top-0 right-0 h-1/8"
        />
        <canvas
          ref={canvasRef}
          className="absolute z-20 top-0 left-0 h-1/5"
        />
      </div>

      {/* ✅ Show the debug data overlay */}
      <DebugOverlay data={debugData} />
    </>
  );
}
