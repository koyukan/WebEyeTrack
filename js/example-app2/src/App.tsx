import React, { useState, useEffect, useRef, useMemo } from 'react';
import { WebcamClient, WebWorkerLibTs, GazeResult } from 'webeyetrack';
import GazeDot from './GazeDot.jsx';
import DebugOverlay from './DebugOverlay';

// import Worker from "worker?worker"
// const worker = new Worker();
// const url = new URL('./worker.js', import.meta.url);
// const worker = new Worker(url);
// const WebEyeTrack = new WebEyeTrackProxy();

// worker.onmessage = function (e) {
//   console.log(e.data)
// }

export default function App() {
  const [gaze, setGaze] = useState({ x: 0, y: 0, gazeState: 'closed'});
  const [debugData, setDebugData] = useState({});
  const [perfData, setPerfData] = useState({});
  const hasInitializedRef = useRef(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const eyePatchRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  let canvasDimensionFlag = false;

  const libTs = useMemo(()=>{
      return new WebWorkerLibTs()
  },[])

  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    async function startWebcamAndLandmarker() {
      if (videoRef.current && canvasRef.current) {
        const webcamClient = new WebcamClient(videoRef.current.id);

        // Start the webcam
        webcamClient.startWebcam(async (frame: HTMLVideoElement) => {});
      }
    }

    startWebcamAndLandmarker();
  }, []); // Empty dependency array to run only on mount/unmount

  const handleClick = () => {
    libTs.sendMessage()
  }

  return (
    <>
      <div className="App">
        <button onClick={handleClick}>Send Message</button>
      </div>
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
      <DebugOverlay data={debugData} position="bottom-2 right-2"/>
      <DebugOverlay data={perfData} position="bottom-2 left-2"/>
    </>
  );
}
