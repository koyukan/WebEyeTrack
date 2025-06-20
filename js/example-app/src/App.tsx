import React, { useState, useEffect, useRef, useMemo } from 'react';
import { WebcamClient, WebEyeTrackProxy, GazeResult } from 'webeyetrack';
import GazeDot from './GazeDot.jsx';
import DebugOverlay from './DebugOverlay';

export default function App() {
  const [gaze, setGaze] = useState({ x: 0, y: 0, gazeState: 'closed'});
  const [debugData, setDebugData] = useState({});
  const [perfData, setPerfData] = useState({});
  const hasInitializedRef = useRef(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const eyePatchRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    async function startWebEyeTrack() {
      if (videoRef.current && canvasRef.current) {
        const webcamClient = new WebcamClient(videoRef.current.id);
        const webEyeTrackProxy = new WebEyeTrackProxy(webcamClient);

        // Define callback for gaze results
        webEyeTrackProxy.onGazeResults = (gazeResult: GazeResult) => {
          setGaze({
            x: (gazeResult.normPog[0] + 0.5) * window.innerWidth,
            y: (gazeResult.normPog[1] + 0.5) * window.innerHeight,
            gazeState: gazeResult.gazeState
          });
          setDebugData({
            gazeState: gazeResult.gazeState,
            normPog: gazeResult.normPog,
            headVector: gazeResult.headVector,
            faceOrigin3D: gazeResult.faceOrigin3D,
          });
          setPerfData(gazeResult.durations);
        }
      }
    }

    startWebEyeTrack();
  }, []); // Empty dependency array to run only on mount/unmount

  return (
    <>
      <div className="absolute left-0 right-0 w-full h-full z-100 pointer-events-none">
        <GazeDot x={gaze.x} y={gaze.y} gazeState={gaze.gazeState}/>
      </div>
      <div className="flex flex-col justify-end h-screen w-full bg-black relative">
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

        {/* ✅ Show the debug data overlay */}
        <div className="flex flex-col items-center md:flex-row md:justify-between m-8 mb-16">
          <DebugOverlay data={debugData}/>
          <DebugOverlay data={perfData}/>
        </div>
      </div>
    </>
  );
}
