import React, { useState, useEffect, useRef } from 'react';
import { WebcamClient, WebEyeTrackProxy, GazeResult } from 'webeyetrack';
import GazeDot from './GazeDot.jsx';
import DebugOverlay from './DebugOverlay';
import { drawMesh } from './drawMesh';
import MemoryCleanupErrorBoundary from './MemoryCleanupErrorBoundary';
import CalibrationOverlay from './components/CalibrationOverlay';

function AppContent() {
  const [gaze, setGaze] = useState({ x: 0, y: 0, gazeState: 'closed'});
  const [debugData, setDebugData] = useState({});
  const [perfData, setPerfData] = useState({});
  const menuRef = useRef<HTMLDivElement | null>(null);

  // Toggles
  const [showCamera, setShowCamera] = useState(true);
  const [showFaceMesh, setShowFaceMesh] = useState(true);
  const [showEyePatch, setShowEyePatch] = useState(true);
  const [showDebug, setShowDebug] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const [showCalibration, setShowCalibration] = useState(false);

  const hasInitializedRef = useRef(false);
  const hasCanvasSizeRef = useRef(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const eyePatchRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const webcamClientRef = useRef<WebcamClient | null>(null);
  const eyeTrackProxyRef = useRef<WebEyeTrackProxy | null>(null);

  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;
    let mounted = true;

    async function startWebEyeTrack() {
      if (!mounted || !videoRef.current || !canvasRef.current) return;

      videoRef.current.onloadedmetadata = () => {
        if (!hasCanvasSizeRef.current && videoRef.current) {
          hasCanvasSizeRef.current = true;

          // Set canvas size based on actual video dimensions
          const width = videoRef.current.videoWidth;
          const height = videoRef.current.videoHeight;
          canvasRef.current!.width = width;
          canvasRef.current!.height = height;

          console.log(`Canvas size set to: ${width}x${height}`);
        }
      }

      const webcamClient = new WebcamClient(videoRef.current.id);
      const webEyeTrackProxy = new WebEyeTrackProxy(webcamClient, {
        workerUrl: '/webeyetrack.worker.js'
      });

      // Store refs for cleanup
      webcamClientRef.current = webcamClient;
      eyeTrackProxyRef.current = webEyeTrackProxy;

      // Define callback for gaze results
      webEyeTrackProxy.onGazeResults = (gazeResult: GazeResult) => {
        if (!mounted) return;

        // Ensure gazeResult is not null or undefined
        if (!gazeResult) {
          console.error("Gaze result is null or undefined");
          return;
        }

        // Show EyePatch and Face Mesh
        if (eyePatchRef.current && gazeResult.eyePatch) {
          eyePatchRef.current!.width = gazeResult.eyePatch.width;
          eyePatchRef.current!.height = gazeResult.eyePatch.height;
          const eyePatchCtx = eyePatchRef.current!.getContext('2d');
          if (eyePatchCtx) {
            eyePatchCtx.clearRect(0, 0, eyePatchRef.current!.width, eyePatchRef.current!.height);
            eyePatchCtx.putImageData(gazeResult.eyePatch, 0, 0);
          }
        }
        drawMesh(gazeResult, canvasRef.current!)

        // Update gaze position and state
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

    startWebEyeTrack();

    // Cleanup function
    return () => {
      mounted = false;
      hasInitializedRef.current = false; // Reset for StrictMode remounting

      // Dispose resources
      if (webcamClientRef.current) {
        webcamClientRef.current.dispose();
        webcamClientRef.current = null;
      }

      if (eyeTrackProxyRef.current) {
        eyeTrackProxyRef.current.dispose();
        eyeTrackProxyRef.current = null;
      }

      console.log('App cleanup completed');
    };
  }, []); // Empty dependency array to run only on mount/unmount

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    }

    if (menuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [menuOpen]);

  return (
    <>
      {/* Calibration Overlay */}
      {showCalibration && eyeTrackProxyRef.current && (
        <CalibrationOverlay
          tracker={eyeTrackProxyRef.current}
          onComplete={() => {
            console.log('Calibration completed successfully');
            setShowCalibration(false);
          }}
          onCancel={() => {
            console.log('Calibration cancelled');
            setShowCalibration(false);
          }}
        />
      )}

      {/* Burger Menu */}
      <div className="flex items-center top-0 left-0 w-full h-12 bg-white relative z-50">
        <div ref={menuRef} className="relative">
          <button
            className="ml-1 p-2 h-10 border-2 border-gray-300 rounded-md shadow-md md:hidden"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            ☰
          </button>

          {/* Dropdown */}
          <div
            className={`absolute w-48 top-12 left-0 bg-white p-4 rounded-md shadow-lg space-y-2 transition-opacity duration-200 ${
              menuOpen ? 'block' : 'hidden'
            } md:w-full md:block md:static md:shadow-none md:pl-2 md:p-0 md:space-y-0 md:flex md:gap-4`}
          >
            <label className="block md:inline-flex items-center gap-2">
              <input type="checkbox" checked={showCamera} onChange={() => setShowCamera(!showCamera)} />
              Show Camera
            </label>
            <label className="block md:inline-flex items-center gap-2">
              <input type="checkbox" checked={showFaceMesh} onChange={() => setShowFaceMesh(!showFaceMesh)} />
              Show Face Mesh
            </label>
            <label className="block md:inline-flex items-center gap-2">
              <input type="checkbox" checked={showEyePatch} onChange={() => setShowEyePatch(!showEyePatch)} />
              Show Eye Patch
            </label>
            <label className="block md:inline-flex items-center gap-2">
              <input type="checkbox" checked={showDebug} onChange={() => setShowDebug(!showDebug)} />
              Show Debug Data
            </label>
          </div>
        </div>

        {/* Calibrate Button */}
        <button
          className="ml-auto mr-2 px-4 py-2 bg-blue-600 text-white rounded-md shadow-md hover:bg-blue-700 transition font-semibold"
          onClick={() => setShowCalibration(true)}
          disabled={!eyeTrackProxyRef.current}
        >
          Calibrate
        </button>
      </div>

      {/* Gaze Dot */}
      <div className="absolute left-0 right-0 w-full h-full z-100 pointer-events-none">
        <GazeDot x={gaze.x} y={gaze.y} gazeState={gaze.gazeState}/>
      </div>

      {/* Main layout */}
      <div className="flex flex-col min-h-screen justify-between w-full bg-black">
        <div className="flex flex-col md:flex-row md:justify-between">
          <div className="w-full relative md:w-1/5">
            <canvas
              ref={canvasRef}
              className="w-full h-full absolute z-20"
              hidden={!showFaceMesh}
            />
            <video
              id='webcam'
              ref={videoRef}
              autoPlay
              playsInline
              // hidden={!showCamera}
              // className="w-full z-10"
              className={`w-full z-10 ${showCamera ? '' : 'opacity-0'}`}
              
            />
          </div>

          <div>
            <canvas
              ref={eyePatchRef}
              className="md:max-width-[480px] w-full md:h-full z-20"
              hidden={!showEyePatch}
            />
          </div>

        </div>

        {/* ✅ Show the debug data overlay */}
        <div className="flex flex-col items-center md:flex-row md:justify-between m-8 mb-16">
          { showDebug && (
            <>
              <DebugOverlay data={debugData}/>
              <DebugOverlay data={perfData}/>
            </>
          )
          }
        </div>
      </div>
    </>
  );
}

export default function App() {
  const handleCleanup = () => {
    console.log('Error boundary triggered cleanup');
  };

  return (
    <MemoryCleanupErrorBoundary onCleanup={handleCleanup}>
      <AppContent />
    </MemoryCleanupErrorBoundary>
  );
}
