import "./App.css";
import { useState, useEffect, useRef } from 'react';
import { WebcamClient, WebEyeTrackProxy, type GazeResult } from 'webeyetrack';
import GazeDot from './GazeDot.tsx';
import MemoryCleanupErrorBoundary from './MemoryCleanupErrorBoundary';

function AppContent() {
  const [gaze, setGaze] = useState({ x: 0, y: 0, gazeState: 'closed'});
  const hasInitializedRef = useRef(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamClientRef = useRef<WebcamClient | null>(null);
  const eyeTrackProxyRef = useRef<WebEyeTrackProxy | null>(null);

  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;
    let mounted = true;

    async function startWebEyeTrack() {
      if (!mounted || !videoRef.current) return;

      const webcamClient = new WebcamClient(videoRef.current.id);
      const webEyeTrackProxy = new WebEyeTrackProxy(webcamClient);

      // Store refs for cleanup
      webcamClientRef.current = webcamClient;
      eyeTrackProxyRef.current = webEyeTrackProxy;

      // Define callback for gaze results
      webEyeTrackProxy.onGazeResults = (gazeResult: GazeResult) => {
        if (!mounted) return;

        // Update gaze position and state
        setGaze({
          x: (gazeResult.normPog[0] + 0.5) * window.innerWidth,
          y: (gazeResult.normPog[1] + 0.5) * window.innerHeight,
          gazeState: gazeResult.gazeState
        });
      }
    }

    startWebEyeTrack();

    // Cleanup function
    return () => {
      mounted = false;

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

  return (
    <>
      {/* Gaze Dot */}
      <div className="absolute left-0 right-0 w-full h-full z-100 pointer-events-none">
        <GazeDot x={gaze.x} y={gaze.y} gazeState={gaze.gazeState}/>
      </div>

      {/* Video Element to capture gaze */}
      <video
        id='webcam'
        ref={videoRef}
        autoPlay
        playsInline
        style={{ display: 'none' }}
      />
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
