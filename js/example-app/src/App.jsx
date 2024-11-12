import React, { useEffect, useRef } from 'react';
// import WebcamClient from './WebcamClient'; // Import your WebcamClient class
import {WebcamClient} from "webeyetrack";

export default function App() {
  // Reference to the video element
  const videoRef = useRef(null);

  useEffect(() => {
    // Instantiate the WebcamClient once the component mounts
    if (videoRef.current) {
      const webcamClient = new WebcamClient(videoRef.current.id);

      // Start the webcam with a frame callback for custom processing
      const frameCallback = (frame) => {
        console.log('Frame captured:', frame);
        // Additional frame processing logic can be added here
      };

      // Start the webcam
      webcamClient.startWebcam(frameCallback);

      // Clean up: stop the webcam when the component unmounts
      return () => {
        webcamClient.stopWebcam();
      };
    }
  }, []); // Empty dependency array to run only on mount/unmount

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-100">
      <h1 className="text-3xl font-bold text-gray-800">
        Webcam Video Stream
      </h1>
      <video
        id="webcam"
        ref={videoRef}
        autoPlay
        playsInline
        className="border-4 border-gray-800 rounded-md max-w-full max-h-full"
      />
    </div>
  );
}
