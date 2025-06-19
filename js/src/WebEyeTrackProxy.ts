import WebcamClient from "./WebcamClient";
import { GazeResult } from "./types";

// @ts-ignore
import WebEyeTrackWorker from "worker-loader?inline=no-fallback!./WebEyeTrackWorker.ts";
export default class WebEyeTrackProxy {
  private worker: Worker;

  constructor(webcamClient: WebcamClient) {
    this.worker = new WebEyeTrackWorker();
    console.log('WebEyeTrackProxy worker initialized');
    this.worker.onmessage = (mess) =>{
      console.log(`[WebEyeTrackWorker] ${mess.data}`)
      // console.log('[WebEyeTrackProxy] received message', mess);

      // Switch state based on message type
      switch (mess.data.type) {
        case 'ready':
          console.log('[WebEyeTrackProxy] Worker is ready');
          break;
        case 'stepResult':
          // Handle gaze results
          const gazeResult: GazeResult = mess.data.result;
          this.onGazeResults(gazeResult);
          break;
        default:
          console.warn(`[WebEyeTrackProxy] Unknown message type: ${mess.data.type}`);
          break;
      }
    }

    // Initialize the worker
    this.worker.postMessage({ type: 'init' });

    // Start the webcam client and set up the frame callback
    webcamClient.startWebcam(async (frame: ImageData, timestamp: number) => {
      // Send the frame to the worker for processing
      this.worker.postMessage({
        type: 'step',
        payload: { frame, timestamp }
      })
    });
  }

  // Callback for gaze results
  onGazeResults: (gazeResult: GazeResult) => void = () => { 
    console.warn('onGazeResults callback not set');
  }
}
