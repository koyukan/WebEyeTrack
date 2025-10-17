import WebcamClient from "./WebcamClient";
import { GazeResult } from "./types";
import { IDisposable } from "./IDisposable";
import { createWebEyeTrackWorker, WorkerConfig } from "./WorkerFactory";

export default class WebEyeTrackProxy implements IDisposable {
  private worker: Worker;
  private clickHandler: ((e: MouseEvent) => void) | null = null;
  private messageHandler: ((e: MessageEvent) => void) | null = null;
  private _disposed: boolean = false;

  public status: 'idle' | 'inference' | 'calib' = 'idle';

  constructor(webcamClient: WebcamClient, workerConfig?: WorkerConfig) {

    // Initialize the WebEyeTrackWorker using the factory
    this.worker = createWebEyeTrackWorker(workerConfig);
    console.log('WebEyeTrackProxy worker initialized');

    // Store message handler reference for cleanup
    this.messageHandler = (mess) =>{
      // console.log(`[WebEyeTrackWorker] ${mess.data}`)
      // console.log('[WebEyeTrackProxy] received message', mess);

      // Switch state based on message type
      switch (mess.data.type) {
        case 'ready':
          console.log('[WebEyeTrackProxy] Worker is ready');

          // Start the webcam client and set up the frame callback
          webcamClient.startWebcam(async (frame: ImageData, timestamp: number) => {
            // Send the frame to the worker for processing
            if (this.status === 'idle') {
              this.worker.postMessage({
                type: 'step',
                payload: { frame, timestamp }
              })
            }
          });
          break;

        case 'stepResult':
          // Handle gaze results
          const gazeResult: GazeResult = mess.data.result;
          this.onGazeResults(gazeResult);
          break;

        case 'statusUpdate':
          this.status = mess.data.status;
          break;

        default:
          console.warn(`[WebEyeTrackProxy] Unknown message type: ${mess.data.type}`);
          break;
      }
    };

    this.worker.onmessage = this.messageHandler;

    // Initialize the worker
    this.worker.postMessage({ type: 'init' });

    // Store click handler reference for cleanup
    this.clickHandler = (e: MouseEvent) => {
      // Convert px to normalized coordinates
      const normX = (e.clientX / window.innerWidth) - 0.5;
      const normY = (e.clientY / window.innerHeight) - 0.5;
      console.log(`[WebEyeTrackProxy] Click at (${normX}, ${normY})`);
      this.worker.postMessage({ type: 'click', payload: { x: normX, y: normY }});
    };

    // Add mouse handler for re-calibration
    window.addEventListener('click', this.clickHandler);
  }

  // Callback for gaze results
  onGazeResults: (gazeResult: GazeResult) => void = () => {
    console.warn('onGazeResults callback not set');
  }

  /**
   * Disposes the proxy, terminating the worker and removing all event listeners.
   */
  dispose(): void {
    if (this._disposed) {
      return;
    }

    // Remove window click listener
    if (this.clickHandler) {
      window.removeEventListener('click', this.clickHandler);
      this.clickHandler = null;
    }

    // Remove message handler
    if (this.messageHandler) {
      this.worker.onmessage = null;
      this.messageHandler = null;
    }

    // Send disposal message to worker before terminating
    if (this.worker) {
      this.worker.postMessage({ type: 'dispose' });
      this.worker.terminate();
    }

    this._disposed = true;
  }

  /**
   * Returns true if dispose() has been called.
   */
  get isDisposed(): boolean {
    return this._disposed;
  }
}
