// import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
// const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
// import vision from "@mediapipe/tasks-vision";
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
// const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

export default class FaceLandmarkerClient {
  private faceLandmarker: any;
  private drawingUtils: any;
  private canvasCtx: CanvasRenderingContext2D | null = null;
  private videoElement: HTMLVideoElement;

  constructor(videoElement: HTMLVideoElement, canvasElement: HTMLCanvasElement) {
    this.videoElement = videoElement;
    this.canvasCtx = canvasElement.getContext("2d");
  }

  async initialize() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
        delegate: "GPU",
      },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });

    // Initialize DrawingUtils
    if (this.canvasCtx) {
      this.drawingUtils = new DrawingUtils(this.canvasCtx);
    }
  }

  async processFrame(frame: HTMLVideoElement | null): Promise<any> {
    if (!this.faceLandmarker) {
      console.error("FaceLandmarker is not loaded yet.");
      return;
    }
    console.log("FaceLandmarker: Processing frame...");

    const startTimeMs = performance.now();
    let result: any;
    if (!frame) {
      result = await this.faceLandmarker.detectForVideo(this.videoElement, startTimeMs);
    } else {
      result = await this.faceLandmarker.process(frame, startTimeMs);
    }

    // Clear the canvas before drawing
    if (this.canvasCtx) {
      this.canvasCtx.clearRect(0, 0, this.canvasCtx.canvas.width, this.canvasCtx.canvas.height);
    }

    if (result.faceLandmarks) {
      for (const landmarks of result.faceLandmarks) {
        this.drawLandmarks(landmarks);
      }
    }

    return result;
  }

  drawLandmarks(landmarks: any) {
    if (this.drawingUtils) {
      this.drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      this.drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        { color: "#FF3030" }
      );
      this.drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        { color: "#30FF30" }
      );
      this.drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0" }
      );
      this.drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        { color: "#E0E0E0" }
      );
      this.drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: "#FF3030" }
      );
      this.drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: "#30FF30" }
      );
    }
  }
}
