import { FaceLandmarker, FilesetResolver, DrawingUtils, FaceLandmarkerResult } from "@mediapipe/tasks-vision";

// References
// https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js#video
export default class FaceLandmarkerClient {
  private faceLandmarker: any;

  constructor() {
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
      outputFacialTransformationMatrixes: true,
      runningMode: "IMAGE",
      numFaces: 1,
    });
  }

  async processFrame(frame: ImageData): Promise<FaceLandmarkerResult | null> {
    if (!this.faceLandmarker) {
      console.error("FaceLandmarker is not loaded yet.");
      return null;
    }

    let result: FaceLandmarkerResult;
    result = await this.faceLandmarker.detect(frame);
    return result;
  }
}
