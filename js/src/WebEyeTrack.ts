import { FaceLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import { computeFaceOrigin3D, getHeadVector, obtainEyePatch, computeEAR } from "./mathUtils";

import { Point, GazeResult } from "./types";
import BlazeGaze from "./BlazeGaze";
import FaceLandmarkerClient from "./FaceLandmarkerClient";

export default class WebEyeTrack {
    
  private blazeGaze: BlazeGaze;
  private faceLandmarkerClient: FaceLandmarkerClient;

  constructor(videoRef: HTMLVideoElement, canvasRef: HTMLCanvasElement) {
    console.log('WebEyeTrack constructor');
    this.blazeGaze = new BlazeGaze();
    this.faceLandmarkerClient = new FaceLandmarkerClient(videoRef, canvasRef);
  }

  async initialize(): Promise<void> {
    await this.faceLandmarkerClient.initialize();
    await this.blazeGaze.loadModel();
  }

    prepare_input(frame: HTMLVideoElement, result: FaceLandmarkerResult):  [ImageData, number[], number[]] {

      // Convert the normalized landmarks to non-normalized coordinates
      const width = frame.videoWidth;
      const height = frame.videoHeight;
      const landmarks = result.faceLandmarks[0];
      const landmarks2d: Point[] = landmarks.map((landmark: NormalizedLandmark) => {
        return [
          Math.floor(landmark.x * width),
          Math.floor(landmark.y * height),
        ];
      });

      // First, extract the eye patch
      const eyePatch = obtainEyePatch(
        frame,
        landmarks2d,
      );

      // Second, compute the face origin in 3D space
      const face_origin_3d = computeFaceOrigin3D(
        frame,
        landmarks2d,
        result.facialTransformationMatrixes[0]
      )

      // Third, compute the head vector
      const head_vector = getHeadVector(
        result.facialTransformationMatrixes[0]
      );

      return [
        eyePatch,
        head_vector,
        face_origin_3d
      ];
    }

    async step(frame: HTMLVideoElement): Promise<GazeResult> {

      let result = await this.faceLandmarkerClient.processFrame(frame);
      if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
        throw new Error("No face landmarks detected");
      }

      // Perform preprocessing to obtain the eye patch, head_vector, and face_origin_3d
      const [eyePatch, headVector, faceOrigin3D] = this.prepare_input(frame, result);

      // Compute the EAR ratio to determine if the eyes are open or closed
      let gaze_state: 'open' | 'closed' = 'open';
      if (computeEAR(result.faceLandmarks[0], 'left') < 0.2 || 
          computeEAR(result.faceLandmarks[0], 'right') < 0.2) {
        gaze_state = 'closed';
      }

      // Perform the gaze estimation via BlazeGaze Model (tensorflow.js)

      // Return GazeResult
      let gaze_result: GazeResult = {
        facialLandmarks: result.faceLandmarks[0],
        faceRt: result.facialTransformationMatrixes[0],
        faceBlendshapes: result.faceBlendshapes,
        eyePatch: eyePatch,
        headVector: headVector,
        faceOrigin3D: faceOrigin3D,
        metric_transform: {rows: 3, columns: 3, data: [1, 0, 0, 1, 0, 0, 1, 0, 0]}, // Placeholder, should be computed
        gazeState: gaze_state, // Placeholder, should be computed
        normPog: [0, 0], // Placeholder, should be computed
        durations: {}
    };

    return gaze_result;
  }
}