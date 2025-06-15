import { FaceLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import { obtainEyePatch } from "./mathUtils";

import { Point, GazeResult } from "./types";
import { Matrix } from "mathjs";

export default class WebEyeTrack {
    constructor() {
      console.log('WebEyeTrack constructor');
    }

    prepare_input(frame: HTMLVideoElement, result: FaceLandmarkerResult): ImageData {

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

      return eyePatch;
    }

    step(result: FaceLandmarkerResult, frame: HTMLVideoElement): GazeResult {
      // Perform preprocessing to obtain the eye patch, head_vector, and face_origin_3d
      const eyePatch = this.prepare_input(frame, result);

      // Return GazeResult
      let gaze_result: GazeResult = {
        facialLandmarks: result.faceLandmarks[0],
        faceRt: result.facialTransformationMatrixes[0],
        faceBlendshapes: result.faceBlendshapes,
        eyePatch: eyePatch,
        headVector: [0, 0, 0], // Placeholder, should be computed
        faceOrigin3D: [0, 0, 0], // Placeholder, should be computed
        metric_transform: {rows: 3, columns: 3, data: [1, 0, 0, 1, 0, 0, 1, 0, 0]}, // Placeholder, should be computed
        gazeState: 'open', // Placeholder, should be computed
        normPog: [0, 0], // Placeholder, should be computed
        durations: {}
    };

    return gaze_result;
  }
}