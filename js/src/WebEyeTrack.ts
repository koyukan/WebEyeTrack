import { FaceLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import * as tf from '@tensorflow/tfjs';
import { Matrix } from 'ml-matrix';

import { computeFaceOrigin3D, createIntrinsicsMatrix, createPerspectiveMatrix, translateMatrix, faceReconstruction, estimateFaceWidth, getHeadVector, obtainEyePatch, computeEAR } from "./mathUtils";
import { Point, GazeResult } from "./types";
import BlazeGaze from "./BlazeGaze";
import FaceLandmarkerClient from "./FaceLandmarkerClient";

export default class WebEyeTrack {
    
  private blazeGaze: BlazeGaze;
  private faceLandmarkerClient: FaceLandmarkerClient;
  private faceWidthComputed: boolean = false;
  private faceWidthCm: number = -1;
  private perspectiveMatrixSet: boolean = false;
  private perspectiveMatrix: Matrix = new Matrix(4, 4);
  private intrinsicsMatrixSet: boolean = false;
  private intrinsicsMatrix: Matrix = new Matrix(3, 3);

  constructor(videoRef: HTMLVideoElement, canvasRef: HTMLCanvasElement) {
    this.blazeGaze = new BlazeGaze();
    this.faceLandmarkerClient = new FaceLandmarkerClient(videoRef, canvasRef);
  }

  async initialize(): Promise<void> {
    await this.faceLandmarkerClient.initialize();
    await this.blazeGaze.loadModel();
  }

  computeFaceOrigin3D(frame: HTMLVideoElement, faceLandmarks: Point[], faceRT: Matrix): number[] {

    // Estimate the face width in centimeters if not set
    if (this.faceWidthComputed === false) {
      this.faceWidthCm = estimateFaceWidth(faceLandmarks);
      this.faceWidthComputed = true;
    }

    // Perform 3D face reconstruction and determine the pose in 3d cm space
    const [metricTransform, metricFace] = faceReconstruction(
      this.perspectiveMatrix,
      faceLandmarks as [number, number][],
      faceRT,
      this.intrinsicsMatrix,
      this.faceWidthCm,
      frame.videoWidth,
      frame.videoHeight
    );

    // Lastly, compute the gaze origins in 3D space using the metric face
    const faceOrigin3D = computeFaceOrigin3D(
      metricFace
    );

    return faceOrigin3D;
  }

  prepareInput(frame: HTMLVideoElement, result: FaceLandmarkerResult):  [ImageData, number[], number[]] {

    // Get the dimensions of the video frame
    const width = frame.videoWidth;
    const height = frame.videoHeight;

    // If perspective matrix is not set, initialize it
    if (!this.perspectiveMatrixSet) {
      const aspectRatio = width / height;
      this.perspectiveMatrix = createPerspectiveMatrix(aspectRatio);
      this.perspectiveMatrixSet = true;
    }

    // If intrinsics matrix is not set, initialize it
    if (!this.intrinsicsMatrixSet) {
      this.intrinsicsMatrix = createIntrinsicsMatrix(width, height);
    }

    // Convert the normalized landmarks to non-normalized coordinates
    const landmarks = result.faceLandmarks[0];
    const landmarks2d: Point[] = landmarks.map((landmark: NormalizedLandmark) => {
      return [
        Math.floor(landmark.x * width),
        Math.floor(landmark.y * height),
      ];
    });

    // Convert from MediaPipeMatrix to ml-matrix Matrix
    const faceRT = translateMatrix(result.facialTransformationMatrixes[0]);

    // First, extract the eye patch
    const eyePatch = obtainEyePatch(
      frame,
      landmarks2d,
    );

    // Second, compute the face origin in 3D space
    const face_origin_3d = this.computeFaceOrigin3D(
      frame,
      landmarks2d,
      faceRT
    )

    // Third, compute the head vector
    const head_vector = getHeadVector(
      faceRT
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
    const [eyePatch, headVector, faceOrigin3D] = this.prepareInput(frame, result);

    // Compute the EAR ratio to determine if the eyes are open or closed
    let gaze_state: 'open' | 'closed' = 'open';
    if (computeEAR(result.faceLandmarks[0], 'left') < 0.2 || 
        computeEAR(result.faceLandmarks[0], 'right') < 0.2) {
      gaze_state = 'closed';
    }

    // Perform the gaze estimation via BlazeGaze Model (tensorflow.js)
    const inputTensor = tf.browser.fromPixels(eyePatch).toFloat().expandDims(0);

    // Divide the inputTensor by 255 to normalize pixel values
    const normalizedInputTensor = inputTensor.div(tf.scalar(255.0));

    const headVectorTensor = tf.tensor2d(headVector, [1, 3]);
    const faceOriginTensor = tf.tensor2d(faceOrigin3D, [1, 3]);
    const outputTensor = await this.blazeGaze.predict(normalizedInputTensor, headVectorTensor, faceOriginTensor);

    // Extract the 2D gaze point data from the output tensor
    if (!outputTensor || outputTensor.shape.length === 0) {
      throw new Error("BlazeGaze model did not return valid output");
    }
    const normPog = outputTensor.arraySync() as number[][];

    // Return GazeResult
    let gaze_result: GazeResult = {
      facialLandmarks: result.faceLandmarks[0],
      faceRt: result.facialTransformationMatrixes[0],
      faceBlendshapes: result.faceBlendshapes,
      eyePatch: eyePatch,
      headVector: headVector,
      faceOrigin3D: faceOrigin3D,
      metric_transform: {rows: 3, columns: 3, data: [1, 0, 0, 1, 0, 0, 1, 0, 0]}, // Placeholder, should be computed
      gazeState: gaze_state,
      normPog: normPog[0],
      durations: {}
  };

    return gaze_result;
  }
}