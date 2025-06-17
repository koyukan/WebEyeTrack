import { FaceLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import * as tf from '@tensorflow/tfjs';
import { Matrix } from 'ml-matrix';

import { computeFaceOrigin3D, createIntrinsicsMatrix, createPerspectiveMatrix, translateMatrix, faceReconstruction, estimateFaceWidth, getHeadVector, obtainEyePatch, computeEAR } from "./mathUtils";
import { Point, GazeResult } from "./types";
import BlazeGaze from "./BlazeGaze";
import FaceLandmarkerClient from "./FaceLandmarkerClient";

function generateSupport(
  eyePatches: ImageData[],
  headVectors: number[][],
  faceOrigins3D: number[][],
  normPogs: number[][]
) {
  // Implementation for generating support samples
}

export default class WebEyeTrack {
  
  // Instance variables
  private blazeGaze: BlazeGaze;
  private faceLandmarkerClient: FaceLandmarkerClient;
  private faceWidthComputed: boolean = false;
  private faceWidthCm: number = -1;
  private perspectiveMatrixSet: boolean = false;
  private perspectiveMatrix: Matrix = new Matrix(4, 4);
  private intrinsicsMatrixSet: boolean = false;
  private intrinsicsMatrix: Matrix = new Matrix(3, 3);

  // Public variables
  public loaded: boolean = false;
  public latestGazeResult: GazeResult | null = null;

  constructor(videoRef: HTMLVideoElement, canvasRef: HTMLCanvasElement) {
    this.blazeGaze = new BlazeGaze();
    this.faceLandmarkerClient = new FaceLandmarkerClient(videoRef, canvasRef);

    window.addEventListener('click', async (event: MouseEvent) => {
      const x = event.clientX;
      const y = event.clientY;
      console.log(`🖱️ Global click at: (${x}, ${y}), ${this.loaded}`);
    });
  }

  async initialize(): Promise<void> {
    await this.faceLandmarkerClient.initialize();
    await this.blazeGaze.loadModel();
    this.loaded = true;
  }

  computeFaceOrigin3D(frame: HTMLVideoElement, normFaceLandmarks: Point[], faceLandmarks: Point[], faceRT: Matrix): number[] {

    // Estimate the face width in centimeters if not set
    if (this.faceWidthComputed === false) {
      this.faceWidthCm = estimateFaceWidth(faceLandmarks);
      this.faceWidthComputed = true;
    }

    // Perform 3D face reconstruction and determine the pose in 3d cm space
    const [metricTransform, metricFace] = faceReconstruction(
      this.perspectiveMatrix,
      normFaceLandmarks as [number, number][],
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

    // return faceOrigin3D;
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
      landmarks.map((l: NormalizedLandmark) => [l.x, l.y]),
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

  async adapt(
    eyePatches: ImageData[],
    headVectors: number[][],
    faceOrigins3D: number[][],
    normPogs: number[][],
    stepsInner: number = 5,
    innerLR: number = 1e-5,
    ptType: 'calib' | 'click' = 'calib'
  ) {
    const opt = tf.train.adam(innerLR, 0.85, 0.9, 1e-8);

    // const { support_x, support_y } = generateSupport(
    //   eyePatches,
    //   headVectors,
    //   faceOrigins3D,
    //   normPogs
    // );

    // // Merge with prior calibration data
    // if (calibData.support_x.length > 0) {
    //   for (const key of ['image', 'head_vector', 'face_origin_3d', 'screen_info'] as (keyof SupportX)[]) {
    //     const prior = listToConcatTensor(calibData.support_x.map(s => s[key]), tf.float32, 0);
    //     support_x[key] = tf.concat([support_x[key], prior], 0);
    //   }
    //   const priorY = listToConcatTensor(calibData.support_y, tf.float32, 0);
    //   support_y = tf.concat([support_y, priorY], 0);
    // }

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
    const leftEAR = computeEAR(result.faceLandmarks[0], 'left');
    const rightEAR = computeEAR(result.faceLandmarks[0], 'right');
    if ( leftEAR < 0.2 || rightEAR < 0.2) {
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