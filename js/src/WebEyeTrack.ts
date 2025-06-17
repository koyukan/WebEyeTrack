import { FaceLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import * as tf from '@tensorflow/tfjs';
import { Matrix } from 'ml-matrix';

import { computeFaceOrigin3D, createIntrinsicsMatrix, createPerspectiveMatrix, translateMatrix, faceReconstruction, estimateFaceWidth, getHeadVector, obtainEyePatch, computeEAR } from "./mathUtils";
import { Point, GazeResult } from "./types";
import BlazeGaze from "./BlazeGaze";
import FaceLandmarkerClient from "./FaceLandmarkerClient";

interface SupportX {
  eyePatches: tf.Tensor;
  headVectors: tf.Tensor;
  faceOrigins3D: tf.Tensor;
}

function generateSupport(
  eyePatches: ImageData[],
  headVectors: number[][],
  faceOrigins3D: number[][],
  normPogs: number[][]
): { supportX: SupportX, supportY: tf.Tensor } {

  // Implementation for generating support samples
  const supportX: SupportX = {
    eyePatches: tf.stack(eyePatches.map(patch => tf.browser.fromPixels(patch)), 0).toFloat().div(tf.scalar(255.0)), // Convert ImageData to tensor
    headVectors: tf.tensor(headVectors, [headVectors.length, 3], 'float32'),
    faceOrigins3D: tf.tensor(faceOrigins3D, [faceOrigins3D.length, 3], 'float32')
  };

  // Convert normPogs to tensor
  const supportY = tf.tensor(normPogs, [normPogs.length, 2], 'float32');

  return { supportX, supportY };
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
  public calibData: {
    supportX: SupportX[],
    supportY: tf.Tensor[],
    timestamps: number[],
    ptType: ('calib' | 'click')[]
  } = {
    supportX: [],
    supportY: [],
    timestamps: [],
    ptType: ['calib']
  };

  // Configuration
  public maxPoints: number = 100;
  public clickTTL: number = 60; // Time-to-live for click points in seconds

  constructor(
      videoRef: HTMLVideoElement, 
      canvasRef: HTMLCanvasElement,
      maxPoints: number = 100,
      clickTTL: number = 60 // Time-to-live for click points in seconds
    ) {

    // Initialize services
    this.blazeGaze = new BlazeGaze();
    this.faceLandmarkerClient = new FaceLandmarkerClient(videoRef, canvasRef);
    
    // Storing configs
    this.maxPoints = maxPoints;
    this.clickTTL = clickTTL;

    // Handling mouse clicks for calibration
    window.addEventListener('click', this.handleClick.bind(this), false);
    console.log('👁️ WebEyeTrack initialized');
  }

  async initialize(): Promise<void> {
    await this.faceLandmarkerClient.initialize();
    await this.blazeGaze.loadModel();
    this.loaded = true;
  }

  pruneCalibData() {
    
    // Prune the calibration data to keep only the last maxPoints points
    if (this.calibData.supportX.length > this.maxPoints) {
      this.calibData.supportX = this.calibData.supportX.slice(-this.maxPoints);
      this.calibData.supportY = this.calibData.supportY.slice(-this.maxPoints);
      this.calibData.timestamps = this.calibData.timestamps.slice(-this.maxPoints);
      this.calibData.ptType = this.calibData.ptType.slice(-this.maxPoints);
    }

    // Apply time-to-live pruning for 'click' points
    const currentTime = Date.now();
    const ttl = this.clickTTL * 1000; // Convert seconds to milliseconds

    // Filter all together
    const filteredIndices = this.calibData.timestamps.map((timestamp, index) => {
      return (currentTime - timestamp <= ttl || this.calibData.ptType[index] !== 'click') ? index : -1;
    }).filter(index => index !== -1);
    this.calibData.supportX = filteredIndices.map(index => this.calibData.supportX[index]);
    this.calibData.supportY = filteredIndices.map(index => this.calibData.supportY[index]);
    this.calibData.timestamps = filteredIndices.map(index => this.calibData.timestamps[index]);
    this.calibData.ptType = filteredIndices.map(index => this.calibData.ptType[index]);
  }

  handleClick(event: MouseEvent) {
    const x = event.clientX;
    const y = event.clientY;
    console.log(`🖱️ Global click at: (${x}, ${y}), ${this.loaded}`);

    if (this.loaded && this.latestGazeResult) {
      // Adapt the model based on the click position
      this.adapt(
        [this.latestGazeResult?.eyePatch as ImageData],
        [this.latestGazeResult?.headVector as number[]],
        [this.latestGazeResult?.faceOrigin3D as number[]],
        [[x / window.innerWidth - 0.5, y / window.innerHeight - 0.5]], // Normalize click position
      );
    }
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

  adapt(
    eyePatches: ImageData[],
    headVectors: number[][],
    faceOrigins3D: number[][],
    normPogs: number[][],
    stepsInner: number = 5,
    innerLR: number = 1e-5,
    ptType: 'calib' | 'click' = 'calib'
  ) {

    // Prune old calibration data
    this.pruneCalibData();

    // Prepare the inputs
    const opt = tf.train.adam(innerLR, 0.85, 0.9, 1e-8);
    let { supportX, supportY } = generateSupport(
      eyePatches,
      headVectors,
      faceOrigins3D,
      normPogs
    );

    // Append the new support data to the calibration data
    this.calibData.supportX.push(supportX);
    this.calibData.supportY.push(supportY);
    this.calibData.timestamps.push(Date.now());
    this.calibData.ptType.push(ptType);

    // Now extend the supportX and supportY tensors with prior calib data
    let tfEyePatches: tf.Tensor;
    let tfHeadVectors: tf.Tensor;
    let tfFaceOrigins3D: tf.Tensor;
    let tfSupportY: tf.Tensor;
    if (this.calibData.supportX.length > 1) {
      tfEyePatches = tf.concat(this.calibData.supportX.map(s => s.eyePatches), 0);
      tfHeadVectors = tf.concat(this.calibData.supportX.map(s => s.headVectors), 0);
      tfFaceOrigins3D = tf.concat(this.calibData.supportX.map(s => s.faceOrigins3D), 0);
      tfSupportY = tf.concat(this.calibData.supportY, 0);
    } else {
      // If there is no prior calibration data, we use the current supportX and supportY
      tfEyePatches = supportX.eyePatches;
      tfHeadVectors = supportX.headVectors;
      tfFaceOrigins3D = supportX.faceOrigins3D;
      tfSupportY = supportY;
    }

    for (let i = 0; i < stepsInner; i++) {
      opt.minimize(() => {
        const supportPreds = this.blazeGaze.predict(
          tfEyePatches,
          tfHeadVectors,
          tfFaceOrigins3D
        );
        if (!supportPreds) {
          throw new Error("BlazeGaze model did not return valid predictions");
        }
        const loss = tf.losses.meanSquaredError(
          tfSupportY,
          supportPreds
        );
        loss.data().then(l => console.log(`Support loss (${i}), innerLR=${innerLR}: ${l[0].toFixed(4)}`));

        return loss.asScalar();
      });
    }

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
    const outputTensor = this.blazeGaze.predict(normalizedInputTensor, headVectorTensor, faceOriginTensor);

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

    // Update the latest gaze result
    this.latestGazeResult = gaze_result;

    return gaze_result;
  }
}