import { Matrix, inverse } from 'ml-matrix';

export class KalmanFilter2D {
  private x: Matrix;  // State [x, y, vx, vy]
  private F: Matrix;  // State transition matrix
  private H: Matrix;  // Measurement matrix
  private R: Matrix;  // Measurement noise covariance
  private Q: Matrix;  // Process noise covariance
  private P: Matrix;  // Estimate error covariance

  constructor(
    dt = 1.0,
    processNoise = 1e-4,
    measurementNoise = 1e-2
  ) {
    this.x = Matrix.zeros(4, 1);

    this.F = new Matrix([
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]);

    this.H = new Matrix([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ]);

    this.R = Matrix.eye(2).mul(measurementNoise);
    this.Q = Matrix.eye(4).mul(processNoise);
    this.P = Matrix.eye(4);
  }

  predict(): number[] {
    this.x = this.F.mmul(this.x);
    this.P = this.F.mmul(this.P).mmul(this.F.transpose()).add(this.Q);
    return this.x.subMatrix(0, 1, 0, 0).to1DArray(); // Return [x, y]
  }

  update(z: number[]): number[] {
    const zMat = new Matrix([[z[0]], [z[1]]]); // [2, 1]
    const y = zMat.sub(this.H.mmul(this.x));   // innovation
    const S = this.H.mmul(this.P).mmul(this.H.transpose()).add(this.R);
    const K = this.P.mmul(this.H.transpose()).mmul(inverse(S));

    this.x = this.x.add(K.mmul(y));
    const I = Matrix.eye(4);
    this.P = I.sub(K.mmul(this.H)).mmul(this.P);

    return this.x.subMatrix(0, 1, 0, 0).to1DArray(); // Return [x, y]
  }

  step(z: number[]): number[] {
    this.predict();
    return this.update(z);
  }
}