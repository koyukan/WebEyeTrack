import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-4, measurement_noise=1e-2):
        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Measurement noise covariance
        self.R = measurement_noise * np.eye(2)

        # Process noise covariance
        self.Q = process_noise * np.eye(4)

        # Estimate error covariance
        self.P = np.eye(4)

    def predict(self):
        # Predict the next state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z):
        # z is the measurement [x, y]
        z = np.reshape(z, (2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]

    def step(self, z):
        self.predict()
        return self.update(z)