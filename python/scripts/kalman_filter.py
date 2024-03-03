from filterpy.kalman import KalmanFilter
import numpy as np

class GazeFilter:
    def __init__(self, scale: float = 0.3):

        # Saving parameters
        self.scale = scale 

        # Define Kalman filter
        # 6 for [rotation vector, translation]
        self.kf = KalmanFilter(dim_x=2, dim_z=2)

        # State Transition matrix - Identity for pose (rotation + translation)
        self.kf.F = np.eye(2)

        # Measurement matrix - Identity for pose
        self.kf.H = np.eye(2)

        # Process noise covariance - Adjust based on your needs
        self.kf.Q = np.diag([1e-5, 1e-5])  # Assuming translation might have a bit more noise

        # Measurement noise covariance - Adjust based on your needs
        self.kf.R = np.diag([1e-2, 1e-1])  # Assuming translation measurements might be less precise

        # Error covariance - Adjust based on your needs
        self.kf.P *= 1e-1

    def compute_uncertainty(self, z: np.ndarray):

        # Calculate the prior gaze
        prior_gaze = self.kf.x

        # Compute the difference between the prior and post-processed poses
        pose_difference = np.linalg.norm(prior_gaze - z)

        # Compute the uncertainty
        uncertainty = 2 * (1 / (1 + np.exp(-self.scale * pose_difference)) - 0.5)

        # Ensure the value is bounded between 0 and 1
        uncertainty = max(0, min(1, uncertainty))

        return uncertainty

    def process(self, gaze_vector: np.ndarray):
        # Convert to numpy arrays for concatenation
        gvec = np.array(gaze_vector).flatten()
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(gvec)

        # Predict
        self.kf.predict()

        # Update
        self.kf.update(gvec)

        # Separate the state into rotation and translation for return
        return self.kf.x, uncertainty