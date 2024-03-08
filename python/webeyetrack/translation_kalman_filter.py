from filterpy.kalman import KalmanFilter
import numpy as np

class TranslationKalmanFilter:
    def __init__(self, scale: float = 0.3):

        # Saving parameters
        self.scale = scale 

        # Define Kalman filter
        # 3 for translation vector only
        self.kf = KalmanFilter(dim_x=3, dim_z=3)

        # State Transition matrix - Identity for translation vector
        self.kf.F = np.eye(3)

        # Measurement matrix - Identity for translation vector
        self.kf.H = np.eye(3)

        # Process noise covariance - Adjust based on your needs for translation
        self.kf.Q = np.diag([1e-3] * 3)  # Adjust the noise level as needed

        # Measurement noise covariance - Adjust based on your needs for translation
        self.kf.R = np.diag([1e-1] * 3)  # Adjust the measurement noise level as needed

        # Error covariance - A smaller value indicates more trust in the model's initial state
        self.kf.P *= 1e-1

    def compute_uncertainty(self, tvec: np.ndarray):

        # Calculate the prior translation vector
        prior_tvec = self.kf.x[:3]

        # Compute the difference between the prior and the new translation vectors
        tvec_difference = np.linalg.norm(prior_tvec - tvec)

        # Compute the uncertainty
        uncertainty = 2 * (1 / (1 + np.exp(-self.scale * tvec_difference)) - 0.5)

        # Ensure the value is bounded between 0 and 1
        uncertainty = max(0, min(1, uncertainty))

        return uncertainty

    def process(self, tvec: np.ndarray):
        # Ensure tvec is a numpy array and flatten it
        tvec = np.array(tvec).flatten()
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(tvec)

        # Predict
        self.kf.predict()

        # Update
        self.kf.update(tvec)

        # The state is just the translation vector now
        return self.kf.x[:3], uncertainty
