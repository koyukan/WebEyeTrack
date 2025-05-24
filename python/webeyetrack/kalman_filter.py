from filterpy.kalman import KalmanFilter
import numpy as np

def create_kalman_filter(dt=1.0, process_var=1.0, measurement_var=5.0):
    """Creates a Kalman Filter for 2D gaze tracking with constant velocity."""
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State: [x, y, dx, dy]
    kf.x = np.array([0, 0, 0, 0], dtype=float)

    # State transition matrix (assuming constant velocity model)
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])

    # Measurement matrix: we observe position only
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # Measurement noise covariance
    kf.R *= measurement_var  # Measurement noise

    # Process noise covariance (small values for smoother filter)
    q = process_var
    kf.Q = q * np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ])

    # Initial uncertainty
    kf.P *= 500.

    return kf
