import numpy as np

def vector_to_pitch_yaw(vector: np.ndarray) -> np.ndarray:
    # Ensure the input vector is normalized (unit vector)
    vector = vector / np.linalg.norm(vector)
    
    # Extract components
    x, y, z = vector
    
    # Calculate yaw (azimuth), which is the angle around the Z-axis
    yaw = np.arctan2(y, x)  # In radians, between -π and π
    
    # Calculate pitch (elevation), which is the angle from the XY plane
    pitch = np.arcsin(z)  # In radians, between -π/2 and π/2

    # Convert radians to degrees if needed
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    
    return np.array([pitch_deg, yaw_deg])