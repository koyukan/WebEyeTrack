import numpy as np

def rotation_matrix_to_euler_angles(R):
    # Ensure the matrix is 3x3
    assert R.shape == (3, 3)
    
    # Extract pitch, yaw, roll from the rotation matrix
    pitch = np.arcsin(-R[2, 0])  # Pitch around X-axis
    yaw = np.arctan2(R[2, 1], R[2, 2])  # Yaw around Y-axis
    roll = np.arctan2(R[1, 0], R[0, 0])  # Roll around Z-axis (optional)

    # Convert radians to degrees if necessary
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll

def pitch_yaw_to_gaze_vector(pitch, yaw):
    """
    Converts pitch and yaw angles into a 3D gaze direction vector (unit vector).

    Arguments:
    pitch -- pitch angle in degrees
    yaw -- yaw angle in degrees

    Returns:
    A 3D unit gaze direction vector as a numpy array [x, y, z].
    """
    # Convert degrees to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Calculate the 3D gaze vector using spherical-to-Cartesian transformation
    x = np.cos(pitch_rad) * np.cos(yaw_rad)
    y = np.cos(pitch_rad) * np.sin(yaw_rad)
    z = np.sin(pitch_rad)

    # Return the 3D gaze vector
    return np.array([x, y, z])

def vector_to_pitch_yaw(vector):
    # Ensure the input vector is normalized (unit vector)
    vector = vector / np.linalg.norm(vector)
    
    # Extract components
    x, y, z = vector
    
    # Yaw (azimuth angle): the angle in the XY plane from the X-axis
    yaw = np.arctan2(y, x)  # In radians, between -π and π
    
    # Pitch (elevation angle): the angle from the XY plane
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))  # In radians, between -π/2 and π/2

    # Convert radians to degrees if needed
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    
    return pitch_deg, yaw_deg

def get_rotation_matrix_from_vector(vec):
    """
    Generates a rotation matrix that aligns the Z-axis with the input 3D unit vector.
    """
    # Normalize the input vector to ensure it's a unit vector
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec
    
    # Default Z-axis vector
    z_axis = np.array([0, 0, 1])
    
    # Cross product to find the axis of rotation
    axis = np.cross(z_axis, vec)
    axis_len = np.linalg.norm(axis)
    
    if axis_len != 0:
        axis = axis / axis_len  # Normalize the rotation axis
    
    # Angle between the Z-axis and the input vector
    angle = np.arccos(np.dot(z_axis, vec))
    
    # Compute rotation matrix using axis-angle formula (Rodrigues' rotation formula)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R