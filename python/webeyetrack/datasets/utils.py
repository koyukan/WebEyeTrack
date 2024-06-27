
def resize_intrinsics(intrinsic_matrix, original_size, new_size):
    """
    Adjusts the intrinsic matrix for a resized image.
    
    Parameters:
    - intrinsic_matrix: The original intrinsic matrix.
    - original_size: The original image size (width, height).
    - new_size: The new image size (width, height).
    
    Returns:
    - new_intrinsic_matrix: The adjusted intrinsic matrix for the resized image.
    """
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    new_intrinsic_matrix = intrinsic_matrix.copy()
    new_intrinsic_matrix[0, 0] *= scale_x  # fx
    new_intrinsic_matrix[1, 1] *= scale_y  # fy
    new_intrinsic_matrix[0, 2] *= scale_x  # cx
    new_intrinsic_matrix[1, 2] *= scale_y  # cy

    return new_intrinsic_matrix

def resize_annotations(annotations, original_size, new_size):

    # Any 2D point in the image plane needs to be scaled by the same factor
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    annotations.facial_landmarks_2d[0, :] *= scale_x
    annotations.facial_landmarks_2d[1, :] *= scale_y
    annotations.face_origin_2d[0] *= scale_x
    annotations.face_origin_2d[1] *= scale_y
    
    return annotations