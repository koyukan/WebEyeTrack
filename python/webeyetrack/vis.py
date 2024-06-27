import cv2

def draw_gaze_origin(image, gaze_origin):
    # Draw gaze origin
    draw_image = image.copy()
    x, y = gaze_origin
    cv2.circle(draw_image, (int(x), int(y)), 10, (255, 0, 0), -1)

    return draw_image

def draw_gaze_direction(image, gaze_origin, gaze_dst):
    # Draw gaze direction
    draw_image = image.copy()
    x, y = gaze_origin
    dx, dy = gaze_dst
    cv2.arrowedLine(draw_image, (int(x), int(y)), (int(dx), int(dy)), (255, 0, 0), 2)

    return draw_image