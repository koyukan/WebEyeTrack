import cv2

def draw_gaze_origin(image, gaze_origin):
    # Draw gaze origin
    draw_image = image.copy()
    x, y = gaze_origin
    cv2.circle(draw_image, (int(x), int(y)), 10, (255, 0, 0), -1)

    return draw_image