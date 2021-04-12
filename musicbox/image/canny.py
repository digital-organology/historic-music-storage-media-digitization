import cv2

def canny_threshold(img_color, low_threshhold, high_threshhold):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # _, img_gray = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    img_blur = cv2.blur(img_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshhold, high_threshhold, 3)
    mask = detected_edges != 0
    dst = img_color * (mask[:,:,None].astype(img_color.dtype))
    return dst