import cv2

def canny_threshold(img_color, img_gray, low_threshhold, high_threshhold):
    # img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshhold, high_threshhold, 3)
    mask = detected_edges != 0
    dst = img_color * (mask[:,:,None].astype(img_color.dtype))
    return dst