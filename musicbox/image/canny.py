import cv2

def canny_threshold(img_color, threshhold):
    img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_grey, (3,3))
    detected_edges = cv2.Canny(img_blur, threshhold, threshhold * 3, 3)
    mask = detected_edges != 0
    dst = img_color * (mask[:,:,None].astype(img_color.dtype))
    return dst