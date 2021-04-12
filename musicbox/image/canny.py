import cv2

def canny_threshold(img_color, low_threshhold, high_threshhold):
    # img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_grey, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshhold, high_threshhold, 3)
    mask = detected_edges != 0
    dst = original_img * (mask[:,:,None].astype(original_img.dtype))
    return dst