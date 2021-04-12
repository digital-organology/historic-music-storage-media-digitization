import cv2

def canny_threshold(original_img, processed_img, threshhold):
    #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) -> already done in change_contrast_brightness()
    img_blur = cv2.blur(processed_img, (3,3))
    detected_edges = cv2.Canny(img_blur, threshhold, threshhold * 3, 3)
    mask = detected_edges != 0
    dst = original_img * (mask[:,:,None].astype(original_img.dtype))
    return dst