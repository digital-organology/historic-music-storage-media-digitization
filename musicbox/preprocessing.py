import cv2

def binarization(proc):
    _, img_threshold = cv2.threshold(proc.current_image, 60, 255, cv2.THRESH_BINARY)
    img_out = (img_threshold > 0).astype(int)
    proc.current_image = img_out
    return True