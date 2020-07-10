import cv2
from PIL import Image
import numpy as np
im = np.asarray(Image.open("data/val/0/4/depth/400.png"))
#im = cv2.imread("data/val/0/4/depth/400.png", cv2.IMREA)

cv2.imshow('', im.astype(np.uint16))
cv2.waitKey()

cv2.imshow('', im.astype(np.uint8))
cv2.waitKey()