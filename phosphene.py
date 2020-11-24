# coding=<utf-8>
"""
Author: DoranLyong 
GitGub: https://github.com/DoranLyong
"""


import cv2
import numpy as np 
from scipy import signal


imageSize = 640   # reshape to 640x640
pixelateSize = 64 # it should be a common divisor of the 'imageSize'. 
faceSize = int(imageSize/pixelateSize)
strength = 2



gray = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray , (imageSize, imageSize), interpolation=cv2.INTER_NEAREST) 

H, W = gray.shape 
pix_w, pix_h = (pixelateSize, pixelateSize) # "pixelated"-size 
                        # 16 x 16 pixels 

## **** Pixelated **** ##  
"""
(ref) https://stackoverflow.com/questions/55508615/how-to-pixelate-image-using-opencv-in-python
"""
out_temp = cv2.resize(gray, (pix_w, pix_h), interpolation=cv2.INTER_LINEAR)  # downsample 
pixelate_img = cv2.resize(out_temp, (H, W), interpolation=cv2.INTER_NEAREST)        # upsample 


## **** Phosphene **** ## 
"""
(ref) https://www.mdpi.com/1424-8220/17/10/2439/htm
(ref) https://www.sciencedirect.com/science/article/pii/S0042698909000467#fig3
"""
def gkern(kernlen=21, std=strength):
    """Returns a 2D Gaussian kernel array.
    (ref) https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d



kernel = gkern(faceSize)
print(kernel.shape)
cv2.imshow("kernel", kernel )


temp = np.zeros_like(pixelate_img, dtype=float)

for y in range(0, H, faceSize ):
    for x in range(0, W, faceSize):
        """
        (ref) https://github.com/ashushekar/image-convolution-from-scratch
        """
        temp[y:y+faceSize, x:x+faceSize] = kernel

#temp[2::2, 2::2 ] = 1
cv2.imshow("Phosphene_face", temp )


Phosphene_img = (pixelate_img  * temp).astype('uint8')



## **** Image show **** ### 
images = np.hstack((gray , pixelate_img, Phosphene_img))

cv2.imshow("Show", images)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("outputs.jpg", images)