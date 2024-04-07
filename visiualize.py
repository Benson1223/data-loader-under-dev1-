###OpenCV
import cv2


image = cv2.imread('image.jpg')


cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



###Matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('image.jpg')


plt.imshow(image)
plt.axis('off') 
plt.show()


###PIL（Python Imaging Library）
from PIL import Image

image = Image.open('image.jpg')

image.show()