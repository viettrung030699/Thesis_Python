import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

gray_src = cv2.imread("Test_img/demo1.jpg", 0)
gray = cv2.resize(gray_src, (128, 128))
f = open("Gradient.txt", "w")
f.write("Woops! I have deleted the content!")
im = np.float32(gray) / 255.0

# Calculate gradient
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Calculate the horizontal
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
f.write("\nhorizontal gradient: ")
np.savetxt("gx.csv", gx, fmt='%10.5f', delimiter=",")
print("Gx = ", gx.shape)


# Calculate the vertical
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
np.savetxt("gy.csv", gy, fmt='%10.5f', delimiter=",")
print("Gy = ", gy.shape)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
print("Direction", angle.shape)

np.savetxt("magnitude.csv", mag, fmt='%10.5f', delimiter=",")
np.savetxt("direction.csv", angle, fmt='%10.5f', delimiter=",")

print("Magnitude = ", mag.shape)

plt.subplot(3,3,1),plt.imshow(gray,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

face_detect = dlib.get_frontal_face_detector()
rects = face_detect(gray, 1)

for (i, rect) in enumerate(rects):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 255), 3)


plt.subplot(3,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3),plt.imshow(gx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(gy,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5),plt.imshow(gray,cmap = 'gray')
plt.title('Final'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,6), plt.hist(angle)

plt.show()
f.close()

