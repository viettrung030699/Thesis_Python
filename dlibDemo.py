import time
import dlib
import cv2
import os
from mtcnn import MTCNN

fontScale = 1
color = (255, 0, 0)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

# Camera information:
focus_length = '66mm'
aperture = 'f4'
distance = '1.6m_25'
info = focus_length + '_' + aperture + '_' + distance


def convertImage(src, w, h):
    src = cv2.resize(src, (w, h))
    return src


# Đọc ảnh đầu vào
image = cv2.imread('D:/Test_detector/_DSC4834_contrast-25.jpg')
image = convertImage(image, 1200, 800)


# Khai báo việc sử dụng các hàm của dlib
hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# Thực hiện xác định bằng HOG và SVM
start = time.time()
faces_hog = hog_face_detector(image, 1)
end = time.time()
print("Hog + SVM Execution time: " + str(end - start))

# Vẽ một đường bao màu xanh lá xung quanh các khuôn mặt được xác định ra bởi HOG + SVM
i = 0
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    org = (x + w, y + h)
    cv2.putText(image, 'HOG Detector', org, font,
                fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    i = i + 1
print("Face Detected: ", i)

detector = MTCNN()
start = time.time()
detections = detector.detect_faces(image)
end = time.time()
print(detections)
for detection in detections:
    score = detection["confidence"]
    bounding_box = detection['box']
    print(bounding_box)
    x = bounding_box[0]
    y = bounding_box[1]
    h = bounding_box[2]
    w = bounding_box[3]
    cv2.rectangle(image, (x, y), (x + h, y + w), (0, 255, 255), 2)
    cv2.putText(image, 'MTCNN Detector', (x + h, y), font, fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
    # cv2.rectangle(image, (bounding_box[0], bounding_box[1]),
    #            (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 255), 2)
    # cv2.putText(image, 'MTCNN', (bounding_box[0], bounding_box[1]), font,
    #                 fontScale, color, thickness, cv2.LINE_AA)

print("CNN Execution time: " + str(end - start))

# # Thực hiện xác định bằng CNN
# start = time.time()
# faces_cnn = cnn_face_detector(image, 1)
# end = time.time()
# print("CNN Execution time: " + str(end-start))
#
# # Vẽ một đường bao đỏ xung quanh các khuôn mặt được xác định bởi CNN
# j = 0
# for face in faces_cnn:
#   x = face.rect.left()
#   y = face.rect.top()
#   w = face.rect.right() - x
#   h = face.rect.bottom() - y
#
#   cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
#   org = (x, y)
#   cv2.putText(image, 'CNN', org, font,
#               fontScale, color, thickness, cv2.LINE_AA)
#   j = j + 1
# print("Face Detected: ", j)
cv2.putText(image, info, (10, 30), font, fontScale, (51, 255, 189), thickness, cv2.LINE_AA)
fileName = "Results\{}.png".format(info)
print(fileName)

cv2.imshow("image", image)
image = cv2.resize(image, (250, 150))
status = cv2.imwrite(fileName, image)

print(status)

cv2.waitKey(0)
