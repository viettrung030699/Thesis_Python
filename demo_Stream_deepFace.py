import cv2
import dlib
import os

# #Read Image
from imutils import face_utils
import time
# faceCascade = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# video_capture = cv2.VideoCapture(0)

# Call the trained model yml file to recognize faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Train/training_set.yml")

# Names corresponding to each id
names = []
for users in os.listdir("Dataset/processed_number/testing_set"):
    names.append(users)

count = 0
# Video capture
while True:
    start_time = time.time()
    # Read Image
    img = cv2.imread("D:\Thesis_test\_DSC4812.jpg")
    img = img.resize(img, (600, 400))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Call the facial detection model
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray_image, 1)

    # faces = faceCascade.detectMultiScale(
    #     gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE
    # )

    # Accordingly add the names
    count = count + 1
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray_image[y : y + h, x : x + w])
        print(id-1, names[id - 1], confidence)
        if id:
            cv2.putText(
                img,
                names[id - 1],
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                img,
                "Unknown",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
    cv2.imshow("Recognize", img)
    end_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# video_capture.release()
print("Result: ",id-1, names[id - 1], confidence)
print("Execution time: ", (end_time - start_time))
cv2.destroyAllWindows()