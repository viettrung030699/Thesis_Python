import time

from deepface import DeepFace
from deepface.basemodels import Boosting
from deepface.commons import functions, realtime
from deepface import DeepFace
import cv2




images_test = [
    "Test_img/ITITIU17096_test_1.jpg",
    "Test_img/ITITIU17067_test_1.jpg",
    "Test_img/ITITIU17067_test_2.jpg",
    "Test_img/ITITIU17073_test_1.jpg",
    "Test_img/ITITIU17073_test_2.jpg",
    "Test_img/ITITIU17098_test_1.jpg",
    # "Test_img/ITITIU17098_test_2.jpg",
    "Test_img/BAFNIU17003_test_1.jpg",
    "Test_img/BAFNIU17003_test_2.jpg",
    "Test_img/BAFNIU17003_test_3.jpg",
    "Test_img/BAFNIU17003_test_4.jpg",
    "Test_img/BAFNIU17003_test_5.jpg"
]

def real_time(img_path):
    resp_obj = DeepFace.find(img_path=img_path, db_path="Dataset/IU_student", model_name="Ensemble", enforce_detection = True, detector_backend = 'mtcnn')
    for i in range(len(img_path)):
        print("Student {}: ".format(i), resp_obj[i].head())
    # print(resp_obj)
    return resp_obj

def face_recognition(img_path):
    demo = []
    values = []
    ids = []
    consines = []
    df = real_time(img_path)
    for i in range(len(df)):
        demo.append(df[i]["identity"].head())
        values.append(df[i]["score"].head())
    for j in range(len(demo)):
        print("ID {}: ".format(j), demo[j][0].split("/")[-1].replace(".jpg", "").split("_")[0])
        id = demo[j][0].split("/")[-1].replace(".jpg", "").split("_")[0]
        ids.append(id)
        print("Values {}: ".format(j), values[j][0])
        cosine = values[j][0]
        consines.append(cosine)
        print("ID {} - Value {}: {} - {}".format(j, j, id, cosine))
    return ids, consines


ids, values = face_recognition(images_test)
print("Test kq id: ", ids)
print("Test kq value", values)

# import cv2 as cv
# font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale = 1
# color = (255, 0, 0)
# thickness = 2
# detector = MTCNN()
# freeze = False
# face_detected = False
# face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
# freezed_frame = 0
# # tic = time.time()
#
# img = []
#
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if freeze == False:
#         detections = detector.detect_faces(frame)
#
#         if len(detections) == 0:
#             face_included_frames = 0
#
#     else:
#         faces = []
#     detected_faces = []
#     face_index = 0
#     #-------------------------------------#
#     # detections = detector.detect_faces(frame)
#     for detection in detections:
#         face_detected = True
#         if face_index == 0:
#             face_included_frames = face_included_frames + 1  # increase frame for a single face
#         face_index = face_index + 1
#         score = detection["confidence"]
#         bounding_box = detection['box']
#         print(bounding_box)
#         x = bounding_box[0]
#         y = bounding_box[1]
#         h = bounding_box[2]
#         w = bounding_box[3]
#         image = cv.imwrite("Demo.png", frame)
#         img.append(image)
#
#         # ids, values = face_recognition(img)
#         # print(ids)
#         cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 255), 2)
#         cv2.putText(frame, 'MTCNN Detector', (x + h, y), font, fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
#
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     # Display the resulting frame
#     cv.imshow('Face recognition', frame)
#     if cv.waitKey(1) == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()

