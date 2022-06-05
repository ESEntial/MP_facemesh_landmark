import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 이미지 파일의 경우을 사용하세요.:
IMAGE_FILES = ['test_image.jpg']

# 표현되는 랜드마크의 굵기와 반경
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # 작업 전에 BGR 이미지를 RGB로 변환합니다.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:

            # 각 랜드마크를 image에 overlay 시켜줌
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec)
                # connection_drawing_spec=mp_drawing_styles     <---- 이 부분, 눈썹과 눈, 오른쪽 왼쪽 색깔(초록색, 빨강색)
                # .get_default_face_mesh_contours_style())

        # cv2.imwrite('/tmp/annotated_image' +
        #             str(idx) + '.png', annotated_image)

        # 얼굴부분 crop
        
        # haarcascade 불러오기
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 이미지 불러오기
        img = cv2.imread('test_image.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 얼굴 찾기
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped = img[y: y+h, x: x+w]
            resize = cv2.resize(cropped, (180, 180))
            # 이미지 저장하기
            #cv2.imwrite("./images/cutting_faces/test_image.jpg", resize)

            cv2.imshow("crop&resize", resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()




    #    # 랜드마크의 좌표 정보 확인
    #         for id, lm in enumerate(face_landmarks.landmark):
    #             ih, iw, ic = annotated_image.shape
    #             x, y = int(lm.x*iw), int(lm.y*ih)
    #             # print(id,x,y)
    #             # print("INDEX ",id," : (",x,",",y,")")

    #             # 랜드마크 위치 확인,,!
    #             if (id == 93):
    #                 cv2.putText(annotated_image, str(id), (x, y),
    #                             cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)



    #     cv2.imshow("Image_ESEntial",annotated_image)

        # esc 입력시 종료
        key = cv2.waitKey(50000)
        if key == 27:
            break

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as face_mesh:
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("웹캠을 찾을 수 없습니다.")
#             # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
#             continue

#         # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image)

#         # 이미지 위에 얼굴 그물망 주석을 그립니다.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         if results.multi_face_landmarks:

#             # 랜드마크를 얼굴에 표시
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_CONTOURS,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles
#                     .get_default_face_mesh_contours_style())
            
#             # 랜드마크의 좌표 정보 확인
#             for id, lm in enumerate(face_landmarks.landmark):
#                 ih, iw, ic = image.shape
#                 x,y = int(lm.x*iw),int(lm.y*ih)
#                 print(id,x,y)

#                 # 번호 출력 원할시 입력
#                 cv2.putText(image,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

                    
#         # 보기 편하게 이미지를 좌우 반전합니다.
#         cv2.imshow('MediaPipe Face Mesh(ESEntial)', cv2.flip(image, 1))

#         # esc 입력시 종료
#         key = cv2.waitKey(10)
#         if key == 27:
#             break

# cap.release()