import face_recognition
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
known_image = face_recognition.load_image_file('ttt.jpeg')
class_name = 'Eu'
known_encoding = face_recognition.face_encodings(known_image)

while True:
    ret, frame = cap.read()
    try:            
        face_locations = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), model = 'hog')
        matches = []
        unknown_encoding_list = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),face_locations)

        for unknown_encoding in unknown_encoding_list:
            results = face_recognition.compare_faces([known_encoding][0], unknown_encoding, tolerance = 0.6)
            print(results)
            matches.append(results) 
        # print(matches)
    except IndexError:
        print('Face não detectada!!!')
        continue

    for face_tuple, match in zip(face_locations,matches):
        cv2.rectangle(frame,(face_tuple[3],face_tuple[0]),(face_tuple[1],face_tuple[2]),(255,0,255),2)
        cv2.rectangle(frame,(face_tuple[3],face_tuple[2]-35),(face_tuple[1],face_tuple[2]),(255,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        if match[0] == True:
            cv2.putText(frame, class_name, (face_tuple[3]+6,face_tuple[2]-6), font, 1.0, (255,255,255),1)
        else:
            cv2.putText(frame, 'DESCONHECIDO', (face_tuple[3]+6,face_tuple[2]-6), font, 1.0, (255,255,255),1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()