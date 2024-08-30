import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = "img"
images = []
personNames = []
myList = os.listdir(path)
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open("attendence.csv", 'r+') as f:
        cur_date=datetime.now().strftime("%d/%m/%Y")
        
        myDataList = f.readlines()
        nameList = []
        dateList=[]
        if len(myDataList):
            for line in myDataList:
                entry = line.split(',')
                print(entry)
                if len(entry)==4:
                    nameList.append(entry[0])
                    dateList.append(entry[2])
            
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime("%d/%m/%Y")
            f.writelines(f'\n{name},{tStr},{dStr},{"CSE"}')
        else:
            nameList.reverse()
            dateList.reverse()
            for i in range(len(nameList)):
                
                if(name==nameList[i]and cur_date==dateList[i]):
                    break
                elif(name==nameList[i] and cur_date!=dateList[i]):
                    # print(name+" "+" "+nameList[i]+" "+cur_date+" "+dateList[i])
                    time_now = datetime.now()
                    tStr = time_now.strftime('%H:%M:%S')
                    dStr = time_now.strftime("%d/%m/%Y")
                    f.writelines(f'\n{name},{tStr},{dStr},{"CSE"}')
                    break
                    
                
    f.close()

encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
