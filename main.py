import cv2
import pyttsx3
import time

engine = pyttsx3.init()
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_count = 0
last_speech = 0
cooldown = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Face Detection", frame)

    current_count = len(faces)
    if current_count != last_count and (time.time() - last_speech) > cooldown:
        if current_count == 0:
            engine.say("No faces detected")
        elif current_count == 1:
            engine.say("One face detected")
            print("One face detected")
            print("Coordinates:", faces[0])

        else:
            engine.say(f"{current_count} faces detected")
        engine.runAndWait()
        last_speech = time.time()
        last_count = current_count

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

