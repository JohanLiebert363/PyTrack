import cv2
import time
import winsound

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_fullbody.xml'
)

cooldown = 2.5
last_alert = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Draw bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Body", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Face and Body Detection", frame)

    now = time.time()

    # 🔴 PRIORITY ALERT SYSTEM
    if now - last_alert >= cooldown:
        if len(faces) > 0:
            print("HUMAN CONFIRMED (FACE)")
            winsound.Beep(1500, 400)  # higher pitch
            last_alert = now

        elif len(bodies) > 0:
            print("POSSIBLE HUMAN (BODY)")
            winsound.Beep(1000, 250)  # lower pitch
            last_alert = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

