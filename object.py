import cv2
import time
import pyttsx3
import threading
import speech_recognition as sr
import mediapipe as mp
from ultralytics import YOLO

# ========== INIT TEXT TO SPEECH ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
for voice in voices:
    if "zira" in voice.id.lower():
        engine.setProperty('voice', voice.id)
        break

engine.say("VisionMate activated. Hello boss.")
engine.runAndWait()

# ========== YOLOv5 SETUP ==========
model = YOLO('yolov5s.pt')
important_classes = ['person']

# ========== VOICE RESPONSES ==========
responses = {
    "hi": "Good morning, boss.",
    "who are you": "I am Edith, your smart assistant.",
    "what is your purpose": "To help you see the world better.",
    "goodbye": "Goodbye boss, shutting down VisionMate."
}

# ========== VOICE COMMAND LISTENER THREAD ==========
def listen_to_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("🎤 Listening...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print("🗣️ You said:", command)
                for key in responses:
                    if key in command:
                        engine.say(responses[key])
                        engine.runAndWait()
                        if "goodbye" in command:
                            exit()
        except:
            pass

threading.Thread(target=listen_to_voice, daemon=True).start()

# ========== MEDIAPIPE HAND DETECTION ==========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=1)
finger_tips = [4, 8, 12, 16, 20]
last_gesture = ""

# ========== TRACKED PEOPLE STORAGE ==========
cap = cv2.VideoCapture(0)
tracked_people = {}  # id -> {'x': center_x, 'name': name}
id_counter = 0
last_spoken = ""
last_time = time.time()

print("👓 VisionMate is running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error.")
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror view

    # ===== YOLOv5 PERSON DETECTION =====
    results = model(frame)[0]
    person_boxes = []

    for *box, conf, cls in results.boxes.data.tolist():
        label = model.names[int(cls)]
        if label == 'person':
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            person_boxes.append((x1, y1, x2, y2, center_x))

    for x1, y1, x2, y2, center_x in person_boxes:
        matched = False
        for pid in tracked_people:
            if abs(tracked_people[pid]['x'] - center_x) < 50:
                name = tracked_people[pid]['name']
                matched = True
                break

        if not matched:
            engine.say("New person detected. Please say your name.")
            engine.runAndWait()
            try:
                with sr.Microphone() as source:
                    recog = sr.Recognizer()
                    recog.adjust_for_ambient_noise(source)
                    audio = recog.listen(source, timeout=5)
                    name = recog.recognize_google(audio)
            except:
                name = f"Person {id_counter}"

            tracked_people[id_counter] = {"x": center_x, "name": name}
            engine.say(f"Hello, {name}")
            engine.runAndWait()
            id_counter += 1

        # Draw box and name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if name != last_spoken and (time.time() - last_time) > 5:
            engine.say(f"{name} is in front of you.")
            engine.runAndWait()
            last_spoken = name
            last_time = time.time()

    # ===== SIGN LANGUAGE GESTURE DETECTION (BASIC) =====
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands_detector.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            lm_list = []
            h, w, _ = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if lm_list:
                fingers = []
                if lm_list[finger_tips[0]][0] > lm_list[finger_tips[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                for i in range(1, 5):
                    if lm_list[finger_tips[i]][1] < lm_list[finger_tips[i] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)
                gesture = ""

                if total_fingers == 0:
                    gesture = "Stop"
                elif total_fingers == 1:
                    gesture = "Hello"
                elif total_fingers == 2:
                    gesture = "Peace"
                elif total_fingers == 5:
                    gesture = "I am fine"

                if gesture and gesture != last_gesture:
                    print("🤟 Sign detected:", gesture)
                    engine.say(gesture)
                    engine.runAndWait()
                    last_gesture = gesture

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ===== SHOW CAMERA FEED =====
    cv2.imshow("VisionMate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        engine.say("VisionMate shutting down. Goodbye boss.")
        engine.runAndWait()
        break

cap.release()
cv2.destroyAllWindows()
