import cv2
import mediapipe as mp
import time
import pyautogui
import pydirectinput

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L",";"],
    ["Z","X","C","V","B","N","M",",",".","/","<"]
]

finalText = ""
click_delay = 0     


# ------------------ BUTTON CLASS -------------------
class Button:
    def __init__(self, pos, text, size=(100, 100)):
        self.pos = pos
        self.size = size
        self.text = text


# Create all keyboard buttons
buttonList = []
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        buttonList.append(Button((100*j + 50, 100*i + 50), key))


# ------------- DRAW ALL BUTTONS ------------------
def draw_all(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 30, y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    return img


# ------------- DIST FUNCTION ------------------
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


# ---------------- MAIN LOOP --------------------
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, img = cap.read()
        if not ret:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        img = draw_all(img, buttonList)

        lmList = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))

                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)


        # If hand landmarks found:
        if lmList:
            index_finger = lmList[8]    # Tip of index finger
            middle_finger = lmList[12]  # Tip of middle finger

            # Loop through each button
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                # Hover detection
                if x < index_finger[0] < x + w and y < index_finger[1] < y + h:
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 30, y + 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 255, 255), 5)

                    dist = distance(index_finger, middle_finger)

                    # Click when fingers close
                    if dist < 40 and time.time() - click_delay > 0.5:
                        click_delay = time.time()

                        # Highlight press
                        cv2.rectangle(img, (x, y), (x + w, y + h),
                                      (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 30, y + 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (255, 255, 255), 5)

                        # Handle backspace
                        if button.text == "<":
                            finalText = finalText[:-1]
                        else:
                            finalText += button.text
                            pydirectinput.press(button.text.lower())

                        time.sleep(0.2)


        # Draw final text box
        cv2.rectangle(img, (50, 350), (900, 450), (175, 0, 175), cv2.FILLED)
        cv2.putText(img, finalText, (60, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        cv2.imshow("Virtual Keyboard", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
