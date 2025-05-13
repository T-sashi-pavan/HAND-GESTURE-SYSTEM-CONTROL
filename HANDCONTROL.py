import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import screen_brightness_control as sbc
import mediapipe as mp

pyautogui.PAUSE = 0
video = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.7, maxHands=2)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,  # Detect up to 2 faces in the frame
    min_detection_confidence=0.5,  # Minimum confidence for detecting a face
    min_tracking_confidence=0.5 )
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

startDist = None
scale = 0
panStart = None
dragging = False  

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands, img = detector.findHands(frame)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh tessellation (blue landmarks)
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=1)  # Blue landmarks
            )
            # Draw face contours (red outer border)
            mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(20,20,20), thickness=0, circle_radius=0),  # Red outer border
            mp_drawing.DrawingSpec(color=(20, 20, 20), thickness=2, circle_radius=2)  # Red connections
           )

    
    if len(hands) == 2:
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            print("Zoom Gesture Detected")
            if startDist is None:
                length, _, _ = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                startDist = length
            
            length, _, _ = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
            scale = int((length - startDist) // 2)
            print(f"Zoom Scale: {scale}")
        else:
            startDist = None
            scale = 0
        
        pyautogui.scroll(scale)
    
    elif len(hands) == 1:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand["center"]

        if fingers == [0, 1, 1, 1, 1]:
            if panStart is None:
                panStart = (cx, cy)
            else:
                dx, dy = cx - panStart[0], cy - panStart[1]
                pyautogui.moveRel(dx, dy, duration=0.1)
                panStart = (cx, cy)
        else:
            panStart = None

        if fingers == [1, 1, 1, 0, 0]:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
            pyautogui.moveRel(-50, 0, duration=1)
            print("Dragging Left")
        elif fingers == [1, 0, 1, 0, 0]:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
            pyautogui.moveRel(50, 0, duration=1)
            print("Dragging Right")
        elif fingers == [1, 1, 1, 1, 0]:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
            pyautogui.moveRel(0, -50, duration=1)
            print("Dragging Up")
        elif fingers == [1, 1, 0, 0, 1]:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
            pyautogui.moveRel(0, 50, duration=1)
            print("Dragging Down")
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False

        if fingers == [0, 0, 0, 0, 0]:
            pyautogui.press("volumeup")
            print("Volume Up")
        if fingers == [0, 0, 0, 0, 1]:
            pyautogui.press("volumedown")
            print("Volume Down")
        if fingers == [1, 0, 0, 0, 0]:
            sbc.set_brightness(min(sbc.get_brightness()[0] + 10, 100))
            print("Brightness Up")
        if fingers == [0, 1, 0, 0, 0]:
            sbc.set_brightness(max(sbc.get_brightness()[0] - 10, 0))
            print("Brightness Down")
        if fingers == [0, 1, 1, 0, 0]:
            print("Exit Gesture Detected")
            break

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
