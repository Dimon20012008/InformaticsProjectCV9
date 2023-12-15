import pygame.camera
import pygame.image
import sys
import cv2
import numpy as np
import mediapipe as mp

# https://stackoverflow.com/questions/29673348/how-to-open-camera-with-pygame-in-windows
handsDetector = mp.solutions.hands.Hands()
pygame.camera.init()

cameras = pygame.camera.list_cameras()
webcam = pygame.camera.Camera(cameras[0])
webcam.start()
img = webcam.get_image()

WIDTH = img.get_width()
HEIGHT = img.get_height()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("pyGame Camera View")

last_point = None
current_point = None
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()
    screen.blit(img, (0, 0))
    view = pygame.surfarray.array3d(img).transpose([1, 0, 2])

    img_cv2 = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    flipped = np.fliplr(img_cv2)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        finger_tip_cords = results.multi_hand_landmarks[0].landmark[8]
        if current_point is None:
            last_point = ((1 - finger_tip_cords.x) * WIDTH, finger_tip_cords.y * HEIGHT)
            current_point = ((1 - finger_tip_cords.x) * WIDTH, finger_tip_cords.y * HEIGHT)
        else:
            last_point = current_point
            current_point = ((1 - finger_tip_cords.x) * WIDTH, finger_tip_cords.y * HEIGHT)
            pygame.draw.line(screen, (0, 0, 0), last_point, current_point, width=1)
    pygame.display.flip()
    pygame.display.update()

    img = webcam.get_image()


