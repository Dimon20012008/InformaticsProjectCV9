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

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()
    screen.blit(img, (0, 0))
    pygame.display.flip()
    img = webcam.get_image()

    view = pygame.surfarray.array3d(img).transpose([1, 0, 2])

    img_cv2 = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    flipped = np.fliplr(img_cv2)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is None:
        continue
    finger = results.multi_hand_landmarks[0].landmark[8]
    pygame.draw.circle(img, (0, 255, 0), ((1 - finger.x) * WIDTH, finger.y * HEIGHT), 3)
