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


class Sector(pygame.sprite.Sprite):
    def __init__(self, screen, start_point, end_point):
        pygame.sprite.Sprite.__init__(self)
        self.screen = screen
        self.start_point = start_point
        self.end_point = end_point

    def update(self):
        pygame.draw.line(screen, (0, 0, 0), self.start_point, self.end_point, width=1)

Sectors = pygame.sprite.Group()

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()

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
            Sectors.add(Sector(screen, last_point, current_point))

    screen.blit(img, (0, 0))
    Sectors.update()
    pygame.display.flip()
    pygame.display.update()

    img = webcam.get_image()
