import math

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
print(WIDTH, HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("pyGame Camera View")

last_point = None
current_point = None
start_point = None


class Sector(pygame.sprite.Sprite):
    def __init__(self, screen, start_point, end_point):
        pygame.sprite.Sprite.__init__(self)
        self.screen = screen
        self.start_point = start_point
        self.end_point = end_point

    def update(self):
        pygame.draw.line(screen, (0, 0, 0), self.start_point, self.end_point, width=1)


Sectors = pygame.sprite.Group()
drawing = False
has_exited_start = False
frame_exited_start = 0
pygame.font.init()
myfont = pygame.font.SysFont("monospace", 15)
font_color = 255
frame = 0
while True:
    screen.blit(pygame.transform.flip(img, True, False), (0, 0))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()
        if e.type == pygame.KEYDOWN:
            drawing = True

    if not drawing:
        label = myfont.render("Press any key to play", 1, (font_color, font_color, font_color))
        screen.blit(label, (100, 100))
        font_color = int(250 + 5 * math.cos(frame / 20))
    else:

        view = pygame.surfarray.array3d(img).transpose([1, 0, 2])

        img_cv2 = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        flipped = np.fliplr(img_cv2)
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        results = handsDetector.process(flippedRGB)
        if results.multi_hand_landmarks is not None and results.multi_handedness[0].classification[0].score > 0.7:
            finger_tip_cords = results.multi_hand_landmarks[0].landmark[8]
            if current_point is None:
                last_point = (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT)
                current_point = last_point
                start_point = last_point
            else:
                if math.sqrt((start_point[0] - current_point[0]) ** 2 + (
                        start_point[1] - current_point[1]) ** 2) > 10 and not has_exited_start:
                    frame_exited_start = frame
                    has_exited_start = True
                    print("out")
                elif math.sqrt((start_point[0] - current_point[0]) ** 2 + (
                        start_point[1] - current_point[1]) ** 2) < 5 and has_exited_start and frame - frame_exited_start > 30:
                    drawing = False
                    print("in")
                    has_exited_start = False
                    Sectors.empty()
                    last_point = None
                    current_point = None
                    start_point = None
                    continue

                last_point = current_point
                current_point = (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT)
                Sectors.add(Sector(screen, last_point, current_point))

        Sectors.update()
        pygame.draw.circle(screen, (0, 0, 255), start_point, 2)

    pygame.display.flip()
    pygame.display.update()

    img = webcam.get_image()
    frame += 1
