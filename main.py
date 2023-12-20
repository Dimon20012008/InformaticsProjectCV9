import math

import pygame.camera
import pygame.image
import sys
import cv2
import numpy as np
import mediapipe as mp
from os import listdir
from os.path import isfile, join
from random import randint

# https://stackoverflow.com/questions/29673348/how-to-open-camera-with-pygame-in-windows
handsDetector = mp.solutions.hands.Hands()
pygame.init()
pygame.camera.init()

cameras = pygame.camera.list_cameras()
webcam = pygame.camera.Camera(cameras[0])
webcam.start()
img = webcam.get_image()

WIDTH = img.get_width()
HEIGHT = img.get_height()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Krivoi Chertila")

last_point = None
current_point = None
start_point = None
contours_to_display = [f for f in listdir("resources/images_contours") if isfile(join("resources/images_contours", f))]


class Sector(pygame.sprite.Sprite):
    def __init__(self, screen, start_point, end_point, hue):
        pygame.sprite.Sprite.__init__(self)
        self.screen = screen
        self.start_point = start_point
        self.end_point = end_point
        self.h = hue

    def update(self):
        h, s, v = self.h, 255, 255
        blank_img = np.ones((1, 1, 3), np.uint8)
        blank_img[:] = (h, s, v)
        r, g, b = cv2.cvtColor(blank_img, cv2.COLOR_HSV2RGB)[0, 0]
        pygame.draw.line(screen, (r, g, b), self.start_point, self.end_point, width=3)


Sectors = pygame.sprite.Group()
drawing = False
has_exited_start = False
frame_exited_start = 0

myfont = pygame.font.SysFont("resources/fonts/Montserrat-Black.ttf", 50)

frame = 0
src = None

contour = np.array([[0, 0]], ndmin=2)

score = None
pixels_cords = None
np_image = None
while True:
    if src is None:
        src = f'resources/images_contours/{contours_to_display[randint(0, len(contours_to_display) - 1)]}'
        np_image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    screen.blit(pygame.transform.flip(img, True, False), (0, 0))
    screen.blit(pygame.image.load(src), (0, 0))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()
        if e.type == pygame.KEYDOWN:
            drawing = True

    if not drawing:
        screen.blit(pygame.image.load("resources/menu_elements/start_label.png"), (0, 0))
        if score is not None:
            h, s, v = 30 - 30 * math.cos(math.pi * score / 100), 255, 255

            blank_img = np.ones((1, 1, 3), np.uint8)
            blank_img[:] = (h, s, v)
            r, g, b = cv2.cvtColor(blank_img, cv2.COLOR_HSV2RGB)[0, 0]
            label = myfont.render(f'{score} %', 1, (r, g, b))
            text_rect = label.get_rect(center=(WIDTH / 2, HEIGHT / 2))
            screen.blit(label, text_rect)

        view = pygame.surfarray.array3d(img).transpose([1, 0, 2])

        img_cv2 = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        flipped = np.fliplr(img_cv2)
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        results = handsDetector.process(flippedRGB)
        if results.multi_hand_landmarks is not None and results.multi_handedness[0].classification[0].score > 0.7:
            finger_tip_cords = results.multi_hand_landmarks[0].landmark[8]
            pygame.draw.circle(screen, (0, 0, 255), (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT), 5)

    else:

        img_np = pygame.surfarray.array3d(img).transpose([1, 0, 2])
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        flipped = np.fliplr(img_cv2)
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        results = handsDetector.process(flippedRGB)
        if results.multi_hand_landmarks is not None and results.multi_handedness[0].classification[0].score > 0.7:
            finger_tip_cords = results.multi_hand_landmarks[0].landmark[8]
            if pixels_cords is None:
                pixels_cords = np.array([[0, 0]], ndmin=2)
                for x in range(WIDTH):
                    for y in range(HEIGHT):
                        if np_image[y, x, 3] > 0:
                            pixels_cords = np.append(pixels_cords, [[x, y]], axis=0)

            if current_point is None:
                last_point = (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT)
                current_point = last_point
                start_point = last_point
            else:
                if math.sqrt((start_point[0] - current_point[0]) ** 2 + (
                        start_point[1] - current_point[1]) ** 2) > 12 and not has_exited_start:
                    frame_exited_start = frame
                    has_exited_start = True
                elif math.sqrt((start_point[0] - current_point[0]) ** 2 + (
                        start_point[1] - current_point[
                    1]) ** 2) < 8 and has_exited_start and frame - frame_exited_start > 30:

                    pixels_cords = pixels_cords[1:]
                    contour = contour[1:]
                    error_from_contour = np.array([0, 0])
                    error_from_path = np.array([0, 0])
                    for point in pixels_cords:
                        error_from_contour = np.append(error_from_contour, [math.e ** (0.05 * np.min(
                            np.sqrt((contour[:, 0] - point[0]) ** 2 + (contour[:, 1] - point[1]) ** 2)))])
                    for countour_point in contour:
                        error_from_path = np.append(error_from_path, [math.e ** (0.1 * np.min(
                            np.sqrt((pixels_cords[:, 0] - countour_point[0]) ** 2 + (
                                    pixels_cords[:, 1] - countour_point[1]) ** 2)))])
                    error_from_contour = error_from_contour[1:]
                    error_from_path = error_from_path[1:]
                    score = round(
                        100 * math.e ** (-0.04 * max(np.average(error_from_contour), np.average(error_from_path))), 1)

                    drawing = False
                    has_exited_start = False
                    Sectors.empty()
                    last_point = None
                    current_point = None
                    start_point = None
                    contour = np.array([[0, 0]], ndmin=2)
                    pixels_cords = None
                    src = None
                    continue

                last_point = current_point
                current_point = (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT)
                contour = np.append(contour, [current_point], axis=0)
                dist = np.min(np.sqrt((pixels_cords[:, 0] - current_point[0]) ** 2 + (
                        pixels_cords[:, 1] - current_point[1]) ** 2))

                Sectors.add(Sector(screen, last_point, current_point, 60 * math.e ** (-0.05 * dist)))

        Sectors.update()
        if start_point is not None:
            pygame.draw.circle(screen, (255, 255, 0), start_point, 8)

    pygame.display.flip()
    pygame.display.update()

    img = webcam.get_image()
    frame += 1
