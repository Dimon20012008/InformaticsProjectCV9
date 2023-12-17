import math

import pygame.camera
import pygame.image
import sys
import cv2
import numpy as np
import mediapipe as mp

# https://stackoverflow.com/questions/29673348/how-to-open-camera-with-pygame-in-windows
sys.stdout.flush()

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
start_point = None


class Sector(pygame.sprite.Sprite):
    def __init__(self, screen, start_point, end_point):
        pygame.sprite.Sprite.__init__(self)
        self.screen = screen
        self.start_point = start_point
        self.end_point = end_point

    def update(self):
        pygame.draw.line(screen, (0, 255, 0), self.start_point, self.end_point, width=1)


Sectors = pygame.sprite.Group()
drawing = False
has_exited_start = False
frame_exited_start = 0
pygame.font.init()
myfont = pygame.font.SysFont("resources/fonts/Montserrat-Black.ttf", 50)

frame = 0
src = "resources/images_contours/circle.png"

contour = np.array([[0, 0]], ndmin=2)

score = None
while True:
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
            font_color_param = int(255 * math.e ** (-0.05 * score))
            label = myfont.render(f'{score} %', 1, (font_color_param, 255 - font_color_param, 0))
            text_rect = label.get_rect(center=(WIDTH / 2, HEIGHT / 2))
            screen.blit(label, text_rect)
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
                elif math.sqrt((start_point[0] - current_point[0]) ** 2 + (
                        start_point[1] - current_point[
                    1]) ** 2) < 5 and has_exited_start and frame - frame_exited_start > 30:

                    np_image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
                    pixels_cords = np.array([[0, 0]], ndmin=2)
                    for x in range(WIDTH):
                        for y in range(HEIGHT):
                            if np_image[y, x, 3] > 0:
                                pixels_cords = np.append(pixels_cords, [[x, y]], axis=0)
                    pixels_cords = pixels_cords[1:]
                    contour = contour[1:]
                    error = np.array([0, 0])

                    for point in pixels_cords:
                        error = np.append(error, [np.min(
                            np.sqrt((contour[:, 0] - point[0]) ** 2 + (contour[:, 1] - point[1]) ** 2))])

                    for countour_point in contour:
                        error = np.append(error, [np.min(
                            np.sqrt((pixels_cords[:, 0] - countour_point[0]) ** 2 + (
                                    pixels_cords[:, 1] - countour_point[1]) ** 2))])

                    error = error[1:]
                    score = round(100 * math.e ** (-0.015 * np.average(error)), 1)

                    drawing = False
                    has_exited_start = False
                    Sectors.empty()
                    last_point = None
                    current_point = None
                    start_point = None
                    contour = np.array([[0, 0]], ndmin=2)
                    continue

                last_point = current_point
                current_point = (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT)
                contour = np.append(contour, [current_point], axis=0)
                Sectors.add(Sector(screen, last_point, current_point))

        Sectors.update()
        if start_point is not None:
            pygame.draw.circle(screen, (0, 0, 255), start_point, 2)

    pygame.display.flip()
    pygame.display.update()

    img = webcam.get_image()
    frame += 1
