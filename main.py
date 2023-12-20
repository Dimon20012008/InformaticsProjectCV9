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


# class Sector - a pygame.Sprite.sprite object with starting and ending point, color, screen; draws a line when
# .update is used.
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


# source — https://stackoverflow.com/questions/29673348/how-to-open-camera-with-pygame-in-windows
# initialization of camera
handsDetector = mp.solutions.hands.Hands()
pygame.init()
pygame.camera.init()
cameras = pygame.camera.list_cameras()
webcam = pygame.camera.Camera(cameras[0])
webcam.start()
webcam_img = webcam.get_image()

# initialization of window
WIDTH = webcam_img.get_width()
HEIGHT = webcam_img.get_height()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Krivoi Chertila")
myfont = pygame.font.SysFont("resources/fonts/Montserrat-Black.ttf", 50)
frame = 0

# initialization of various parameters

# parameters for drawing the path
last_point = None
current_point = None
start_point = None
Sectors = pygame.sprite.Group()
path_cords = np.array([[0, 0]], ndmin=2)
frame_exited_start = 0
has_exited_start = False
drawing = False
score = None

# parameters for contour
contours_to_display = [f for f in listdir("resources/images_contours") if isfile(join("resources/images_contours", f))]
contour_np_image = None
contour_src = None
contour_cords = None

while True:
    # if src is None, then a random one is picked
    if contour_src is None:
        contour_src = f'resources/images_contours/{contours_to_display[randint(0, len(contours_to_display) - 1)]}'
        contour_np_image = cv2.imread(contour_src, cv2.IMREAD_UNCHANGED)

    # drawing of webcam image and contour
    screen.blit(pygame.transform.flip(webcam_img, True, False), (0, 0))
    screen.blit(pygame.image.load(contour_src), (0, 0))

    # check for being closed or start of the game
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()
        if e.type == pygame.KEYDOWN:
            drawing = True

    # this is a global difference - whether right now user is drawing or not
    if not drawing:
        # if not, then we draw contour, start label, score (if exists) and circle around fingertip
        # drawing of start label
        screen.blit(pygame.image.load("resources/menu_elements/start_label.png"), (0, 0))
        # drawing of score
        if score is not None:
            # formula is used to calculate correct hue (see process journal)
            h, s, v = 30 - 30 * math.cos(math.pi * score / 100), 255, 255
            blank_img_for_conversion = np.ones((1, 1, 3), np.uint8)
            blank_img_for_conversion[:] = (h, s, v)
            r, g, b = cv2.cvtColor(blank_img_for_conversion, cv2.COLOR_HSV2RGB)[0, 0]
            score_label = myfont.render(f'{score} %', 1, (r, g, b))
            score_text_rect = score_label.get_rect(center=(WIDTH / 2, HEIGHT / 2))
            screen.blit(score_label, score_text_rect)

        # detecting and drawing circle around fingertip
        np_webcam_img = pygame.surfarray.array3d(webcam_img).transpose([1, 0, 2])
        cv2_webcam_img = cv2.cvtColor(np_webcam_img, cv2.COLOR_RGB2BGR)
        flipped_webcam_img = np.fliplr(cv2_webcam_img)
        flipped_webcam_imgRGB = cv2.cvtColor(flipped_webcam_img, cv2.COLOR_BGR2RGB)
        hands_detector_results = handsDetector.process(flipped_webcam_imgRGB)
        # if detection failed, or it is not certain, nothing is drawn
        if hands_detector_results.multi_hand_landmarks is not None and \
                hands_detector_results.multi_handedness[0].classification[0].score > 0.7:
            finger_tip_cords = hands_detector_results.multi_hand_landmarks[0].landmark[8]
            pygame.draw.circle(screen, (0, 0, 255), (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT), 5)

    else:
        # if user is drawing, then we need to draw the path and do calculation when path is closed
        # detecting fingertip
        np_webcam_img = pygame.surfarray.array3d(webcam_img).transpose([1, 0, 2])
        cv2_webcam_img = cv2.cvtColor(np_webcam_img, cv2.COLOR_RGB2BGR)
        flipped_webcam_img = np.fliplr(cv2_webcam_img)
        flipped_webcam_imgRGB = cv2.cvtColor(flipped_webcam_img, cv2.COLOR_BGR2RGB)
        hands_detector_results = handsDetector.process(flipped_webcam_imgRGB)
        if hands_detector_results.multi_hand_landmarks is not None and \
                hands_detector_results.multi_handedness[0].classification[0].score > 0.7:
            finger_tip_cords = hands_detector_results.multi_hand_landmarks[0].landmark[8]

            # translating contour into array of pixels
            if contour_cords is None:
                contour_cords = np.array([[0, 0]], ndmin=2)
                for x in range(WIDTH):
                    for y in range(HEIGHT):
                        if contour_np_image[y, x, 3] > 0:
                            contour_cords = np.append(contour_cords, [[x, y]], axis=0)

            # if current_point is None, then it is the first frame of drawing
            if current_point is None:
                last_point = (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT)
                current_point = last_point
                start_point = last_point
            # else there must be done some checks
            else:
                # detection of when user exits start (gets 12 pixels farther away)
                if math.sqrt((start_point[0] - current_point[0]) ** 2 + (
                        start_point[1] - current_point[1]) ** 2) > 12 and not has_exited_start:
                    frame_exited_start = frame
                    has_exited_start = True
                # detection of when user returns (gets within 8 pixels from start and already exited and 0.5 sec passed)
                elif math.sqrt((start_point[0] - current_point[0]) ** 2 + (
                        start_point[1] - current_point[
                    1]) ** 2) < 8 and has_exited_start and frame - frame_exited_start > 30:
                    # we need to crop the array, because in init there is an element added for correct structure
                    contour_cords = contour_cords[1:]
                    path_cords = path_cords[1:]

                    # calculation of average distance from contour to path and from path to contour
                    dists_from_contour = np.array([0, 0])
                    dists_from_path = np.array([0, 0])
                    for point in contour_cords:
                        dists_from_contour = np.append(dists_from_contour, [math.e ** (0.05 * np.min(
                            np.sqrt((path_cords[:, 0] - point[0]) ** 2 + (path_cords[:, 1] - point[1]) ** 2)))])
                    for contour_point in path_cords:
                        dists_from_path = np.append(dists_from_path, [math.e ** (0.1 * np.min(
                            np.sqrt((contour_cords[:, 0] - contour_point[0]) ** 2 + (
                                    contour_cords[:, 1] - contour_point[1]) ** 2)))])
                    dists_from_contour = dists_from_contour[1:]
                    dists_from_path = dists_from_path[1:]
                    # score is calculated to represent how accurate user is, it is not some exact representation of path
                    score = round(
                        100 * math.e ** (-0.04 * max(np.average(dists_from_contour), np.average(dists_from_path))), 1)

                    # reinitialization of params
                    drawing = False
                    has_exited_start = False
                    Sectors.empty()
                    last_point = None
                    current_point = None
                    start_point = None
                    path_cords = np.array([[0, 0]], ndmin=2)
                    contour_cords = None
                    contour_src = None
                    continue

                # otherwise, we just need to draw the path.
                last_point = current_point
                current_point = (finger_tip_cords.x * WIDTH, finger_tip_cords.y * HEIGHT)
                path_cords = np.append(path_cords, [current_point], axis=0)
                current_dist_to_contour = np.min(np.sqrt((contour_cords[:, 0] - current_point[0]) ** 2 + (
                        contour_cords[:, 1] - current_point[1]) ** 2))
                Sectors.add(Sector(screen, last_point, current_point, 60 * math.e ** (-0.05 * current_dist_to_contour)))

        Sectors.update()
        if start_point is not None:
            pygame.draw.circle(screen, (255, 255, 0), start_point, 8)

    pygame.display.flip()
    pygame.display.update()

    webcam_img = webcam.get_image()
    frame += 1
