# Computer Vision: Assignment 2
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 04-03-2023

import cv2
import numpy as np


class BackgroundSubtractor:
    def __init__(self, background_path, video_path):
        self.background = background_path
        self.video = video_path
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernel = np.ones((3, 3), np.uint8)

    def subtract_background(self):
        background = cv2.VideoCapture(self.background)
        video = cv2.VideoCapture(self.video)

        # Read background video and train background model
        ret, frame = background.read()
        self.fgbg.apply(frame)

        while True:
            # Read video, subtract background without training the background model
            ret, frame = video.read()
            foreground = self.fgbg.apply(frame, None, 0)
            if frame is None:
                break

            # Postprocessing
            # foreground[np.abs(foreground) < 254] = 0
            foreground_mask = self.draw_contours(foreground)

            # Show video
            cv2.imshow('Frame', frame)
            cv2.imshow('Foreground Mask', foreground_mask)
            # cv2.imwrite(f'data/cam2/frames2/{counter}.png', foreground_mask)
            # counter += 1

            keyboard = cv2.waitKey(1)
            if keyboard == 'q':
                break

        video.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(0)

    # Functionality to find and draw contours on a frame
    def draw_contours(self, image):
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create array containing the index, area and parent of every contour
        contours2 = np.zeros((len(contours), 3))
        for i in range(len(contours)):
            contours2[i][0] = i
            contours2[i][1] = float(cv2.contourArea(contours[i]))
            contours2[i][2] = hierarchy[0][i][3]

        # Sort contours in a large-small order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours2 = contours2[contours2[:, 1].argsort()][::-1]

        # Make copy of the input image to draw the contours on
        image_copy = np.zeros_like(image)

        # Set minimum area for contours to be drawn
        min_area = 1000

        # Draw contours of the subject on the copy image
        for i in range(0, len(contours)):
            if contours2[i][1] > min_area:
                cv2.drawContours(image_copy, [contours[i]], -1, (255, 255, 255), thickness=cv2.FILLED)

        return image_copy
