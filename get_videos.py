# Computer Vision: Assignment 3
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 18-03-2023

import cv2
import numpy as np

"""
The functions in this file were used to crop the videos and obtain the foreground mask videos
"""


def crop_video(path, dest):
    counter = 0
    video = cv2.VideoCapture(path)
    out = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (644, 486), isColor=True)

    while True:
        ret, frame = video.read()
        counter += 1

        if frame is None:
            break

        if (counter + 10) % 10 == 0: # The video was cropped to roughly 5 fps
            out.write(frame)

    video.release()
    out.release()

# Function for creating the foreground mask videos
def create_mask_video(background, foreground, dest):
    counter = 0
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    background = cv2.VideoCapture(background)
    video = cv2.VideoCapture(foreground)
    out = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (644, 486), isColor=False)

    while True:
        ret, frame = background.read()

        if frame is None:
            break

        fgbg.apply(frame)

    background.release()

    while True:
        ret, frame = video.read()
        counter += 1

        if frame is None:
            break

        if (counter + 10) % 10 == 0: # The video was cropped to roughly 5 fps
            foreground = fgbg.apply(frame, None, 0)
            foreground_mask = draw_contours(foreground)
            kernel = np.ones((3,3), np.uint8)
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
            #cv2.imshow("img", foreground_mask)
            out.write(foreground_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    out.release()
    #cv2.destroyAllWindows()

def draw_contours(image):
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

def main():
    for cam in np.arange(0, 4):
        bg = f"data/cam{cam}/background.avi"
        fg = f"data/cam{cam}/video.avi"
        dest_fg = f"data/cam{cam}/foreground_cropped.avi"
        create_mask_video(bg, fg, dest_fg)

        path = f"data/cam{cam}/video.avi"
        dest_video = f"data/cam{cam}/video_cropped.avi"

        crop_video(path, dest_video)

# Executing the code
if __name__ == "__main__":
    main()
