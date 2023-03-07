import numpy as np
import cv2

def calculate_mean(path):
    video = cv2.VideoCapture(path)
    length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, frame_rgb = video.read()
    frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    shape = frame_hsv.shape
    sum = np.zeros(shape, dtype=int) + frame_hsv

    while True:
        ret, frame = video.read()

        if frame is None:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sum += frame

    video.release()
    return sum / length, shape

def calculate_sd(path , mean, shape):
    video = cv2.VideoCapture(path)
    length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    SSE = np.zeros(shape, dtype=np.float64)

    while True:
        ret, frame = video.read()

        if frame is None:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            SSE += (frame - mean)**2

    video.release()
    return np.sqrt(SSE / length)

def get_foreground(path, mean, sd, shape):
    video = cv2.VideoCapture(path)

    while True:
        foreground = np.ones(shape[0:2], dtype=np.uint8) * 255
        ret, frame = video.read()

        if frame is None:
            break

        i = 4
        j = 8
        l = 12
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        c1 = frame[:, :, 0] > mean[:, :, 0] - i*sd[:, :, 0]
        c2 = frame[:, :, 1] > mean[:, :, 1] - j*sd[:, :, 1]
        c3 = frame[:, :, 2] > mean[:, :, 2] - l*sd[:, :, 2]
        lower = np.logical_and(c1, c2, c3)
        lower = c1

        c4 = frame[:, :, 0] < mean[:, :, 0] + i*sd[:, :, 0]
        c5 = frame[:, :, 1] < mean[:, :, 1] + j*sd[:, :, 1]
        c6 = frame[:, :, 2] < mean[:, :, 2] + l*sd[:, :, 2]
        upper = np.logical_and(c4, c5, c6)

        foreground[np.logical_and(lower, upper)] = 0
        foreground = contours(foreground)
        lower = c4

        cv2.imshow("foreground", foreground)
        cv2.waitKey(1)

    video.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_difference(path):
    video = cv2.VideoCapture(path)
    ret, frame_1 = video.read()
    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV)

    while True:
        fg = np.ones(frame_1.shape[0:2], dtype=np.uint8)*255
        ret, frame_2 = video.read()

        if frame_2 is None:
            break

        frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2HSV)
        d = 5
        c1 = frame_2[:, :, 0] - frame_1[:, :, 0] < d
        c2 = frame_2[:, :, 1] - frame_1[:, :, 1] < d
        c3 = frame_2[:, :, 2] - frame_1[:, :, 2] < d
        frame_1 = frame_2
        fg[np.logical_and(c1, c2, c3)] = 0
        cv2.imshow("fg", fg)
        cv2.waitKey(1)

    video.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contours(image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Make copy of the input image to draw the contours on
    image_copy = np.zeros_like(image)

    # Draw contours of the subject on the copy image
    cv2.drawContours(image_copy, contours[0:4], -1, (255, 255, 255), thickness=cv2.FILLED)

    return image_copy







path = "data/cam3/background.avi"
video = "data/cam4/video.avi"

#mean, shape = calculate_mean(path)
#sd = calculate_sd(path, mean, shape)
#fg = get_foreground(video, mean, sd, shape)

calculate_difference(video)
