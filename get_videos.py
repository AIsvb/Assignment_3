import cv2
import numpy as np


def create_mask_video(background, foreground, dest):
    counter = 0
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    background = cv2.VideoCapture(background)
    video = cv2.VideoCapture(foreground)
    #print(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    #return
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

        if (counter + 10) % 10 == 0:
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

def create_XOR_videos():
    foreground_1 = cv2.VideoCapture("data/cam1/foreground_cropped.avi")
    foreground_2 = cv2.VideoCapture("data/cam2/foreground_cropped.avi")
    foreground_3 = cv2.VideoCapture("data/cam3/foreground_cropped.avi")
    foreground_4 = cv2.VideoCapture("data/cam4/foreground_cropped.avi")

    out1 = cv2.VideoWriter("data/cam1/XOR.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
                          (644, 486), isColor=False)
    out2 = cv2.VideoWriter("data/cam2/XOR.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
                          (644, 486), isColor=False)
    out3 = cv2.VideoWriter("data/cam3/XOR.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
                          (644, 486), isColor=False)
    out4 = cv2.VideoWriter("data/cam4/XOR.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
                          (644, 486), isColor=False)

    _, mask_1a = foreground_1.read()
    mask_1a = cv2.cvtColor(mask_1a, cv2.COLOR_BGR2GRAY)

    _, mask_2a = foreground_2.read()
    mask_2a = cv2.cvtColor(mask_2a, cv2.COLOR_BGR2GRAY)

    _, mask_3a = foreground_3.read()
    mask_3a = cv2.cvtColor(mask_3a, cv2.COLOR_BGR2GRAY)

    _, mask_4a = foreground_4.read()
    mask_4a = cv2.cvtColor(mask_4a, cv2.COLOR_BGR2GRAY)

    while True:
        _, mask_1b = foreground_1.read()
        if mask_1b is None:
            break

        mask_1b = cv2.cvtColor(mask_1b, cv2.COLOR_BGR2GRAY)

        _, mask_2b = foreground_2.read()
        mask_2b = cv2.cvtColor(mask_2b, cv2.COLOR_BGR2GRAY)

        _, mask_3b = foreground_3.read()
        mask_3b = cv2.cvtColor(mask_3b, cv2.COLOR_BGR2GRAY)

        _, mask_4b = foreground_4.read()
        mask_4b = cv2.cvtColor(mask_4b, cv2.COLOR_BGR2GRAY)


        XOR_1 = mask_1b ^ mask_1a
        XOR_2 = mask_2b ^ mask_2a
        XOR_3 = mask_3b ^ mask_3a
        XOR_4 = mask_4b ^ mask_4a

        out1.write(XOR_1)
        out2.write(XOR_2)
        out3.write(XOR_3)
        out4.write(XOR_4)

        mask_1a = mask_1b
        mask_2a = mask_2b
        mask_3a = mask_3b
        mask_4a = mask_4b

    foreground_1.release()
    foreground_2.release()
    foreground_3.release()
    foreground_4.release()

    out1.release()
    out2.release()
    out3.release()
    out4.release()

#create_XOR_videos()

cam = 4
bg = f"data/cam{cam}/background.avi"
fg = f"data/cam{cam}/video.avi"
dest = f"data/cam{cam}/foreground_cropped.avi"
create_mask_video(bg, fg, dest)