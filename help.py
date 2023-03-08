import numpy as np
import cv2

cam = 2
#path = f"data/cam{cam}/extrinsics.png"
path = "C:/Users/svben/Downloads/MicrosoftTeams-image (1).png"
img = cv2.imread(path)

reader = cv2.FileStorage(f"data/cam{cam}/config.xml", cv2.FileStorage_READ)
t_vecs = reader.getNode("translation_vectors").mat()
r_vecs = reader.getNode("rotation_vectors").mat()
mtx = reader.getNode("camera_matrix").mat()
cfs = reader.getNode("distortion_coefficients").mat()

x = 3400
y = 4700
z = 2000

dx = -1900
dy = -700

cube = np.float32([[dx, dy, 0], [dx, y + dy, 0], [x + dx, y + dy, 0], [x + dx, dy, 0],
                   [dx, dy, -z], [dx, y + dy, -z], [x + dx, y + dy, -z], [x + dx, dy, -z]])

imgpts_c, _ = cv2.projectPoints(cube, r_vecs, t_vecs, mtx, cfs)


imgpts = np.int32(imgpts_c).reshape(-1, 2)

# draw the bottom square
img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 255, 0), 2)

# draw pillars
for i, j in zip(range(4), range(4, 8)):
    img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 2)

# draw top square
img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
import cv2
import numpy as np
def create_mask_video(background, foreground, dest):
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

        if frame is None:
            break

        foreground = fgbg.apply(frame, None, 0)
        foreground_mask = draw_contours(foreground)
        cv2.imshow("img", foreground_mask)
        out.write(foreground_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

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

cam = 4
bg = f"data/cam{cam}/background.avi"
fg = f"data/cam{cam}/video.avi"
dest = f"data/cam{cam}/foreground.avi"
create_mask_video(bg, fg, dest)
"""

