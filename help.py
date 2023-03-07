import numpy as np
import cv2

cam = 1
path = f"data/cam{cam}/extrinsics.png"
img = cv2.imread(path)

reader = cv2.FileStorage(f"data/cam{cam}/config.xml", cv2.FileStorage_READ)
t_vecs = reader.getNode("translation_vectors").mat()
r_vecs = reader.getNode("rotation_vectors").mat()
mtx = reader.getNode("camera_matrix").mat()
cfs = reader.getNode("distortion_coefficients").mat()

x = 2500
y = 2500
z = 2000

dx = -500
dy = -500

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





