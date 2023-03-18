# Computer Vision: Assignment 3
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 18-03-2023


import numpy as np
import cv2

"""
Explanation: This following script was used to visualise the voxel space as a cube drawn on the color images
"""

def main(cam):
    # Loading the image onto which the cube must be drawn
    path = f"data/cam{cam}/color.png"
    img = cv2.imread(path)

    # Loading the calibration data
    reader = cv2.FileStorage(f"data/cam{cam}/config.xml", cv2.FileStorage_READ)
    t_vecs = reader.getNode("translation_vectors").mat()
    r_vecs = reader.getNode("rotation_vectors").mat()
    mtx = reader.getNode("camera_matrix").mat()
    cfs = reader.getNode("distortion_coefficients").mat()

    # The dimensions of the cube in mm
    x = 3400
    y = 4700
    z = 2000

    # The offset of the cubes origin
    dx = -1900
    dy = -700

    cube = np.float32([[dx, dy, 0], [dx, y + dy, 0], [x + dx, y + dy, 0], [x + dx, dy, 0],
                       [dx, dy, -z], [dx, y + dy, -z], [x + dx, y + dy, -z],
                       [x + dx, dy, -z]])  # Coordinates of the cubes corners

    imgpts_c, _ = cv2.projectPoints(cube, r_vecs, t_vecs, mtx,
                                    cfs)  # Image points that correspond with the cubes corners
    imgpts = np.int32(imgpts_c).reshape(-1, 2)

    # draw the bottom square
    img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 255, 0), 2)

    # draw pillars
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 2)

    # draw top square
    img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 255, 0), 2)

    #  Showing the result
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Execution of the code
if __name__ == "__main__":
    for cam in np.arange(1, 5):
        main(cam)
