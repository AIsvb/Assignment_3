import numpy as np
import cv2
from LookupTable import LookupTable as LT
from LookupTable import Voxel

# Masks used for initialization (frame 0 of each view)
mask1 = cv2.imread('data/cam1/voxel.png')
mask2 = cv2.imread('data/cam2/voxel.png')
mask3 = cv2.imread('data/cam3/voxel.png')
mask4 = cv2.imread('data/cam4/voxel.png')

# Create a look-up table
LT = LT(150, 75, 100, mask1, mask2, mask3, mask4)
LT.create_voxels()
LT.create_lookup_table()

# List of first voxels that need to be shown
voxel_list = LT.get_voxels(mask1, mask2, mask3, mask4)

coords = np.empty((len(voxel_list), 3), dtype=np.float32)
for i, voxel in enumerate(voxel_list):
    x, y, z = voxel.voxel_coordinates
    coords[i] = np.array([x, y, z], dtype=np.float32)

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(coords, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)


voxel_data = np.append(coords, label, axis=1)

camera = 2
reader = cv2.FileStorage(f"data/cam{camera}/config.xml", cv2.FileStorage_READ)
t_vecs = reader.getNode("translation_vectors").mat()
r_vecs = reader.getNode("rotation_vectors").mat()
camera_matrix = reader.getNode("camera_matrix").mat()
distortion_coef = reader.getNode("distortion_coefficients").mat()

img = cv2.imread("data/cam2/color.png")
for c in center:
    pts = np.float32(c)
    image_points, _ = cv2.projectPoints(pts, r_vecs, t_vecs, camera_matrix, distortion_coef)
    cp = tuple(image_points[0].ravel().astype(int))
    print(cp)
    img = cv2.circle(img, cp, 3, (0, 0, 255), -1)


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()