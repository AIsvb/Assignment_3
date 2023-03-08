import glm
import numpy as np
import cv2
from LookupTable import LookupTable as LT

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

block_size = 1.0

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([57/255,138/255,123/255])
    return data, colors

def set_voxel_positions():
    data, colors = [], []

    for v in voxel_list:
        data.append([v.voxel_coordinates[0] * 0.05 + 10, -v.voxel_coordinates[2] * 0.05, v.voxel_coordinates[1] * 0.05 - 30])
        colors.append([v.color[0], v.color[1], v.color[2]])
    return data, colors

# Function to set voxels based on a XOR-mask
def set_voxel_positions_XOR(frame1, frame2, frame3, frame4, list):
    data, colors = [], []
    new_voxel_list = LT.get_voxels_XOR(frame1, frame2, frame3, frame4, list)
    for v in new_voxel_list:
        data.append([v.voxel_coordinates[0] * 0.05 + 10, -v.voxel_coordinates[2] * 0.05, v.voxel_coordinates[1] * 0.05 - 30])
        colors.append([v.color[0], v.color[1], v.color[2]])

    return data, colors, new_voxel_list

# Method to get the camera positions (translation)
def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    v = []  # List to store the coordinates of all the cameras
    for i in range(4):
        cam = i + 1
        file_name = f"data/cam{cam}/config.xml"

        # Obtaining the translation and rotation data
        reader = cv2.FileStorage(file_name, cv2.FileStorage_READ)

        t_vecs = reader.getNode("translation_vectors").mat()
        r_vecs = reader.getNode("rotation_vectors").mat()
        r_vecs, _ = cv2.Rodrigues(r_vecs)

        t_vecs = -np.matrix(r_vecs).T * np.matrix(t_vecs)

        t_vecs = t_vecs.astype(int)
        translation= [(t_vecs[0,0]-500)/20, -t_vecs[2,0]/20, t_vecs[1,0]/20]

        v.append(translation)
    return v, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

# Method to get the rotation matrices
def get_cam_rotation_matrices():
    cam_rotations = []
    for i in range(4):
        cam = i + 1
        file_name = f"data/cam{cam}/config.xml"

        # Reading the rotation vectors from the config.xml file
        reader = cv2.FileStorage(file_name, cv2.FileStorage_READ)

        r_vecs = reader.getNode("rotation_vectors").mat()

        # Calculating the rotation matrix from the rotation vector
        r_mat = cv2.Rodrigues(r_vecs)

        # Putting the rotation matrix in the upper left corner of a 4 by 4 identity matrix
        I = np.identity(4, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                I[i, j] = r_mat[0][i, j]

        # Correct the rotation (glm treats the rotation matrices differently apparently)
        mat = glm.mat4(I)
        mat = glm.rotate(mat, glm.radians(90), (0, 0, 1))

        cam_rotations.append(mat)
    return cam_rotations
