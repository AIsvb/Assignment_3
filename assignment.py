import glm
import numpy as np
import cv2
from LookUp import LookupTable as LT

'''
# Masks used for initialization (frame 0 of each view)
mask1 = cv2.imread('data/cam1/voxel.png')
mask2 = cv2.imread('data/cam2/voxel.png')
mask3 = cv2.imread('data/cam3/voxel.png')
mask4 = cv2.imread('data/cam4/voxel.png')

# Create a look-up table
LT = LT(150, 75, 100)
LT.create_voxels()
LT.create_lookup_table()

# List of first voxels that need to be shown
voxel_list = LT.get_voxels(mask1, mask2, mask3, mask4)
'''


foreground_1 = cv2.VideoCapture("data/cam1/XOR.avi")
foreground_2 = cv2.VideoCapture("data/cam2/XOR.avi")
foreground_3 = cv2.VideoCapture("data/cam3/XOR.avi")
foreground_4 = cv2.VideoCapture("data/cam4/XOR.avi")

_, mask_1a = foreground_1.read()
mask_1a = cv2.cvtColor(mask_1a, cv2.COLOR_BGR2GRAY)

_, mask_2a = foreground_2.read()
mask_2a = cv2.cvtColor(mask_2a, cv2.COLOR_BGR2GRAY)

_, mask_3a = foreground_3.read()
mask_3a = cv2.cvtColor(mask_3a, cv2.COLOR_BGR2GRAY)

_ ,mask_4a = foreground_4.read()
mask_4a = cv2.cvtColor(mask_4a, cv2.COLOR_BGR2GRAY)

table = LT(170, 75, 100)

frame_no = 0
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
    global mask_1a, mask_2a, mask_3a, mask_4a

    data, colors = table.get_voxels([mask_1a, mask_2a, mask_3a, mask_4a])
    
    return data, colors

# Function to set voxels based on a XOR-mask
def set_voxel_positions_XOR():
    global mask_1a, mask_2a, mask_3a, mask_4a

    _, mask_1b = foreground_1.read()
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

    mask_1a = mask_1b
    mask_2a = mask_2b
    mask_3a = mask_3b
    mask_4a = mask_4b

    data, colors = table.get_voxels_XOR([XOR_1, XOR_2, XOR_3, XOR_4])

    return data, colors

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
