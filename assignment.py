import glm
import numpy as np
import cv2
from LookUp import LookupTable as LT
import time

foreground1 = cv2.VideoCapture("data/cam1/foreground_cropped.avi")
foreground2 = cv2.VideoCapture("data/cam2/foreground_cropped.avi")
foreground3 = cv2.VideoCapture("data/cam3/foreground_cropped.avi")
foreground4 = cv2.VideoCapture("data/cam4/foreground_cropped.avi")


voxel_size = 50
table = LT(68, 94, 40, voxel_size)

# voxel_size = 25
# table = LT(136, 188, 80, voxel_size)

frame_no = 0
block_size = 1

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
    _, mask_1a = foreground1.read()
    mask_1a = cv2.cvtColor(mask_1a, cv2.COLOR_BGR2GRAY)

    _, mask_2a = foreground2.read()
    mask_2a = cv2.cvtColor(mask_2a, cv2.COLOR_BGR2GRAY)

    _, mask_3a = foreground3.read()
    mask_3a = cv2.cvtColor(mask_3a, cv2.COLOR_BGR2GRAY)

    _, mask_4a = foreground4.read()
    mask_4a = cv2.cvtColor(mask_4a, cv2.COLOR_BGR2GRAY)

    data, colors = table.get_voxels([mask_1a, mask_2a, mask_3a, mask_4a])
    
    return data, colors

def set_voxel_positions2():
    global mask_1a, mask_2a, mask_3a, mask_4a

    data, colors = table.get_voxels([mask_1a, mask_2a, mask_3a, mask_4a])

    clusters = cluster(data)

    data2 = []
    colors2 = []

    for i in range(4):
        for j in range(len(clusters[i])):
            data2.append([clusters[i][j][0], clusters[i][j][1], clusters[i][j][2]])
            if i == 0:
                colors2.append([1, 0, 0])
            if i == 1:
                colors2.append([0, 1, 0])
            if i == 2:
                colors2.append([0, 0, 1])
            if i == 3:
                colors2.append([1, 1, 1])


    return data2, colors2

# Function to set voxels based on a XOR-mask
def set_voxel_positions_XOR():
    global mask_1a, mask_2a, mask_3a, mask_4a, voxel_space

    _, mask_1b = foreground1.read()
    mask_1b = cv2.cvtColor(mask_1b, cv2.COLOR_BGR2GRAY)

    _, mask_2b = foreground2.read()
    mask_2b = cv2.cvtColor(mask_2b, cv2.COLOR_BGR2GRAY)

    _, mask_3b = foreground3.read()
    mask_3b = cv2.cvtColor(mask_3b, cv2.COLOR_BGR2GRAY)

    _, mask_4b = foreground4.read()
    mask_4b = cv2.cvtColor(mask_4b, cv2.COLOR_BGR2GRAY)

    XOR_1 = np.bitwise_xor(mask_1b, mask_1a)
    XOR_2 = np.bitwise_xor(mask_2b, mask_2a)
    XOR_3 = np.bitwise_xor(mask_3b, mask_3a)
    XOR_4 = np.bitwise_xor(mask_4b, mask_4a)

    mask_1a = np.copy(mask_1b)
    mask_2a = np.copy(mask_2b)
    mask_3a = np.copy(mask_3b)
    mask_4a = np.copy(mask_4b)

    data, colors = table.get_voxels_XOR([XOR_1, XOR_2, XOR_3, XOR_4], voxel_space)
    #data, colors = table.get_voxels([mask_1b, mask_2b, mask_3b, mask_4b])
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

def cluster(voxels):
    start = time.time()
    coords = np.empty((len(voxels), 3), dtype=np.float32)
    for i, voxel in enumerate(voxels):
        x, y, z = voxel[0], voxel[1], voxel[2]
        coords[i] = np.array([x, y, z], dtype=np.float32)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(coords, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labeled_voxels = np.hstack((coords, label)).astype(int)

    clusters = [labeled_voxels[labeled_voxels[:, 3] == 0, :], labeled_voxels[labeled_voxels[:, 3] == 1, :],
                labeled_voxels[labeled_voxels[:, 3] == 2, :], labeled_voxels[labeled_voxels[:, 3] == 3, :]]

    end = time.time()
    print(f"Execution time clustering: {(end - start)} seconds")
    return clusters
