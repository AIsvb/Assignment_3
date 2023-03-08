import cv2
from LookupTable import LookupTable as LT
import numpy as np

def live_clustering(cam1, cam2, cam3, cam4, video):
    counter = 0
    cam1 = cv2.VideoCapture(cam1)
    cam2 = cv2.VideoCapture(cam2)
    cam3 = cv2.VideoCapture(cam3)
    cam4 = cv2.VideoCapture(cam4)
    video = cv2.VideoCapture(video)

    camera = 2
    reader = cv2.FileStorage(f"data/cam{camera}/config.xml", cv2.FileStorage_READ)
    t_vecs = reader.getNode("translation_vectors").mat()
    r_vecs = reader.getNode("rotation_vectors").mat()
    camera_matrix = reader.getNode("camera_matrix").mat()
    distortion_coef = reader.getNode("distortion_coefficients").mat()

    _, frame1 = cam1.read()
    _, frame2 = cam2.read()
    _, frame3 = cam3.read()
    _, frame4 = cam4.read()
    _, img = video.read()

    table = LT(150, 75, 100, frame1, frame2, frame3, frame4)
    table.create_voxels()
    table.create_lookup_table()

    voxel_list = table.get_voxels(frame1, frame2, frame3, frame4)

    labels, centers = find_clusters(voxel_list)
    result = show_cluster_centers(centers, img, r_vecs, t_vecs, camera_matrix, distortion_coef)
    cv2.imshow("img", result)
    cv2.waitKey(1)

    while True:
        counter += 1
        _, frame1a = cam1.read()
        _, frame2a = cam2.read()
        _, frame3a = cam3.read()
        _, frame4a = cam4.read()
        _, img = video.read()

        if frame1a is None or frame2a is None or frame3a is None or frame4a is None or img is None:
            break

        if (counter + 100) % 100 == 0:
            XOR_1 = frame1a ^ frame1
            XOR_2 = frame2a ^ frame2
            XOR_3 = frame3a ^ frame3
            XOR_4 = frame4a ^ frame4

            print("start_1")
            new_voxel_list = table.get_voxels_XOR(XOR_1, XOR_2, XOR_3, XOR_4, voxel_list)
            print("start_2")
            labels, centers = find_clusters(new_voxel_list)
            print("end")
            result = show_cluster_centers(centers, img, r_vecs, t_vecs, camera_matrix, distortion_coef)
            cv2.imshow("img", result)
            cv2.waitKey(1)

        frame1 = frame1a
        frame2 = frame2a
        frame3 = frame3a
        frame4 = frame4a

    cam1.release()
    cam2.release()
    cam3.release()
    cam4.release()
    video.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_clusters(voxel_list):
    coords = np.empty((len(voxel_list), 3), dtype=np.float32)

    for i, voxel in enumerate(voxel_list):
        x, y, z = voxel.voxel_coordinates
        coords[i] = np.array([x, y, z], dtype=np.float32)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(coords, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    return label, center

def show_cluster_centers(centers, img, r_vecs, t_vecs, camera_matrix, distortion_coef):
    for c in centers:
        pts = np.float32(c)
        image_points, _ = cv2.projectPoints(pts, r_vecs, t_vecs, camera_matrix, distortion_coef)
        cp = tuple(image_points[0].ravel().astype(int))
        img = cv2.circle(img, cp, 3, (0, 0, 255), -1)
    return img

cam1 = "data/cam1/foreground.avi"
cam2 = "data/cam2/foreground.avi"
cam3 = "data/cam3/foreground.avi"
cam4 = "data/cam4/foreground.avi"
video = "data/cam2/video.avi"

live_clustering(cam1, cam2, cam3, cam4, video)