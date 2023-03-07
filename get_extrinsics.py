# Computer Vision: Assignment 2
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 04-03-2023

import cv2
from CameraCalibrator import CameraCalibrator
import glob

"""
This script is meant to find the extrinsic parameters of each of the four cameras.
"""
if __name__ == "__main__":

    # Creating a CameraCalibrator object for the camera for which we want to find the extrinsics
    cam = 1
    path = f"data/cam{cam}/extrinsics.png"
    calibrator = CameraCalibrator((8, 6), 115, path)

    reader = cv2.FileStorage(f"data/cam{cam}/config.xml", cv2.FileStorage_READ)
    camera_matrix = reader.getNode("camera_matrix").mat()
    distortion_coefficients = reader.getNode("distortion_coefficients").mat()

    # Finding the extrinsic parameters
    rotation_vectors, translation_vectors, corners = calibrator.get_extrinsic(camera_matrix,
                                                                              distortion_coefficients)

    # Writing the data to XML file
    file_name = f"data/cam{cam}/config.xml"
    writer = cv2.FileStorage(file_name, cv2.FileStorage_WRITE)

    writer.write("camera_matrix", camera_matrix)
    writer.write("distortion_coefficients", distortion_coefficients)
    writer.write("rotation_vectors", rotation_vectors)
    writer.write("translation_vectors", translation_vectors)

    writer.release()

    # Printing all the parameters

    #print(rotation_vectors)
    #print(translation_vectors)
    #print(camera_matrix)
    #print(distortion_coefficients)


    # Visualising the corners found and used to calibrate the camera
    img = cv2.imread(path)
    calibrator.draw_chessboard_corners(img, corners)

    # Visualising the orientation of the chessboard relative to the camera
    img = cv2.imread(path)
    destination = f"data/cam{cam}/calibration/result.png"
    calibrator.draw_pose(img, rotation_vectors, translation_vectors, camera_matrix,
                         distortion_coefficients, destination, cam)