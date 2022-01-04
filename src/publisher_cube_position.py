#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Int32, Bool
from geometry_msgs.msg import Pose

from cube_tracking.TrajectoryTracker import TrajectoryTracker
from cube_tracking.Camera import Camera

import os
import json
import numpy as np
from numpy.linalg import inv
import cv2
import math
import scipy
from scipy.spatial.transform import Rotation as R




def readConfig(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['dict_to_use'], config['visualize'], config['grey_color'], config['id']



def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])




def averageQuaternions(Q):

    # Q is a Nx4 numpy matrix and contains the quaternions to average in the rows. The quaternions are arranged in the form (w, x, y, z), with w being the scalar value.
    # The result will be the average quaternion of the input. Note that the signs of the output quaternion can be reversed, since q and -q describe the same orientation

    M = Q.shape[0]                 # number of quaternions to average
    A = np.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        A = np.outer(q,q) + A      # multiply q with its transpose q' and add matrix A

    A = (1.0/M)*A

    eigenValues, eigenVectors = np.linalg.eig(A)
 
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]      # sort by largest eigenvalue
    
    return np.real(eigenVectors[:,0].A1)                            # return the real part of the largest eigenvector (has only real part)



class ArUcoDetector:

    ARUCO_DICT = {
	    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    def __init__(self, dict_to_use):
        self.dict_to_use = dict_to_use
        self.arucoDict = cv2.aruco.Dictionary_get(ArUcoDetector.ARUCO_DICT[dict_to_use])
        self.arucoParams = cv2.aruco.DetectorParameters_create()


    def detect(self, image, matrix_coefficients, distortion_coefficients):
        result = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)
        return result

    
    def get_transf_matrix(self, position, rot_matrix):
    
        t = np.reshape(position, (3,1))
        T = np.concatenate((rot_matrix, t), axis=1)
        T = np.concatenate((T, np.reshape(np.array([0, 0, 0, 1]), (1,4))), axis=0)

        return T


    def pose_estimation(self, frame, result, matrix_coefficients, distortion_coefficients, size_marker, size_cube, num_cubes, positions_list, T_camera_robot):
        corners = result[0]
        corners = np.array(corners)
        ids = result[1]

        
        id_objects = np.array([False, False])

        rejected_img_points = result[2]
        ids = ids.reshape(-1)


        #######  OBJECT 1  #######
        object_1_faces = np.sum(np.where(ids<=6,1,0))                  # Number of faces detected of object 1
        corners_object_1 = corners[np.argwhere(ids<=6)]                # Corners of each detected face of object 1
        object_1_id = ids[np.argwhere(ids<=6)]                         # ID of each detected face in object 1


        #######  OBJECT 2  #######
        cond_obj2 = ((ids>6) & (ids<=12))                          
        object_2_faces = np.sum(np.where(cond_obj2, 1, 0))             # Number of faces detected of object 2      
        corners_object_2 = corners[np.argwhere(cond_obj2)]             # Corners of each detected face of object 2
        object_2_id = ids[np.argwhere(cond_obj2)]                      # ID of each detected face in object 2


        #######  OBJECT 3  #######
        cond_obj3 = ((ids>12) & (ids<=18))                        
        object_3_faces = np.sum(np.where(cond_obj3, 1, 0))             # Number of faces detected of object 3
        corners_object_3 = corners[np.argwhere(cond_obj3)]             # Corners of each detected face of object 3
        object_3_id = ids[np.argwhere(cond_obj3)]                      # ID of each detected face in object 3


        #######  OBJECT 4  #######
        cond_obj4 = ((ids>18) & (ids<=24))
        object_4_faces = np.sum(np.where(cond_obj4, 1, 0))             # Number of faces detected of object 4
        corners_object_4 = corners[np.argwhere(cond_obj4)]             # Corners of each detected face of object 4
        object_4_id = ids[np.argwhere(cond_obj4)]                      # ID of each detected face in object 4



        detected_faces = np.array([object_1_faces, object_2_faces, object_3_faces, object_4_faces])                # Number of faces for each object
        detected_objects = np.sum(np.where(detected_faces > 0, 1, 0))                                                  # Total number of objects detected
        id_objects = detected_faces > 0


        new_corners = [corners_object_1, corners_object_2, corners_object_3, corners_object_4]
        new_ids = [object_1_id, object_2_id, object_3_id, object_4_id]

     
        quaternions_output = np.zeros((num_cubes, 4))            # Orientations (in quaternion) for each detected object
        positions_output = np.zeros((num_cubes, 3))              # Average positions for each detected object


        t_1_center = np.reshape(np.array([0, 0, -size_cube/2]), (3,))


        # We enter the loop if at least one object has been detected
        if detected_objects > 0:

            # We loop over the maximum number of cubes detectable  (in our case num_cubes = 4)
            for j in range(num_cubes):
                rotation_vector = None

                # For each object we loop over all its detected faces
                for i in range(0, detected_faces[j]):
                    
                    if positions_list[j][i] is not None:          # sometimes it happens that the depth camera returns some None values for certain detected markers     

                        # Estimate pose of each marker and return the values rotation_vector and position_vector---(different from those of camera coefficients)
                        result_pose_estimation = cv2.aruco.estimatePoseSingleMarkers(new_corners[j][i].reshape(corners[0].shape), size_marker, matrix_coefficients, distortion_coefficients)

                        (rotation_vector, _) = (result_pose_estimation[0][0, 0, :], result_pose_estimation[1][0, 0, :])


                        id_marker = int(new_ids[j][i][0])

                        n_faces_none = sum(x is None for x in positions_list[j])           # number of None values returned from the Realsense position estimation module
                        n_feasible_faces = detected_faces[j] - n_faces_none                # total number of feasible detected markers (needed to compute the average of the position vector over the different faces)


                        if new_ids[j][i][0] % 6 == 1:   # (faces: 1, 7, 13, 19)
                            R_camera_1, _ = cv2.Rodrigues(rotation_vector)           # orientation of frame 1 with respect to camera frame
                            
                            ###   COMPUTING THE POSITION OF THE CENTER OF THE CUBE WITH RESPECT TO THE CAMERA  ###
                            position_camera_center = np.dot(R_camera_1, t_1_center) + positions_list[j][i]

                            ###   AVERAGING OVER THE DETECTED FACES OF EACH CUBE  ###
                            position_avg_i = position_camera_center / n_feasible_faces
                            positions_output[j,:] = positions_output[j,:] + position_avg_i


                        if new_ids[j][i][0] % 6 == 2:   # (faces: 2, 8, 14, 20)

                            R_1_2 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])                         # constant orientation of frame 2 with respect to frame 1
                            t_1_2 = np.reshape(np.array([size_cube/2, 0, -size_cube/2]), (3,))           # constant translation vector from frame 2 to frame 1

                            ###   EXPRESSING THE ROTATION OF THE DETECTED FRAME WITH RESPECT TO FRAME 1  ###
                            R_camera_2, _ = cv2.Rodrigues(rotation_vector)
                            R_camera_1 = np.dot(R_camera_2, inv(R_1_2))
                            rotation_vector, _ = cv2.Rodrigues(R_camera_1)
                            R_camera_1, _ = cv2.Rodrigues(rotation_vector)
            
                            ###   COMPUTING THE POSITION OF THE CENTER OF THE CUBE WITH RESPECT TO THE CAMERA  ###
                            position_camera_1_2 = np.dot(np.dot(R_camera_2, inv(R_1_2)), t_1_2)                                    # offset vector to translate from frame 2 to fame 1 
                            position_offset_center = np.dot(R_camera_1, t_1_center)                                                # offset vector used to translate frame 1 to the center of the cube
                            position_camera_center = positions_list[j][i] + position_offset_center - position_camera_1_2           # position of the center of the cube with respect to the camera frame

                            ###   AVERAGING OVER THE DETECTED FACES OF EACH CUBE  ###
                            position_avg_i = position_camera_center / n_feasible_faces
                            positions_output[j,:] = positions_output[j,:] + position_avg_i


                        if new_ids[j][i][0] % 6 == 3:   # (faces: 3, 9, 15, 21)

                            R_1_3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                            t_1_3 = np.reshape(np.array([0, 0, -size_cube]), (3,))

                            ###   EXPRESSING THE ROTATION OF THE DETECTED FRAME WITH RESPECT TO FRAME 1  ###
                            R_camera_3, _ = cv2.Rodrigues(rotation_vector)
                            R_camera_1 = np.dot(R_camera_3, inv(R_1_3))
                            rotation_vector, _ = cv2.Rodrigues(R_camera_1)
                            R_camera_1, _ = cv2.Rodrigues(rotation_vector)

                            ###   COMPUTING THE POSITION OF THE CENTER OF THE CUBE WITH RESPECT TO THE CAMERA  ###
                            position_camera_1_3 = np.dot(np.dot(R_camera_3, inv(R_1_3)), t_1_3)
                            position_offset_center = np.dot(R_camera_1, t_1_center)
                            position_camera_center = positions_list[j][i] + position_offset_center - position_camera_1_3

                            ###   AVERAGING OVER THE DETECTED FACES OF EACH CUBE  ###
                            position_avg_i = position_camera_center / n_feasible_faces
                            positions_output[j,:] = positions_output[j,:] + position_avg_i


                        if new_ids[j][i][0] % 6 == 4:   # (faces: 4, 10, 16, 22)

                            R_1_4 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                            t_1_4 = np.reshape(np.array([-size_cube/2, 0, -size_cube/2]), (3,))

                            ###   EXPRESSING THE ROTATION OF THE DETECTED FRAME WITH RESPECT TO FRAME 1  ###
                            R_camera_4, _ = cv2.Rodrigues(rotation_vector)
                            R_camera_1 = np.dot(R_camera_4, inv(R_1_4))
                            rotation_vector, _ = cv2.Rodrigues(R_camera_1)
                            R_camera_1, _ = cv2.Rodrigues(rotation_vector)

                            ###   COMPUTING THE POSITION OF THE CENTER OF THE CUBE WITH RESPECT TO THE CAMERA  ###
                            position_camera_1_4 = np.dot(np.dot(R_camera_4, inv(R_1_4)), t_1_4)
                            position_offset_center = np.dot(R_camera_1, t_1_center)
                            position_camera_center = positions_list[j][i] + position_offset_center - position_camera_1_4
                            
                            ###   AVERAGING OVER THE DETECTED FACES OF EACH CUBE  ###
                            position_avg_i = position_camera_center / n_feasible_faces
                            positions_output[j,:] = positions_output[j,:] + position_avg_i


                        if new_ids[j][i][0] % 6 == 5:   # (faces: 5, 11, 17, 23)

                            R_1_5 = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
                            t_1_5 = np.reshape(np.array([0, size_cube/2, -size_cube/2]), (3,))

                            ###   EXPRESSING THE ROTATION OF THE DETECTED FRAME WITH RESPECT TO FRAME 1  ###
                            R_camera_5, _ = cv2.Rodrigues(rotation_vector)
                            R_camera_1 = np.dot(R_camera_5, inv(R_1_5))
                            rotation_vector, _ = cv2.Rodrigues(R_camera_1)
                            R_camera_1, _ = cv2.Rodrigues(rotation_vector)

                            ###   COMPUTING THE POSITION OF THE CENTER OF THE CUBE WITH RESPECT TO THE CAMERA  ###
                            position_camera_1_5 =  np.dot(np.dot(R_camera_5, inv(R_1_5)), t_1_5)
                            position_offset_center = np.dot(R_camera_1, t_1_center)
                            position_camera_center = positions_list[j][i] + position_offset_center - position_camera_1_5
                            
                            ###   AVERAGING OVER THE DETECTED FACES OF EACH CUBE  ###
                            position_avg_i = position_camera_center / n_feasible_faces
                            positions_output[j,:] = positions_output[j,:] + position_avg_i


                        if new_ids[j][i][0] % 6 == 0:   # (faces: 6, 12, 18, 24)

                            R_1_6 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                            t_1_6 = np.reshape(np.array([0, -size_cube/2, -size_cube/2]), (3,))

                            ###   EXPRESSING THE ROTATION OF THE DETECTED FRAME WITH RESPECT TO FRAME 1  ###
                            R_camera_6, _ = cv2.Rodrigues(rotation_vector)
                            R_camera_1 = np.dot(R_camera_6, inv(R_1_6))
                            rotation_vector, _ = cv2.Rodrigues(R_camera_1)
                            R_camera_1, _ = cv2.Rodrigues(rotation_vector)

                            ###   COMPUTING THE POSITION OF THE CENTER OF THE CUBE WITH RESPECT TO THE CAMERA  ###
                            position_camera_1_6 = np.dot(np.dot(R_camera_6, inv(R_1_6)), t_1_6)
                            position_offset_center = np.dot(R_camera_1, t_1_center)
                            position_camera_center = positions_list[j][i] + position_offset_center - position_camera_1_6
                            
                            ###   AVERAGING OVER THE DETECTED FACES OF EACH CUBE  ###
                            position_avg_i = position_camera_center / n_feasible_faces
                            positions_output[j,:] = positions_output[j,:] + position_avg_i


                        # Draw a square around the markers
                        cv2.aruco.drawDetectedMarkers(frame, corners)

                        # Draw Axis
                        cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rotation_vector, position_avg_i * n_feasible_faces, 0.1)


                # If the rotation_vector (which was initialized to None) has been previously computed, it means that the corresponding position vector coming from the 
                # RealSense pose estimation module did not output None for that specific marker

                if rotation_vector is not None: 

                    id_objects[j] = True

                    R_camera_1, _ = cv2.Rodrigues(rotation_vector)

                    rotation = R.from_dcm(R_camera_1)       # rotation = R.from_matrix(C_R1) if using python3        (((NOTE: python2 --> from_dcm, as_dcm    |    python3 --> from_matrix, as_matrix)))
                    quaternion = rotation.as_quat()
                    quaternion_output[j,:] = quaternion

                    T_robot_camera = inv(T_camera_robot)

                    T_camera_1 = self.get_transf_matrix(positions_output[j,:], R_camera_1)
                    T_robot_1 = np.dot(T_robot_camera, T_camera_1)
                    positions_output[j,:] = T_robot_1[:3, 3]

                else:

                    positions_output[j, :] = np.array([0, 0, 0])
                    quaternion_output[j, :] = np.array([0, 0, 0, 0])


        return frame, quaternion_output, positions_output, new_ids, id_objects 






    def environment_calibration(self, frame, result, matrix_coefficients, distortion_coefficients, size_marker, size_cube, robot_marker_detected, calibration_marker):
        corners = result[0]
        ids = result[1]
        rejected_img_points = result[2]


        # We enter the loop if at least one marker has been detected
        if len(corners) > 0:

            for i in range(0, len(ids)):

                # Estimate pose of each marker and return the values rotation_vector and position_vector --- (different from those of camera coefficients)
                result_pose_estimation = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_marker, matrix_coefficients, distortion_coefficients)

                (rotation_vector, position_vector) = (result_pose_estimation[0][0, 0, :], result_pose_estimation[1][0, 0, :])

                id_marker = int(ids[i][0])

                if ids[i][0] == calibration_marker and not robot_marker_detected:                  # calibration_marker can be either 48 or 49 (I will use 49 for the following variable names, but the same applies with marker 48)
                    robot_marker_detected = True

                    R_camera_49, _ = cv2.Rodrigues(rotation_vector)                             # rotation matrix from frame 49 (calibration marker) to Camera   
                    inclination_angle = abs(90 - np.degrees(math.acos(R_camera_49[2,2])))       # this angle will be used afterwards to compensate for a small rotational offset around the x-axis
                    t_camera_49 = np.reshape(position_vector, (3,1))

                    R_49_robot = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
                    R_camera_robot = np.dot(R_camera_49, R_49_robot)
                    T_camera_robot = np.concatenate((R_camera_robot, t_camera_49), axis=1)
                    T_camera_robot = np.concatenate((T_camera_robot, np.reshape(np.array([0, 0, 0, 1]), (1,4))), axis=0)        # transformation matrix from robot to camera frame

                    return T_camera_robot, robot_marker_detected, R_camera_49, inclination_angle

        T_camera_robot = np.zeros((4,4))
        R_camera_49 = np.zeros((3,3))
        inclination_angle = 0

        return T_camera_robot, robot_marker_detected, R_camera_49, inclination_angle




    @staticmethod
    def getImageWithMarkers(input_image, detect_res):
        image = input_image.copy()
        corners, ids, rejected = detect_res

        # verify at least one ArUco marker was detected
        if len(corners) > 0:
        
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):

                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
    
                # compute and draw the center (x, y)-coordinates of the ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image


























if __name__ == '__main__':

    full_path = os.path.realpath(__file__)
    dir_path, file_name = os.path.split(full_path)

    dict_to_use, _, _, _ = readConfig(dir_path+'/config.json')
    arucoDetector = ArUcoDetector(dict_to_use)

    calibration_matrix_path = dir_path + '/calibration_matrix.npy'
    distortion_coefficients_path =  dir_path + '/distortion_coefficients.npy'

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    tracker = TrajectoryTracker()

    camera = Camera()
    camera.startStreaming()

    # Running average array
    quat_running_avg_1 = np.zeros((10,4))
    quat_running_avg_2 = np.zeros((10,4)) 
    quat_running_avg_3 = np.zeros((10,4))
    quat_running_avg_4 = np.zeros((10,4))

    robot_marker_detected = False   
    
    # Marker IDs chosen for the environment calibration
    calibration_marker_1 = 49
    calibration_marker_2 = 48

    size_marker = 0.055        # size is expressed in meters (marker length = 5.5 cm)
    size_cube = 0.07           # size is expressed in meters (cube length = 7 cm)

    num_cubes = 4              # maximum number of detectable cubes



    # Initialize the node with rospy
    rospy.init_node('publisher_cube_position')

    # Create publisher
    id_marker_calibration = rospy.Publisher("~id_marker_calibration", Int32, queue_size=10)

    cube1_pose = rospy.Publisher("~cube1_pose", Pose, queue_size=10)
    is_cube1_detected = rospy.Publisher("~is_cube1_detected", Bool, queue_size=10)

    cube2_pose = rospy.Publisher("~cube2_pose", Pose, queue_size=10)
    is_cube2_detected = rospy.Publisher("~is_cube2_detected", Bool, queue_size=10)

    cube3_pose = rospy.Publisher("~cube3_pose", Pose, queue_size=10)
    is_cube3_detected = rospy.Publisher("~is_cube3_detected", Bool, queue_size=10)

    cube4_pose = rospy.Publisher("~cube4_pose", Pose, queue_size=10)
    is_cube4_detected = rospy.Publisher("~is_cube4_detected", Bool, queue_size=10)
    

    rate = rospy.Rate(10)
   

    while True and not rospy.is_shutdown():
        frame = camera.getNextFrame()
        depth_image, color_image = camera.extractImagesFromFrame(frame)

        # Remove unaligned part of the color_image to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        masked_color_image = np.where(depth_image_3d <= 0, grey_color, color_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Detect markers and draw them on the images
        result = arucoDetector.detect(color_image, k, d)
        color_image = ArUcoDetector.getImageWithMarkers(color_image, result)
        masked_color_image = ArUcoDetector.getImageWithMarkers(masked_color_image, result)
        depth_colormap = ArUcoDetector.getImageWithMarkers(depth_colormap, result)

        

        # As long as the calibration markers are not detected, we should not publish the poses of the other cubes

        cond_init = ((calibration_marker_1 in np.squeeze(result[1])) or (calibration_marker_2 in np.squeeze(result[1])))

        if not robot_marker_detected and result[1] is not None and cond_init:
            position_calibration_marker, calibration_marker = tracker.updateTrajectory_calibration(frame, result, calibration_marker_1, calibration_marker_2)

            if position_calibration_marker != 0:
                T_camera_robot, robot_marker_detected, R_camera_49, inclination_angle = arucoDetector.environment_calibration(color_image, result, k, d, size_marker, size_cube, robot_marker_detected, calibration_marker)
                
                if robot_marker_detected:
                    T_camera_robot[0, 3] = position_calibration_marker[0]
                    T_camera_robot[1, 3] = position_calibration_marker[1]
                    T_camera_robot[2, 3] = position_calibration_marker[2]



        # Once the calibration markers have been detected, we can start the pose estimation of the cubes

        if robot_marker_detected:

            if result[1] is not None:

                # Whenever the calibration markers are detected, we should remove them from detected markers (otherwise we would continuously estimate their pose)
                if calibration_marker_1 in result[1]:
                    id_robot_mark = np.argwhere(result[1] == calibration_marker_1)
                    new_corners = np.delete(result[1], id_robot_mark[0][0])
                    del result[0][id_robot_mark[0][0]]

                    result_list = list(result)
                    result_list[1] = np.array([new_corners])
                    result_list[1] = np.transpose(result_list[1], (1, 0))
                    result = tuple(result_list)

                if calibration_marker_2 in result[1]:
                    id_robot_mark2 = np.argwhere(result[1] == calibration_marker_2)
                    new_corners = np.delete(result[1], id_robot_mark2[0][0])
                    del result[0][id_robot_mark2[0][0]]

                    result_list = list(result)
                    result_list[1] = np.array([new_corners])
                    result = tuple(result_list)


                num_visible_markers = np.count_nonzero(result[1])
                result_list = list(result)
                result_list[1] = np.reshape(result_list[1], (1, num_visible_markers))
                result = tuple(result_list)
                

                # We estimate the cube pose if we have at least one detection in that specific frame (besides the calibration markers)
                if int(result[1].shape[1]) > 0:

                    ###   POSE ESTIMATION OF EACH MARKER FROM REALSENSE DEPTH MAPS  ###
                    positions_list, feasible_ids = tracker.updateTrajectory(frame, result, num_cubes)


                    num_markers =  result[1].flatten().shape[0]

                    result_list = list(result)
                    result_list[1] = np.reshape(result_list[1], (1, num_markers))
                    result = tuple(result_list)

                    # The pose estimation from the RealSense (tracker.updateTrajectory) may return None values in some detected marker.
                    # Hence, we should make sure that there are still feasible markers that we could consider to estimate the cube pose.

                    if feasible_ids:

                        ###  ORIENTATION ESTIMATION, POSITION AVERAGING AND FRAME TRANSFORMATIONS  ###
                        pose_image, quaternions, positions, ids_markers, id_objects = arucoDetector.pose_estimation(color_image, result, k, d, size_marker, size_cube, num_cubes, positions_list, T_camera_robot)

                        ###   QUATERNION AVERAGING  ###
                        quat_running_avg = []
                        for i in range(num_cubes):

                            # For each object we check that it has been previously detected in this iteration
                            if i == 0:
                                if id_objects[i] == True:
                                    quat_running_avg_1[0, :] = quaternions[i, :]
                                    quat_running_avg_1 = np.roll(quat_running_avg_1, 1, axis=0)
                                    quaternion_cube_1 = averageQuaternions(quat_running_avg_1)
                                    quaternion_cube_1 = np.reshape(quaternion_cube_1, (1, 4))
                                else:
                                    quaternion_cube_1 = np.reshape(quaternions[i, :], (1, 4))

                            elif i == 1:
                                if id_objects[i] == True:
                                    quat_running_avg_2[0, :] = quaternions[i, :]
                                    quat_running_avg_2 = np.roll(quat_running_avg_2, 1, axis=0)
                                    quaternion_cube_2 = averageQuaternions(quat_running_avg_2)
                                    quaternion_cube_2 = np.reshape(quaternion_cube_2, (1, 4))
                                else:
                                    quaternion_cube_2 = np.reshape(quaternions[i, :], (1, 4))

                            elif i == 2:
                                if id_objects[i] == True:
                                    quat_running_avg_3[0, :] = quaternions[i, :]
                                    quat_running_avg_3 = np.roll(quat_running_avg_3, 1, axis=0)
                                    quaternion_cube_3 = averageQuaternions(quat_running_avg_3)
                                    quaternion_cube_3 = np.reshape(quaternion_cube_3, (1, 4))
                                else:
                                    quaternion_cube_3 = np.reshape(quaternions[i, :], (1, 4))
 
                            elif i == 3:
                                if id_objects[i] == True:
                                    quat_running_avg_4[0, :] = quaternions[i, :]
                                    quat_running_avg_4 = np.roll(quat_running_avg_4, 1, axis=0)
                                    quaternion_cube_4 = averageQuaternions(quat_running_avg_4)
                                    quaternion_cube_4 = np.reshape(quaternion_cube_4, (1, 4))
                                else:
                                    quaternion_cube_4 = np.reshape(quaternions[i, :], (1, 4))


                        quaternions = np.concatenate((quaternion_cube_1, quaternion_cube_2, quaternion_cube_3, quaternion_cube_4), axis=0)



                        ###   ORIENTATION OFFSET CORRECTION AROUND THE X-AXIS  ###
                        alpha = inclination_angle * np.pi / 180
                        R_x = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])

                        for w in range(num_cubes):
                            # The following condition checks that the cube has been detected in this iteration (otherwise both the orientation and the position vectors would be arrays of zeros)
                            if not np.array_equal(quaternions[w, :], np.array([0, 0, 0, 0])):

                                rotation = R.from_quat(np.squeeze(quaternions[w, :]))
                                C_R_cube = rotation.as_dcm()
                      
                                rotation = R.from_dcm(np.dot(R_x, C_R_cube))
                                quaternions[w, :] = rotation.as_quat()


                        ###   ASSIGN MESSAGES TO PUBLISHERS  ###

                        id_calibration = Int32()
                        id_calibration.data = calibration_marker
                        id_marker_calibration.publish(id_calibration)

                        for z in range(num_cubes):
                            p = Pose()
		                    p.position.x = positions[z, 0]
		                    p.position.y = positions[z, 1]
		                    p.position.z = positions[z, 2]
 
		                    p.orientation.x = quaternions[z, 0]
       	                    p.orientation.y = quaternions[z, 1]
                            p.orientation.z = quaternions[z, 2]
                            p.orientation.w = quaternions[z, 3]

		                    is_cube_detected = Bool()
		                    is_cube_detected.data = id_objects[z]

                            if z == 0:
                                cube1_pose.publish(p)
                                is_cube1_detected.publish(is_cube_detected)
                            if z == 1:
                                cube2_pose.publish(p)
                                is_cube2_detected.publish(is_cube_detected)
                            if z == 2:
                                cube3_pose.publish(p)
                                is_cube3_detected.publish(is_cube_detected)
                            if z == 3:
                                cube4_pose.publish(p)
                                is_cube4_detected.publish(is_cube_detected)


		        rate.sleep()


                        # Show images
                        images = np.hstack((pose_image, masked_color_image, depth_colormap))
                        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                        cv2.imshow('RealSense', images)
                        cv2.waitKey(1)

                      
                else:
                    # Show images
                    images = np.hstack((color_image, masked_color_image, depth_colormap))
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', images)
                    cv2.waitKey(1)

            else:
                # Show images
                images = np.hstack((color_image, masked_color_image, depth_colormap))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)

        # Show images
        images = np.hstack((color_image, masked_color_image, depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)


    camera.stopStreaming()


          
