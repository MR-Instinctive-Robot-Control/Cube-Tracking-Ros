#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Int32, Bool
from geometry_msgs.msg import Pose

from cube_tracking.TrajectoryTracker import TrajectoryTracker

import os
import cv2
import math
import json
import scipy
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import numpy.matlib as npm


def readConfig(path):
    with open(path, 'r') as fin:
        config = json.load(fin)
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


# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)



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

    
    def get_transf_matrix(self, tvec, C_R1):
    
        t = np.reshape(tvec, (3,1))
        T = np.concatenate((C_R1, t), axis=1)
        T = np.concatenate((T, np.reshape(np.array([0, 0, 0, 1]), (1,4))), axis=0)

        return T


    def pose_estimation(self, frame, result, matrix_coefficients, distortion_coefficients, size_marker, size_cube, num_cubes, tvec_depth_list, T_R_ROB):
        corners = result[0]
        corners = np.array(corners)
        ids = result[1]

        
        id_objects = np.array([False, False])

        rejected_img_points = result[2]
        ids = ids.reshape(-1)


        ''' Object 1 '''
        # Number of faces of object 1
        object_1_faces = np.sum(np.where(ids<=6,1,0))
        # Corners of the faces of object 1
        corners_object_1 = corners[np.argwhere(ids<=6)]
        # Id of faces of object 1
        object_1_id = ids[np.argwhere(ids<=6)]


        ''' Object 2 '''
        cond_obj2 = ((ids>6) & (ids<=12))
        # Number of faces of object 2
        object_2_faces = np.sum(np.where(cond_obj2, 1, 0))
        # Corners of the faces of object 2
        corners_object_2 = corners[np.argwhere(cond_obj2)]
        # Id of faces of object 2
        object_2_id = ids[np.argwhere(cond_obj2)]


        ''' Object 3 '''
        cond_obj3 = ((ids>12) & (ids<=18))
        # Number of faces of object 3
        object_3_faces = np.sum(np.where(cond_obj3, 1, 0))
        # Corners of the faces of object 3
        corners_object_3 = corners[np.argwhere(cond_obj3)]
        # Id of faces of object 3
        object_3_id = ids[np.argwhere(cond_obj3)]


        ''' Object 4 '''
        cond_obj4 = ((ids>18) & (ids<=24))
        # Number of faces of object 4
        object_4_faces = np.sum(np.where(cond_obj4, 1, 0))
        # Corners of the faces of object 4
        corners_object_4 = corners[np.argwhere(cond_obj4)]
        # Id of faces of object 4
        object_4_id = ids[np.argwhere(cond_obj4)]


        # Number of faces of each object
        detected_faces = np.array([object_1_faces, object_2_faces, object_3_faces, object_4_faces])
        # Total number of objects detected
        detected_objects = np.sum(np.where(detected_faces>0,1,0))


        new_corners = [corners_object_1, corners_object_2, corners_object_3, corners_object_4]
        new_ids = [object_1_id, object_2_id, object_3_id, object_4_id]


        # Number of quaternions to output = Total number of objects detected
        quaternion_output = np.zeros((num_cubes, 4))
        # Average tvec for each detected object
        tvec_sum = np.zeros((num_cubes, 3))
        tvec_depth_sum = np.zeros((num_cubes, 3))

        id_objects = detected_faces>0


        t_1C = np.array([0, 0, -size_cube/2])
        t_1C = np.reshape(t_1C, (3,))

        # If markers are detected
        if detected_objects > 0:
            # Loop over each object
            for j in range(num_cubes):
                rvec = None
                # Loop over each face of each object
                for i in range(0, detected_faces[j]):
                    
                    if tvec_depth_list[j][i] is not None:
                        # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                        ret = cv2.aruco.estimatePoseSingleMarkers(new_corners[j][i].reshape(corners[0].shape), size_marker, matrix_coefficients, distortion_coefficients)

                        (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])


                        id_marker = int(new_ids[j][i][0])

                        num_none = sum(x is None for x in tvec_depth_list[j])

                        if new_ids[j][i][0] == 1 or new_ids[j][i][0] == 7 or new_ids[j][i][0] == 13 or new_ids[j][i][0] == 19:
                            C_R1, _ = cv2.Rodrigues(rvec)
            
                            tvec_i = (np.dot(C_R1, t_1C) + tvec)/(detected_faces[j]-num_none)
                            tvec_depth_i = (np.dot(C_R1, t_1C) + tvec_depth_list[j][i])/(detected_faces[j]-num_none)

                            tvec_sum[j,:] = tvec_sum[j,:] + tvec_i
                            tvec_depth_sum[j,:] = tvec_depth_sum[j,:] + tvec_depth_i


                        if new_ids[j][i][0] == 2 or new_ids[j][i][0] == 8 or new_ids[j][i][0] == 14 or new_ids[j][i][0] == 20:
                            C_12 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                            C_R2, _ = cv2.Rodrigues(rvec)
                            C_R1 = np.dot(C_R2, inv(C_12))
                            rvec, _ = cv2.Rodrigues(C_R1)

                            t_12 = np.array([size_cube/2, 0, -size_cube/2])
                            t_12 = np.reshape(t_12, (3,))

                            C_R1, _ = cv2.Rodrigues(rvec)
            
                            tvec_i = (np.dot(C_R1, t_1C) + tvec - np.dot(np.dot(C_R2, inv(C_12)), t_12))/(detected_faces[j]-num_none)
                            tvec_depth_i = (np.dot(C_R1, t_1C) + tvec_depth_list[j][i] - np.dot(np.dot(C_R2, inv(C_12)), t_12))/(detected_faces[j]-num_none)

                            tvec_sum[j,:] = tvec_sum[j,:] + tvec_i
                            tvec_depth_sum[j,:] = tvec_depth_sum[j,:] + tvec_depth_i


                        if new_ids[j][i][0] == 3 or new_ids[j][i][0] == 9 or new_ids[j][i][0] == 15 or new_ids[j][i][0] == 21:
                            C_13 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                            C_R3, _ = cv2.Rodrigues(rvec)
                            C_R1 = np.dot(C_R3, inv(C_13))
                            rvec, _ = cv2.Rodrigues(C_R1)

                            t_13 = np.array([0, 0, -size_cube])
                            t_13 = np.reshape(t_13, (3,))

                            C_R1, _ = cv2.Rodrigues(rvec)

                            tvec_i = (np.dot(C_R1, t_1C) + tvec - np.dot(np.dot(C_R3, inv(C_13)), t_13))/(detected_faces[j]-num_none)
                            tvec_depth_i = (np.dot(C_R1, t_1C) + tvec_depth_list[j][i] - np.dot(np.dot(C_R3, inv(C_13)), t_13))/(detected_faces[j]-num_none)
                            
                            tvec_sum[j,:] = tvec_sum[j,:] + tvec_i
                            tvec_depth_sum[j,:] = tvec_depth_sum[j,:] + tvec_depth_i


                        if new_ids[j][i][0] == 4 or new_ids[j][i][0] == 10 or new_ids[j][i][0] == 16 or new_ids[j][i][0] == 22:
                            C_14 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                            C_R4, _ = cv2.Rodrigues(rvec)
                            C_R1 = np.dot(C_R4, inv(C_14))
                            rvec, _ = cv2.Rodrigues(C_R1)

                            t_14 = np.array([-size_cube/2, 0, -size_cube/2])
                            t_14 = np.reshape(t_14, (3,))

                            C_R1, _ = cv2.Rodrigues(rvec)

                            tvec_i = (np.dot(C_R1, t_1C) + tvec - np.dot(np.dot(C_R4, inv(C_14)), t_14))/(detected_faces[j]-num_none)
                            tvec_depth_i = (np.dot(C_R1, t_1C) + tvec_depth_list[j][i] - np.dot(np.dot(C_R4, inv(C_14)), t_14))/(detected_faces[j]-num_none)

                            tvec_sum[j,:] = tvec_sum[j,:] + tvec_i
                            tvec_depth_sum[j,:] = tvec_depth_sum[j,:] + tvec_depth_i


                        if new_ids[j][i][0] == 5 or new_ids[j][i][0] == 11 or new_ids[j][i][0] == 17 or new_ids[j][i][0] == 23:
                            C_15 = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
                            C_R5, _ = cv2.Rodrigues(rvec)
                            C_R1 = np.dot(C_R5, inv(C_15))
                            rvec, _ = cv2.Rodrigues(C_R1)

                            t_15 = np.array([0, size_cube/2, -size_cube/2])
                            t_15 = np.reshape(t_15, (3,))

                            C_R1, _ = cv2.Rodrigues(rvec)

                            tvec_i = (np.dot(C_R1, t_1C) + tvec - np.dot(np.dot(C_R5, inv(C_15)), t_15))/(detected_faces[j]-num_none)
                            tvec_depth_i = (np.dot(C_R1, t_1C) + tvec_depth_list[j][i] - np.dot(np.dot(C_R5, inv(C_15)), t_15))/(detected_faces[j]-num_none)

                            tvec_sum[j,:] = tvec_sum[j,:] + tvec_i
                            tvec_depth_sum[j,:] = tvec_depth_sum[j,:] + tvec_depth_i


                        if new_ids[j][i][0] == 6 or new_ids[j][i][0] == 12 or new_ids[j][i][0] == 18 or new_ids[j][i][0] == 24:
                            C_16 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                            C_R6, _ = cv2.Rodrigues(rvec)
                            C_R1 = np.dot(C_R6, inv(C_16))
                            rvec, _ = cv2.Rodrigues(C_R1)

                            t_16 = np.array([0, -size_cube/2, -size_cube/2])
                            t_16 = np.reshape(t_16, (3,))

                            C_R1, _ = cv2.Rodrigues(rvec)

                            tvec_i = (np.dot(C_R1, t_1C) + tvec - np.dot(np.dot(C_R6, inv(C_16)), t_16))/(detected_faces[j]-num_none)
                            tvec_depth_i = (np.dot(C_R1, t_1C) + tvec_depth_list[j][i] - np.dot(np.dot(C_R6, inv(C_16)), t_16))/(detected_faces[j]-num_none)

                            tvec_sum[j,:] = tvec_sum[j,:] + tvec_i
                            tvec_depth_sum[j,:] = tvec_depth_sum[j,:] + tvec_depth_i


                        # Draw a square around the markers
                        cv2.aruco.drawDetectedMarkers(frame, corners)

                        # Draw Axis
                        cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec_i*(detected_faces[j]-num_none), 0.1)


                if rvec is not None:
                    id_objects[j] = True

                    C_R1, _ = cv2.Rodrigues(rvec)
                    r = R.from_dcm(C_R1)

                    T = self.get_transf_matrix(tvec_sum[j,:], C_R1)
                    T_ROB_R = inv(T_R_ROB)
                    T_robot = np.dot(T_ROB_R, T)
 
                    tvec_sum[j,:] = T_robot[:3, 3]
                    # C_R1 = T_robot[:3, :3]
                    # r = R.from_dcm(C_R1)  					 # r = R.from_matrix(C_R1) if using python3
                                          					 # NOTE: python2 - from_dcm, as_dcm    |    python3 - from_matrix, as_matrix

                    quaternion = r.as_quat()
                    quaternion_output[j,:] = quaternion


                    T_depth = self.get_transf_matrix(tvec_depth_sum[j,:], C_R1)
                    T_robot_depth = np.dot(T_ROB_R, T_depth)

                    tvec_depth_sum[j,:] = T_robot_depth[:3, 3]
                else:
                    tvec_sum[j, :] = np.array([0, 0, 0])
                    tvec_depth_sum[j, :] = np.array([0, 0, 0])
                    quaternion_output[j, :] = np.array([0, 0, 0, 0])


        return frame, quaternion_output, tvec_sum, tvec_depth_sum, new_ids, id_objects 


    def environment_calibration(self, frame, result, matrix_coefficients, distortion_coefficients, size_marker, size_cube, robot_marker_detected, robot_marker_id):
        corners = result[0]
        ids = result[1]
        rejected_img_points = result[2]

        tvec_sum = np.zeros((3,))
        tvec_depth_sum = np.zeros((3,))


        # If markers are detected
        if len(corners) > 0:

            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)

                ret = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_marker, matrix_coefficients, distortion_coefficients)

                (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])

                id_marker = int(ids[i][0])

                if ids[i][0] == robot_marker_id and not robot_marker_detected:
                    robot_marker_detected = True;

                    C_R_49, _ = cv2.Rodrigues(rvec)   #rotation matrix from 49 to Camera   
                    t_R_49 = np.reshape(tvec, (3,1))

                    C_49_ROB = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
                    C_R_ROB = np.dot(C_49_ROB, C_R_49)
                    T_R_ROB = np.concatenate((C_R_ROB, t_R_49), axis=1)
                    T_R_ROB = np.concatenate((T_R_ROB, np.reshape(np.array([0, 0, 0, 1]), (1,4))), axis=0)

                    return T_R_ROB, robot_marker_detected

        T_R_ROB = np.zeros((4,4))

        return T_R_ROB, robot_marker_detected



    @staticmethod
    def getImageWithMarkers(input_image, detect_res):
        image = input_image.copy()
        corners, ids, rejected = detect_res

        # verify *at least* one ArUco marker was detected
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
                #print("[INFO] ArUco marker ID: {}".format(markerID))

        return image


if __name__ == '__main__':

    from cube_tracking.Camera import Camera
    import numpy as np

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
    rvec_RA_obj1 = np.zeros((10,4))
    rvec_RA_obj2 = np.zeros((10,4)) 
    rvec_RA_obj3 = np.zeros((10,4))
    rvec_RA_obj4 = np.zeros((10,4))

    robot_marker_detected = False
    robot_marker_id1 = 49
    robot_marker_id2 = 48

    size_marker = 0.055
    size_cube = 0.07

    num_cubes = 4



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

        
        cond_init = ((robot_marker_id1 in np.squeeze(result[1])) or (robot_marker_id2 in np.squeeze(result[1])))
      
        cond_init = ((robot_marker_id1 in np.squeeze(result[1])) or (robot_marker_id2 in np.squeeze(result[1])))

        if not robot_marker_detected and result[1] is not None and cond_init:
            tvec_depth_robot_id, robot_marker_id = tracker.updateTrajectory_calibration(frame, result, robot_marker_id1, robot_marker_id2)

            if tvec_depth_robot_id != 0:
                T_R_ROB, robot_marker_detected = arucoDetector.environment_calibration(color_image, result, k, d, size_marker, size_cube, robot_marker_detected, robot_marker_id)
                
                if robot_marker_detected:
                    T_R_ROB[0, 3] = tvec_depth_robot_id[0]
                    T_R_ROB[1, 3] = tvec_depth_robot_id[1]
                    T_R_ROB[2, 3] = tvec_depth_robot_id[2]




        if robot_marker_detected:

            if result[1] is not None:

                if robot_marker_id1 in result[1]:
                    id_robot_mark = np.argwhere(result[1] == robot_marker_id1)
                    new_corners = np.delete(result[1], id_robot_mark[0][0])
                    del result[0][id_robot_mark[0][0]]

                    lst = list(result)
                    lst[1] = np.array([new_corners])
                    lst[1] = np.transpose(lst[1], (1, 0))
                    result = tuple(lst)

                if robot_marker_id2 in result[1]:
                    id_robot_mark2 = np.argwhere(result[1] == robot_marker_id2)
                    new_corners = np.delete(result[1], id_robot_mark2[0][0])
                    del result[0][id_robot_mark2[0][0]]

                    lst = list(result)
                    lst[1] = np.array([new_corners])
                    result = tuple(lst)


                num_visible_markers = np.count_nonzero(result[1])
                lst = list(result)
                lst[1] = np.reshape(lst[1], (1, num_visible_markers))
                result = tuple(lst)
                

                if int(result[1].shape[1]) > 0:

                    tvec_depth_list, feasible_ids = tracker.updateTrajectory(frame, result, num_cubes)


                    num_markers =  result[1].flatten().shape[0]

                    lst = list(result)
                    lst[1] = np.reshape(lst[1], (1, num_markers))
                    result = tuple(lst)



                    if feasible_ids:
                        pose_image, rvec, tvec, tvec_depth, ids_markers, id_objects = arucoDetector.pose_estimation(color_image, result, k, d, size_marker, size_cube, num_cubes, tvec_depth_list, T_R_ROB)

                        rvec_RA_obj = []
                        for i in range(num_cubes):
                            if i == 0:
                                if id_objects[i] == True:
                                    rvec_RA_obj1[0, :] = rvec[i, :]
                                    rvec_RA_obj1 = np.roll(rvec_RA_obj1, 1, axis=0)
                                    rvec_1 = averageQuaternions(rvec_RA_obj1)
                                    rvec_1 = np.reshape(rvec_1, (1, 4))
                                else:
                                    rvec_1 = rvec[i, :]
                                    rvec_1 = np.reshape(rvec[i, :], (1, 4))

                            elif i == 1:
                                if id_objects[i] == True:
                                    rvec_RA_obj2[0, :] = rvec[i, :]
                                    rvec_RA_obj2 = np.roll(rvec_RA_obj2, 1, axis=0)
                                    rvec_2 = averageQuaternions(rvec_RA_obj2)
                                    rvec_2 = np.reshape(rvec_2, (1, 4))
                                else:
                                    rvec_2 = rvec[i, :]
                                    rvec_2 = np.reshape(rvec[i, :], (1, 4))

                            elif i == 2:
                                if id_objects[i] == True:
                                    rvec_RA_obj3[0, :] = rvec[i, :]
                                    rvec_RA_obj3 = np.roll(rvec_RA_obj3, 1, axis=0)
                                    rvec_3 = averageQuaternions(rvec_RA_obj3)
                                    rvec_3 = np.reshape(rvec_3, (1, 4))
                                else:
                                    rvec_3 = rvec[i, :]
                                    rvec_3 = np.reshape(rvec[i, :], (1, 4))
 
                            elif i == 3:
                                if id_objects[i] == True:
                                    rvec_RA_obj4[0, :] = rvec[i, :]
                                    rvec_RA_obj4 = np.roll(rvec_RA_obj4, 1, axis=0)
                                    rvec_4 = averageQuaternions(rvec_RA_obj4)
                                    rvec_4 = np.reshape(rvec_4, (1, 4))
                                else:
                                    rvec_4 = rvec[i, :]
                                    rvec_4 = np.reshape(rvec[i, :], (1, 4))


                        rvec = np.concatenate((rvec_1, rvec_2, rvec_3, rvec_4), axis=0)

                        id_calibration = Int32()
                        id_calibration.data = robot_marker_id
                        id_marker_calibration.publish(id_calibration)

                        for z in range(num_cubes):
                            p = Pose()
		            p.position.x = tvec_depth[z, 0]
		            p.position.y = tvec_depth[z, 1]
		            p.position.z = tvec_depth[z, 2]
 
		            p.orientation.x = rvec[z, 0]
       	                    p.orientation.y = rvec[z, 1]
                            p.orientation.z = rvec[z, 2]
                            p.orientation.w = rvec[z, 3]

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


          
