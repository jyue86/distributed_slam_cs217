import cv2
from cv2 import aruco

aruco_dict_type = aruco.DICT_6X6_250
width = 6
length = 8
square_length = 1
marker_length = 0.8
aruco_params = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
board = aruco.CharucoBoard((length,width), square_length, marker_length, aruco_dict)

def create_and_save_charuco_board():
    save_dir = "./charuco_boards/"
    imboard = board.generateImage((2000, 2000))
    cv2.imwrite(save_dir + "chessboard2.tiff", imboard)

def estimatePose(img, camera_matrix, dist_coeffs):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    corners, ids, rejected_img_pts = aruco.detectMarkers(gray_img, aruco_dict_type, parameters=aruco_params)
    aruco.refineDetectedMarkers(gray_img, board, corners, ids, rejected_img_pts)

    if ids != None:
        charuco_ret_val, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray_img, board)
        aruco_detected_img = aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (0, 255, 0))
        ret_val, R, t = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)
        if ret_val == True:
            aruco_detected_img = aruco.drawAxis(aruco_detected_img, camera_matrix, dist_coeffs, R, t, 100)
    else:
        cv2.imshow("ArucoDetection", img)


if __name__ == "__main__":
    # create_and_save_charuco_board()
    pass
