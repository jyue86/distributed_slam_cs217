import cv2
import numpy as np
from frontend.front_end_wrapper import FrontEndWrapper
from utils.aruco import estimatePose
from utils.rs_utils import configure_realsense_cam

if __name__ == "__main__":
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("ORB", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("ArucoDetection", cv2.WINDOW_AUTOSIZE)
    pipeline, config = configure_realsense_cam()
    frontend = FrontEndWrapper()

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Image", color_image)
        
        orb_detected_img = frontend.run_orb_detection(color_image)
        cv2.imshow("ORB", orb_detected_img)

        quit_key = cv2.waitKey(10)
        if quit_key == 27:
            break

    cv2.destroyAllWindows()
