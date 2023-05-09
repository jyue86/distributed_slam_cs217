import cv2


class FrontEndWrapper:
    def __init__(self) -> None:
        self.orb = cv2.ORB_create() 

    def run_orb_detection(self, img):
        kpt = self.orb.detect(img, None)
        kpt, des = self.orb.compute(img, kpt)
        print(kpt)
        orb_img = cv2.drawKeypoints(img, kpt, None, color=(0, 255, 0), flags=0)
        return orb_img

    def calculate_cam_pose(self):
        pass

    def _estimate_f_mat(self):
        pass

    def _estimate_e_mat(self):
        pass
