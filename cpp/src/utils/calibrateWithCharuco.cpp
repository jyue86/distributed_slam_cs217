#include "../frontend/arucoFrontEnd.hpp"
#include "opencv2/opencv.hpp"
#include "realsense.hpp"
#include <cstring>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

using namespace rs2;
using namespace cv;

int main(int argc, char *argv[]) {
  int MAX_N_IMAGES = 40;
  std::string save_path =
      true ? "./calibration_imgs/img" : "./checkerboard_calibs/img";
  ArucoFrontEnd arucoFrontend;
  if (argc != 2 &&
      !(std::strcmp(argv[1], "0") == 0 && std::strcmp(argv[1], "1")) == 0) {
    std::cout << "Invalid args..." << std::endl;
    return 1;
  } else if (std::strcmp(argv[1], "0") == 0) {
    config cfg;
    pipeline pipe;
    configureCamera(cfg, pipe);

    int count = 0, INTERVAL = 250, imgNum = 0;
    bool maybeCapture = false;
    char inputKey = 0;
    // namedWindow("Image");
    namedWindow("Aruco", WINDOW_AUTOSIZE);

    while (true) {
      frameset frames = pipe.wait_for_frames();
      depth_frame depthFrame = frames.get_depth_frame();
      frame imageFrame = frames.get_color_frame();

      Mat image(Size(640, 480), CV_8UC3, (void *)imageFrame.get_data(),
                Mat::AUTO_STEP);
      // imshow("Image", image);
      // arucoFrontend.detectArucoBoardWithoutCalibration(image);
      arucoFrontend.detectCharucoBoardWithoutCalibration(image);

      // count != 0 && count % INTERVAL == 0
      if (maybeCapture) {
        std::cout << "adding img " << imgNum << std::endl;
        imwrite(save_path + std::to_string(imgNum) + ".jpg", image);
        arucoFrontend.getCharucoBoardDataForCalibration(image);
        imgNum += 1;
        maybeCapture = false;
        // if (imgNum == MAX_N_IMAGES)
        //   break;
      }

      inputKey = waitKey(10);
      if (inputKey == 27)
        break;
      else if (inputKey == 'a') {
        maybeCapture = true;
      }
      count += 1;
    }

    arucoFrontend.calibrateCharuco();
    std::cout << arucoFrontend.getCameraMatrix() << std::endl;
    std::cout << arucoFrontend.getDistCoeffs() << std::endl;
    std::cout << arucoFrontend.getReprojectionError() << std::endl;
  } else {
    for (int i = 0; i < MAX_N_IMAGES; i++) {
      Mat image = imread(save_path + std::to_string(i) + ".jpg");
      // arucoFrontend.getArucoBoardDataForCalibration(image);
      arucoFrontend.getCharucoBoardDataForCalibration(image);
      std::cout << "done with img " << i << std::endl;
    }
    // arucoFrontend.calibrateAruco();
    arucoFrontend.calibrateCharuco();
    std::cout << arucoFrontend.getCameraMatrix() << std::endl;
    std::cout << arucoFrontend.getDistCoeffs() << std::endl;
  }

  return 0;
}
