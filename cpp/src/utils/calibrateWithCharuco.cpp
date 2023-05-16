#include "../frontend/arucoFrontEnd.hpp"
#include "opencv2/opencv.hpp"
#include "realsense.hpp"
#include <iostream>
#include <librealsense2/rs.hpp>
#include <string>

using namespace rs2;
using namespace cv;

int main() {
  ArucoFrontEnd arucoFrontend;

  config cfg;
  pipeline pipe;
  configureCamera(cfg, pipe);

  char inputKey = 0;
  bool maybeCapture = false;
  namedWindow("Charuco", WINDOW_AUTOSIZE);

  while (true) {
    frameset frames = pipe.wait_for_frames();
    depth_frame depthFrame = frames.get_depth_frame();
    frame imageFrame = frames.get_color_frame();

    Mat image(Size(640, 480), CV_8UC3, (void *)imageFrame.get_data(),
              Mat::AUTO_STEP);

    arucoFrontend.detectCharucoBoardWithoutCalibration(image, maybeCapture);
    if (maybeCapture) {
      std::cout << "calibrated at this view..." << std::endl;
      maybeCapture = false;
    }

    inputKey = waitKey(10);
    std::cout << inputKey << std::endl;
    if (inputKey == 'c')
      maybeCapture = true;
    else if (inputKey == 27)
      break;
  }

  std::cout << "Calibrating now" << std::endl;
  arucoFrontend.calibrate();
  std::cout << arucoFrontend.getCameraMatrix() << std::endl;
  std::cout << arucoFrontend.getDistCoeffs() << std::endl;

  return 0;
}
