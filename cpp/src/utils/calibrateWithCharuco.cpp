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

  char quitKey;
  namedWindow("Image", WINDOW_AUTOSIZE);

  int count = 0;
  int calibImgCount = 0;

  while (true) {
    frameset frames = pipe.wait_for_frames();
    depth_frame depthFrame = frames.get_depth_frame();
    frame imageFrame = frames.get_color_frame();

    Mat image(Size(640, 480), CV_8UC3, (void *)imageFrame.get_data(),
              Mat::AUTO_STEP);
    imshow("Image", image);

    if (count % 50 == 0 && count != 0) {
      // imwrite("calibration_images/img" + std::to_string(calibImgCount) +
      // ".jpg",
      //         image);
      std::cout << "going to calibrate at this view..." << std::endl;
      arucoFrontend.detectCharucoBoardForCalibration(image);
    }

    quitKey = waitKey(10);
    if (quitKey == 27)
      break;
    count++;
  }

  std::cout << "Calibrating now" << std::endl;
  arucoFrontend.calibrate();
  std::cout << arucoFrontend.getCameraMatrix() << std::endl;
  std::cout << arucoFrontend.getDistCoeffs() << std::endl;

  return 0;
}
