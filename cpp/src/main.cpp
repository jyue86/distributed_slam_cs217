#include "frontend/arucoFrontEnd.hpp"
#include "frontend/featureMatcher.hpp"
#include "utils/realsense.hpp"
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace rs2;

int main() {
  FeatureMatcher featureMatcher;
  config cfg;
  pipeline pipe;
  configureCamera(cfg, pipe);
  ArucoFrontEnd arucoFrontEnd;

  char quitKey;
  namedWindow("Aruco");
  // namedWindow("Orb");

  std::vector<Mat> images;

  while (true) {
    frameset frames = pipe.wait_for_frames();
    depth_frame depthFrame = frames.get_depth_frame();
    frame imageFrame = frames.get_color_frame();

    Mat img(Size(640, 480), CV_8UC3, (void *)imageFrame.get_data(),
            Mat::AUTO_STEP);
    images.push_back(img);

    // if (images.size() > 1) {
    //   int lastIdx = images.size() - 1;
    //   imshow("Orb", featureMatcher.orbFeatureMatch(images[lastIdx - 1],
    //                                                images[lastIdx]));
    // }
    // arucoFrontEnd.detectArucoBoardWithCalibration(img);
    // arucoFrontEnd.detectCharucoBoardWithoutCalibration(img);
    arucoFrontEnd.detectCharucoBoardWithCalibration(img);

    quitKey = waitKey(10);
    if (quitKey == 27)
      break;
  }

  return 0;
}
