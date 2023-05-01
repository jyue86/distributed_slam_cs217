#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
  std::cout << "Super Basic OpenCV Camera Feed" << std::endl;
  namedWindow("Image");

  char quitKey;
  Mat cameraImage;
  VideoCapture cap(0);

  while (true) {
    cap >> cameraImage;
    if (cameraImage.empty())
      break;

    imshow("Image", cameraImage);
    quitKey = waitKey(10);
    if (quitKey == 27) {
      break;
    }
  }
  cap.release();

  return 0;
}
