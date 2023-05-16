#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/aruco/charuco.hpp>
#include <opencv4/opencv2/objdetect/aruco_detector.hpp>
#include <vector>

using namespace cv;

class ArucoFrontEnd {
public:
  // ~ArucoFrontEnd();
  void calibrate();
  void detectCharucoBoardWithCalibration(Mat img);
  void detectCharucoBoardWithoutCalibration(Mat img,
                                            bool maybeCalibrate = false);
  Mat getCameraMatrix() const;
  Mat getDistCoeffs() const;

private:
  // Calibration
  Size imgSize{640, 480};
  Mat cameraMatrix, distCoeffs;
  std::vector<std::vector<cv::Point2f>> allCornersConcatenated;
  std::vector<int> allIdsConcatenated;
  std::vector<int> allMarkerCountPerFrame;

  int markersX = 4;
  int markersY = 3;
  float squareLength = 0.04;
  float markerLength = 0.01;
  aruco::Dictionary dictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_50);
  Ptr<aruco::GridBoard> board = new aruco::GridBoard(
      Size(markersX, markersY), squareLength, markerLength, dictionary);

  // Detectors
  aruco::ArucoDetector arucoDetector{dictionary, aruco::DetectorParameters()};

  // Poses
  std::vector<Mat> rvecs, tvecs;
};
