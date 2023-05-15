#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/aruco/charuco.hpp>
#include <opencv4/opencv2/objdetect/aruco_detector.hpp>
#include <vector>

using namespace cv;

class ArucoFrontEnd {
public:
  ArucoFrontEnd();
  void detectCharucoBoardWithCalibration(Mat img);
  void detectCharucoBoardWithoutCalibration(Mat img);
  void detectCharucoBoardForCalibration(Mat image);
  void calibrate();
  Mat getCameraMatrix() const;
  Mat getDistCoeffs() const;

private:
  // Calibration
  Size imgSize;
  std::vector<std::vector<Point2f>> allCharucoCorners;
  std::vector<std::vector<int>> allCharucoIds;
  std::vector<std::vector<Point2f>> allImagePoints;
  std::vector<std::vector<Point3f>> allObjectPoints;
  Mat cameraMatrix, distCoeffs;

  // Board intialization
  int markersX = 8;
  int markersY = 6;
  float squareLength = 1;
  float markerLength = 0.8;
  aruco::Dictionary dictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  Ptr<aruco::CharucoBoard> board = new cv::aruco::CharucoBoard(
      Size(markersX, markersY), squareLength, markerLength, dictionary);

  // Detectors
  aruco::ArucoDetector arucoDetector;
  aruco::CharucoDetector charucoDetector{*board};

  // Poses
  std::vector<Mat> rvecs, tvecs;
};
