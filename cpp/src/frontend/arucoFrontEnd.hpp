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
  void calibrateAruco();
  void calibrateCharuco();
  float getReprojectionError();

  void detectArucoBoardWithCalibration(Mat img);
  void detectArucoBoardWithoutCalibration(Mat img);
  void getArucoBoardDataForCalibration(Mat img);

  void detectCharucoBoardWithoutCalibration(Mat img);
  void detectCharucoBoardWithCalibration(Mat img);
  void getCharucoBoardDataForCalibration(Mat img);

  Mat getCameraMatrix() const;
  Mat getDistCoeffs() const;

private:
  // Calibration
  Size imgSize{640, 480};
  Mat cameraMatrix =
      (Mat_<double>(3, 3) << 384.8156600568191, 0, 324.0743160813822, 0,
       385.6417742825542, 243.5872251288455, 0, 0, 1);
  Mat distCoeffs =
      (Mat_<double>(5, 1) << -0.06456556984873951, 0.08738224785458196,
       0.001220983520104041, 0.001826405907161455, -0.033195537518265);
  std::vector<std::vector<cv::Point2f>> allCornersConcatenated;
  std::vector<int> allIdsConcatenated;
  std::vector<int> allMarkerCountPerFrame;

  // Aruco Board
  int markersX = 4;
  int markersY = 3;
  float squareLength = 0.04;
  float markerLength = 0.01;
  aruco::Dictionary dictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_50);
  Ptr<aruco::GridBoard> board = new aruco::GridBoard(
      Size(markersX, markersY), squareLength, markerLength, dictionary);

  // Charuco Board
  int charucoMarkersX = 5;
  int charucoMarkersY = 7;
  float charucoSquareLength = 0.04;
  float charucoMarkerLength = 0.02;
  aruco::Dictionary charucoDictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_100);
  Ptr<aruco::CharucoBoard> charucoBoard = new cv::aruco::CharucoBoard(
      Size(charucoMarkersX, charucoMarkersY), charucoSquareLength,
      charucoMarkerLength, charucoDictionary);

  std::vector<std::vector<cv::Point2f>> allCharucoCorners;
  std::vector<std::vector<int>> allCharucoIds;

  // Detectors
  aruco::ArucoDetector arucoDetector{dictionary, aruco::DetectorParameters()};

  // Poses
  std::vector<Mat> rvecs, tvecs;
};
