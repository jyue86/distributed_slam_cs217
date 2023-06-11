#ifndef ARUCOFRONTEND
#define ARUCOFRONTEND

// https://stackoverflow.com/questions/70956515/aruco-markers-pose-estimatimation-exactly-for-which-point-the-traslation-and-r
// https://www.youtube.com/watch?v=cIVZRuVdv1o
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <map>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/aruco/charuco.hpp>
#include <opencv4/opencv2/objdetect/aruco_detector.hpp>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace cv;

class ArucoFrontEnd {
public:
  ArucoFrontEnd();

  // void calibrateAruco();
  void calibrateCharuco();

  // Detection functions
  void detectAruco(Mat img);

  void detectCharucoBoardWithoutCalibration(Mat img);
  void detectArucoPnp(const Vec3d &arucoPos, Mat img,
                      const std::vector<int> &markerIds,
                      const std::vector<std::vector<Point2f>> &markerCorners);
  void detectCharucoBoardWithCalibration(
      Mat img, const std::vector<int> &markerIds,
      const std::vector<std::vector<Point2f>> &markerCorners);
  void getCharucoBoardDataForCalibration(Mat img);

  Mat getCameraMatrix() const;
  Mat getDistCoeffs() const;
  Vec3d getWorldPose() const;
  Vec3d getWorldRot() const;

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
  std::vector<Mat> rvecs, tvecs;

  // Charuco Board
  Mat objectPoints;

  int charucoMarkersX = 5;
  int charucoMarkersY = 7;
  float charucoSquareLength = 0.04;
  float charucoMarkerLength = 0.02;
  aruco::Dictionary charucoDictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  Ptr<aruco::CharucoBoard> charucoBoard = new aruco::CharucoBoard(
      Size(charucoMarkersX, charucoMarkersY), charucoSquareLength,
      charucoMarkerLength, charucoDictionary);

  std::vector<std::vector<Point2f>> allCharucoCorners;
  std::vector<std::vector<int>> allCharucoIds;

  // Detectors
  aruco::ArucoDetector arucoDetector{charucoDictionary,
                                     aruco::DetectorParameters()};
  // Pose data
  Vec3d currentWorldPose;
  Vec3d currentWorldRot;

  // todo: make this a const function???
  std::map<std::set<int>, Vec3d> arucoPosMap;
  bool isCharucoBoard(int minMarkerId);
  int findArucoGroupId(int minMarkerId);
  bool isRotationMatrix(Mat &R);
  Vec3d convertRotationToEuler(Mat &R);
  Vec3d getArucoMarkersWorldPose(int arucoId);
};

#endif
