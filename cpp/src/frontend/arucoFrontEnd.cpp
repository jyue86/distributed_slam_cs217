#include "../frontend/arucoFrontEnd.hpp"
#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

// ArucoFrontEnd::~ArucoFrontEnd() { delete board; }

void ArucoFrontEnd::calibrate() {
  int calibrationFlags = 0;
  double error = aruco::calibrateCameraAruco(
      allCornersConcatenated, allIdsConcatenated, allMarkerCountPerFrame, board,
      imgSize, cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);
}

void ArucoFrontEnd::detectCharucoBoardWithCalibration(Mat img) {
  Mat imgCopy;
  // img.copyTo(imgCopy);
  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  if (markerIds.size() > 0) {
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    // aruco::interpolateCornersCharuco(markerCorners, markerIds, img, board,
    //                                  charucoCorners, charucoIds,
    //                                  cameraMatrix, distCoeffs);

    if (charucoIds.size() > 0) {
      aruco::drawDetectedMarkers(imgCopy, charucoCorners, charucoIds,
                                 Scalar(255, 0, 0));
      // Vec3d rvec, tvec;
      // bool valid =
      //     aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board,
      //                                     cameraMatrix, distCoeffs, rvec,
      //                                     tvec);

      // if (valid)
      //   drawFrameAxes(imgCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
      // aruco::drawDetectedCornersCharuco(img, charucoCorners);
    }
  }
}

void ArucoFrontEnd::detectCharucoBoardWithoutCalibration(Mat img,
                                                         bool maybeCalibrate) {
  // Mat imgCopy(Size(640, 480), CV_8UC3, (void *)img.data, Mat::AUTO_STEP);
  Mat imgCopy;
  img.copyTo(imgCopy);

  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;

  arucoDetector.detectMarkers(img, markerCorners, markerIds);
  if (markerIds.size() > 0) {
    aruco::drawDetectedMarkers(imgCopy, markerCorners);

    if (maybeCalibrate) {
      std::vector<Point2f> objPoints;
      std::vector<int> imgPoints;
      // board->matchImagePoints(markerCorners, markerIds, objPoints,
      // imgPoints);
      allIdsConcatenated.insert(allIdsConcatenated.end(), markerIds.begin(),
                                markerIds.end());
      allCornersConcatenated.insert(allCornersConcatenated.end(),
                                    markerCorners.begin(), markerCorners.end());
      allMarkerCountPerFrame.push_back(allIdsConcatenated.size());
    }
  }
  imshow("Charuco", imgCopy);
}

Mat ArucoFrontEnd::getCameraMatrix() const { return cameraMatrix; }

Mat ArucoFrontEnd::getDistCoeffs() const { return distCoeffs; }
