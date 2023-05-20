#include "../frontend/arucoFrontEnd.hpp"
#include <exception>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <ostream>

void ArucoFrontEnd::calibrateAruco() {
  int calibrationFlags = 0;
  std::cout << allCornersConcatenated.size() << " " << allIdsConcatenated.size()
            << " " << allMarkerCountPerFrame.size() << std::endl;
  double error = aruco::calibrateCameraAruco(
      allCornersConcatenated, allIdsConcatenated, allMarkerCountPerFrame, board,
      imgSize, cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);
  std::cout << "Error: " << error << std::endl;
}

void ArucoFrontEnd::calibrateCharuco() {
  int calibrationFlags = 0;
  std::cout << allCharucoCorners.size() << " " << allCharucoIds.size()
            << std::endl;
  double repError = cv::aruco::calibrateCameraCharuco(
      allCharucoCorners, allCharucoIds, charucoBoard, imgSize, cameraMatrix,
      distCoeffs, rvecs, tvecs, calibrationFlags);
  std::cout << "Error: " << repError << std::endl;
}

float ArucoFrontEnd::getReprojectionError() {
  // float error;
  // for (int i = 0; i < 5; i++) {
  //   projectPoints(all, InputArray rvec, InputArray tvec, InputArray
  //   cameraMatrix, InputArray distCoeffs, OutputArray imagePoints)
  // }
  return 0.0;
}

void ArucoFrontEnd::detectArucoBoardWithCalibration(Mat img) {
  Mat imgCopy;
  img.copyTo(imgCopy);

  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  if (markerIds.size() > 0) {
    aruco::drawDetectedMarkers(imgCopy, markerCorners);
    Vec3d rvec, tvec;

    Mat objectPoints, imgPoints;
    board->matchImagePoints(markerCorners, markerIds, objectPoints, imgPoints);
    try {
      solvePnP(objectPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
    } catch (const std::exception &e) {
      std::cout << "solvePnP didn't work ..." << std::endl;
      // Sometimes, the imgPoints and objectPoints are of size 0x0
      // std::cout << objectPoints.size() << " " << imgPoints.size() <<
      // std::endl;
    }
    // std::cout << "Rotation: " << rvec << std::endl;
    // std::cout << "Translation: " << tvec << std::endl;

    int nMarkersDetected = (int)objectPoints.total() / 4;
    std::cout << nMarkersDetected << std::endl;
    if (nMarkersDetected > 0)
      drawFrameAxes(imgCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
  }
  imshow("Aruco", imgCopy);
}

void ArucoFrontEnd::detectArucoBoardWithoutCalibration(Mat img) {
  Mat imgCopy;
  img.copyTo(imgCopy);

  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;

  arucoDetector.detectMarkers(img, markerCorners, markerIds);
  if (markerIds.size() > 0) {
    aruco::drawDetectedMarkers(imgCopy, markerCorners);
  }
  imshow("Aruco", imgCopy);
}

void ArucoFrontEnd::getArucoBoardDataForCalibration(Mat img) {
  Mat imgCopy;
  img.copyTo(imgCopy);
  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  std::vector<std::vector<Point2f>> rejectedMarkerCorners;

  std::vector<Point2f> objPoints;
  std::vector<int> imgPoints;

  arucoDetector.detectMarkers(img, markerCorners, markerIds,
                              rejectedMarkerCorners);
  if (markerIds.size() > 0) {
    allIdsConcatenated.insert(allIdsConcatenated.end(), markerIds.begin(),
                              markerIds.end());
    allCornersConcatenated.insert(allCornersConcatenated.end(),
                                  markerCorners.begin(), markerCorners.end());
    allMarkerCountPerFrame.push_back(markerIds.size());
  }
}

void ArucoFrontEnd::detectCharucoBoardWithoutCalibration(Mat img) {
  Mat imgCopy;
  img.copyTo(imgCopy);

  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  if (markerIds.size() > 0) {
    aruco::drawDetectedMarkers(imgCopy, markerCorners);
    std::vector<Point2f> charucoCorners;
    std::vector<int> charucoIds;
    aruco::interpolateCornersCharuco(markerCorners, markerIds, img,
                                     charucoBoard, charucoCorners, charucoIds);

    if (charucoIds.size() > 0)
      aruco::drawDetectedCornersCharuco(imgCopy, charucoCorners, noArray(),
                                        Scalar(0, 255, 0));
  }
  imshow("Aruco", imgCopy);
}

void ArucoFrontEnd::getCharucoBoardDataForCalibration(Mat img) {
  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  if (markerIds.size() > 0) {
    aruco::drawDetectedMarkers(img, markerCorners);
    std::vector<Point2f> charucoCorners;
    std::vector<int> charucoIds;
    aruco::interpolateCornersCharuco(markerCorners, markerIds, img,
                                     charucoBoard, charucoCorners, charucoIds);

    if (charucoIds.size() > 0) {
      allCharucoIds.push_back(charucoIds);
      allCharucoCorners.push_back(charucoCorners);
    }
  }
}

void ArucoFrontEnd::detectCharucoBoardWithCalibration(Mat img) {
  Mat imgCopy;
  img.copyTo(imgCopy);

  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  //  if at least one marker detected
  if (markerIds.size() > 0) {
    cv::aruco::drawDetectedMarkers(imgCopy, markerCorners, markerIds);
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img,
                                         charucoBoard, charucoCorners,
                                         charucoIds, cameraMatrix, distCoeffs);
    // if at least one charuco corner detected
    if (charucoIds.size() > 0) {
      cv::Scalar color = cv::Scalar(0, 255, 0);
      cv::aruco::drawDetectedCornersCharuco(imgCopy, charucoCorners, charucoIds,
                                            color);
      cv::Vec3d rvec, tvec;
      bool valid = cv::aruco::estimatePoseCharucoBoard(
          charucoCorners, charucoIds, charucoBoard, cameraMatrix, distCoeffs,
          rvec, tvec);

      // if charuco pose is valid
      if (valid)
        cv::drawFrameAxes(imgCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
    }
  }
  imshow("Aruco", imgCopy);
}

Mat ArucoFrontEnd::getCameraMatrix() const { return cameraMatrix; }

Mat ArucoFrontEnd::getDistCoeffs() const { return distCoeffs; }
