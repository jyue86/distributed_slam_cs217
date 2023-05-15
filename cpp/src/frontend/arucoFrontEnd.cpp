#include "../frontend/arucoFrontEnd.hpp"
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

ArucoFrontEnd::ArucoFrontEnd() { imgSize = Size(640, 480); }

void ArucoFrontEnd::detectCharucoBoardForCalibration(Mat image) {
  std::vector<std::vector<Point2f>> currentCharucoCorners;
  std::vector<std::vector<int>> currentCharucoIds;
  std::vector<std::vector<Point3f>> currentObjectPoints;
  std::vector<std::vector<Point2f>> currentImagePoints;
  std::cout << "going to detect points" << std::endl;
  std::cout << currentCharucoCorners.size() << " " << currentCharucoIds.size()
            << std::endl;
  std::cout << image.size << " " << image.channels() << std::endl;
  charucoDetector.detectBoard(image, currentCharucoCorners, currentCharucoIds);
  std::cout << "going to match points" << std::endl;
  board->matchImagePoints(currentCharucoCorners, currentCharucoIds,
                          currentObjectPoints, currentImagePoints);

  std::cout << "got past matching points" << std::endl;
  allObjectPoints.insert(allObjectPoints.end(), currentObjectPoints.begin(),
                         currentObjectPoints.end());
  allImagePoints.insert(allImagePoints.end(), currentImagePoints.begin(),
                        currentImagePoints.end());
  allCharucoCorners.insert(allCharucoCorners.end(),
                           currentCharucoCorners.begin(),
                           currentCharucoCorners.end());
  allCharucoIds.insert(allCharucoIds.end(), currentCharucoIds.begin(),
                       currentCharucoIds.end());
}

void ArucoFrontEnd::calibrate() {
  int calibrationFlags = 0;
  double error = aruco::calibrateCameraCharuco(
      allCharucoCorners, allCharucoIds, board, imgSize, cameraMatrix,
      distCoeffs, rvecs, tvecs, calibrationFlags);
}

void ArucoFrontEnd::detectCharucoBoardWithCalibration(Mat img) {
  Mat imgCopy;
  img.copyTo(imgCopy);
  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  if (markerIds.size() > 0) {
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    aruco::interpolateCornersCharuco(markerCorners, markerIds, img, board,
                                     charucoCorners, charucoIds, cameraMatrix,
                                     distCoeffs);

    if (charucoIds.size() > 0) {
      aruco::drawDetectedCornersCharuco(imgCopy, charucoCorners, charucoIds,
                                        Scalar(255, 0, 0));
      Vec3d rvec, tvec;
      bool valid =
          aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board,
                                          cameraMatrix, distCoeffs, rvec, tvec);

      if (valid)
        drawFrameAxes(imgCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
      aruco::drawDetectedCornersCharuco(img, charucoCorners);
    }
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
    aruco::interpolateCornersCharuco(markerCorners, markerIds, img, board,
                                     charucoCorners, charucoIds);

    if (charucoIds.size() > 0)
      aruco::drawDetectedCornersCharuco(imgCopy, charucoCorners);
  }
  imshow("Charuco", imgCopy);
}

Mat ArucoFrontEnd::getCameraMatrix() const { return cameraMatrix; }

Mat ArucoFrontEnd::getDistCoeffs() const { return distCoeffs; }
