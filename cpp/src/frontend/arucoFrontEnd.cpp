#include "../frontend/arucoFrontEnd.hpp"
#include <algorithm>
#include <cstddef>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <string>

ArucoFrontEnd::ArucoFrontEnd() {
  objectPoints = Mat(4, 1, CV_32FC3);
  objectPoints.ptr<cv::Vec3f>(0)[0] =
      cv::Vec3f(-charucoMarkerLength / 2.f, charucoMarkerLength / 2.f, 0);
  objectPoints.ptr<cv::Vec3f>(0)[1] =
      cv::Vec3f(charucoMarkerLength / 2.f, charucoMarkerLength / 2.f, 0);
  objectPoints.ptr<cv::Vec3f>(0)[2] =
      cv::Vec3f(charucoMarkerLength / 2.f, -charucoMarkerLength / 2.f, 0);
  objectPoints.ptr<cv::Vec3f>(0)[3] =
      cv::Vec3f(-charucoMarkerLength / 2.f, -charucoMarkerLength / 2.f, 0);

  std::set<int> charucoIds;
  for (int i = 0; i <= 16; i++) {
    charucoIds.insert(i);
  }
  arucoPosMap[charucoIds] = Vec3d(0, 0, 0);
  arucoPosMap[{19, 20, 25, 26}] = Vec3d(0.5, 0, 0);
  arucoPosMap[{21, 22, 27, 28}] = Vec3d(1.0, 0, 0);
  arucoPosMap[{43, 44, 49, 50}] = Vec3d(0.5, 0, 0);
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

void ArucoFrontEnd::detectAruco(Mat img) {
  Mat imgCopy;
  img.copyTo(imgCopy);

  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  if (markerIds.size() > 0) {
    int minMarkerId = *std::min_element(markerIds.begin(), markerIds.end());
    bool isCBoard = isCharucoBoard(minMarkerId);

    if (isCBoard) {
      detectCharucoBoardWithCalibration(imgCopy, markerIds, markerCorners);
    } else {
      detectArucoPnp(getArucoMarkersWorldPose(minMarkerId), imgCopy, markerIds,
                     markerCorners);
    }
  } else {
    imshow("Aruco", imgCopy);
  }
}

void ArucoFrontEnd::detectArucoPnp(
    const Vec3d &arucoPos, Mat img, const std::vector<int> &markerIds,
    const std::vector<std::vector<Point2f>> &markerCorners) {
  cv::aruco::drawDetectedMarkers(img, markerCorners, markerIds);
  int nMarkers = markerCorners.size();
  std::vector<cv::Vec3d> arucoRvecs(nMarkers), arucoTvecs(nMarkers);

  // Calculate pose for each marker
  for (int i = 0; i < nMarkers; i++) {
    solvePnP(objectPoints, markerCorners.at(i), cameraMatrix, distCoeffs,
             arucoRvecs.at(i), arucoTvecs.at(i));
  }

  // Draw axis for each marker
  for (unsigned int i = 0; i < markerIds.size(); i++) {
    cv::drawFrameAxes(img, cameraMatrix, distCoeffs, arucoRvecs[i],
                      arucoTvecs[i], 0.05);
  }

  // Only use the first aruco marker
  cv::Vec3d rvec, tvec;
  rvec = arucoRvecs.at(0);
  tvec = arucoTvecs.at(0);
  Vec3d oppositeDirRvec = rvec * -1;
  Vec3d oppositeDirTvec = arucoPos + (tvec * -1);
  Mat R = Mat::zeros(Size(3, 3), CV_64FC1);
  Rodrigues(oppositeDirRvec, R);

  Mat t = R * oppositeDirTvec;
  currentWorldPose = Vec3d(t.reshape(3).at<Vec3d>());
  currentWorldRot = convertRotationToEuler(R);
  // x is red, y is green, z is blue

  std::string rString = std::to_string(currentWorldRot[0]) + " " +
                        std::to_string(currentWorldRot[1]) + " " +
                        std::to_string(currentWorldRot[2]);
  std::string tString = std::to_string(currentWorldPose[0]) + " " +
                        std::to_string(currentWorldPose[1]) + " " +
                        std::to_string(currentWorldPose[2]);
  putText(img, rString, Point(40, 20), FONT_HERSHEY_PLAIN, 1.5,
          Scalar(0, 255, 0));
  putText(img, tString, Point(40, 40), FONT_HERSHEY_PLAIN, 1.5,
          Scalar(0, 255, 0));

  imshow("Aruco", img);
}

void ArucoFrontEnd::detectCharucoBoardWithCalibration(
    Mat img, const std::vector<int> &markerIds,
    const std::vector<std::vector<cv::Point2f>> &markerCorners) {
  cv::aruco::drawDetectedMarkers(img, markerCorners, markerIds);
  std::vector<cv::Point2f> charucoCorners;
  std::vector<int> charucoIds;
  cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img,
                                       charucoBoard, charucoCorners, charucoIds,
                                       cameraMatrix, distCoeffs);
  // if at least one charuco corner detected
  if (charucoIds.size() > 0) {
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::aruco::drawDetectedCornersCharuco(img, charucoCorners, charucoIds,
                                          color);
    cv::Vec3d rvec, tvec;
    bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds,
                                                     charucoBoard, cameraMatrix,
                                                     distCoeffs, rvec, tvec);

    Vec3d oppositeDirRvec = rvec * -1;
    Vec3d oppositeDirTvec = tvec * -1;
    oppositeDirTvec[2] *= -1;
    Mat R = Mat::zeros(Size(3, 3), CV_64FC1);
    Rodrigues(oppositeDirRvec, R);

    Mat t = R * oppositeDirTvec;
    currentWorldPose = Vec3d(t.reshape(3).at<Vec3d>());
    currentWorldRot = convertRotationToEuler(R);
    // x is red, y is green, z is blue

    std::string rString = std::to_string(currentWorldRot[0]) + " " +
                          std::to_string(currentWorldRot[1]) + " " +
                          std::to_string(currentWorldRot[2]);
    std::string tString = std::to_string(currentWorldPose[0]) + " " +
                          std::to_string(currentWorldPose[1]) + " " +
                          std::to_string(currentWorldPose[2]);
    putText(img, rString, Point(40, 20), FONT_HERSHEY_PLAIN, 1.5,
            Scalar(0, 255, 0));
    putText(img, tString, Point(40, 40), FONT_HERSHEY_PLAIN, 1.5,
            Scalar(0, 255, 0));

    // if charuco pose is valid
    if (valid) {
      cv::drawFrameAxes(img, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
    }
  }
  imshow("Aruco", img);
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

Mat ArucoFrontEnd::getCameraMatrix() const { return cameraMatrix; }

Mat ArucoFrontEnd::getDistCoeffs() const { return distCoeffs; }

Vec3d ArucoFrontEnd::getWorldRot() const { return currentWorldRot; }

Vec3d ArucoFrontEnd::getWorldPose() const { return currentWorldPose; }

bool ArucoFrontEnd::isCharucoBoard(int minMarkerId) {
  return minMarkerId >= 0 && minMarkerId <= 16;
}

int ArucoFrontEnd::findArucoGroupId(int minMarkerId) {
  return (minMarkerId / 4) * 4;
}

bool ArucoFrontEnd::isRotationMatrix(Mat &R) {
  Mat rotationMatT = R.t();
  Mat maybeIdentity = rotationMatT * R;
  Mat identityMat = Mat::eye(3, 3, maybeIdentity.type());

  return norm(maybeIdentity - identityMat) < 1e-6;
}

Vec3d ArucoFrontEnd::convertRotationToEuler(Mat &R) {
  assert(isRotationMatrix(R));
  float sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                       R.at<double>(1, 0) + R.at<double>(1, 0));
  bool isSingular = sy < 1e-6;

  float x, y, z;
  if (!isSingular) {
    x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    y = atan2(-R.at<double>(2, 0), sy);
    z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
  } else {
    x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    y = atan2(-R.at<double>(2, 0), sy);
    z = 0;
  }

  return Vec3d(x, y, z);
}

Vec3d ArucoFrontEnd::getArucoMarkersWorldPose(int arucoId) {
  for (auto item : arucoPosMap) {
    if (item.first.count(arucoId) == 1) {
      return item.second;
    }
  }
  return Vec3d::zeros();
}
