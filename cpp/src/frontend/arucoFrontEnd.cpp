#include "../frontend/arucoFrontEnd.hpp"
#include <algorithm>
#include <cstddef>
#include <opencv2/calib3d.hpp>
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
  // Run 1
  // for (int i : {19, 20, 25, 26}) {
  //   arucoPosMap[i] = Vec3d(0.5, 0, 0);
  // }
  // for (int i : {21, 22, 27, 28}) {
  //   arucoPosMap[i] = Vec3d(1.0, 0, 0);
  // }
  // for (int i : {43, 44, 49, 50}) {
  //   arucoPosMap[i] = Vec3d(0.5, 0, 0);
  // }

  // Table at home
  for (int i : {19, 20, 25, 26}) {
    arucoPosMap[i] = Vec3d(0, 0, 0);
  }
  for (int i : {21, 22, 27, 28}) {
    arucoPosMap[i] = Vec3d(-0.25, 0, 0);
    // arucoPosMap[i] = Vec3d(-4, 0, 0);
  }
  for (int i : {41, 42, 47, 48}) {
    arucoPosMap[i] = Vec3d(-0.25, 0.25, 0);
    // arucoPosMap[i] = Vec3d(-8, 0, 0);
  }
  for (int i : {43, 44, 49, 50}) {
    arucoPosMap[i] = Vec3d(-0.25, 0.5, 0);
    // arucoPosMap[i] = Vec3d(-4, 8, 0);
  }
  for (int i : {29, 30, 35, 36}) {
    arucoPosMap[i] = Vec3d(0, 0.5, 0);
    // arucoPosMap[i] = Vec3d(0, 8, 0);
  }
  for (int i : {53, 54, 59, 60}) {
    arucoPosMap[i] = Vec3d(0, 0.25, 0);
    // arucoPosMap[i] = Vec3d(0, 4, 0);
  }
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

void ArucoFrontEnd::detectAruco(Mat img, bool maybePrintMarkerPos,
                                bool maybePrintEstimatedPos) {
  lastArucoIds.clear();
  Mat imgCopy;
  img.copyTo(imgCopy);

  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;
  arucoDetector.detectMarkers(img, markerCorners, markerIds);

  if (markerIds.size() > 0) {
    int minMarkerId = *std::min_element(markerIds.begin(), markerIds.end());
    bool isCBoard = isCharucoBoard(minMarkerId);

    if (isCBoard) {
      lastArucoIds.insert(0);
      detectCharucoBoardWithCalibration(imgCopy, markerIds, markerCorners);
    } else {
      detectArucoPnp(imgCopy, markerIds, markerCorners, maybePrintMarkerPos,
                     maybePrintEstimatedPos);
    }
  } else {
    imshow("Aruco", imgCopy);
  }
}

// x is red, y is green, z is blue
void ArucoFrontEnd::detectArucoPnp(
    Mat img, const std::vector<int> &markerIds,
    const std::vector<std::vector<Point2f>> &markerCorners,
    bool maybePrintMarkerPos, bool maybePrintEstimatedPos) {
  cv::aruco::drawDetectedMarkers(img, markerCorners, markerIds);
  int nMarkers = markerCorners.size();
  std::vector<cv::Vec3d> arucoRvecs(nMarkers), arucoTvecs(nMarkers);

  // Calculate pose for each marker
  for (int i = 0; i < nMarkers; i++) {
    solvePnP(objectPoints, markerCorners.at(i), cameraMatrix, distCoeffs,
             arucoRvecs.at(i), arucoTvecs.at(i));
  }

  // Draw axis for each marker
  Vec3d avgTvec = Vec3d::zeros(); //, avgEulerAngles = Vec3d::zeros();
  int nValidMarkers = 0;
  for (unsigned int i = 0; i < markerIds.size(); i++) {
    cv::drawFrameAxes(img, cameraMatrix, distCoeffs, arucoRvecs[i],
                      arucoTvecs[i], 0.05);
    if (arucoPosMap.count(markerIds[i])) {
      lastArucoIds.insert(markerIds[i]);
      avgTvec += getArucoMarkersWorldPose(markerIds[i], maybePrintMarkerPos) -
                 arucoTvecs[i];
      nValidMarkers += 1;
    }
  }
  avgTvec /= nValidMarkers;
  // set at a constant height
  // avgTvec[2] = 2;

  // Only use the first aruco marker for rotation
  cv::Vec3d rvec;
  rvec = arucoRvecs.at(0);
  Vec3d oppositeDirRvec = rvec * -1;
  Mat R = Mat::zeros(Size(3, 3), CV_64FC1);
  Rodrigues(oppositeDirRvec, R);

  Mat t = R * avgTvec;
  currentWorldPose = Vec3d(t.reshape(3).at<Vec3d>());
  currentWorldRot = convertRotationToEuler(R);
  if (maybePrintEstimatedPos)
    std::cout << "World pos: " << currentWorldPose << std::endl;

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

// void ArucoFrontEnd::detectCharucoBoardWithoutCalibration(Mat img) {
//   Mat imgCopy;
//   img.copyTo(imgCopy);
//
//   std::vector<int> markerIds;
//   std::vector<std::vector<Point2f>> markerCorners;
//   arucoDetector.detectMarkers(img, markerCorners, markerIds);
//
//   if (markerIds.size() > 0) {
//
//   } else {
//     imshow("Aruco", imgCopy);
//   }
// }

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

Vec3d ArucoFrontEnd::getArucoMarkersWorldPose(int markerId,
                                              bool maybePrintArucoPos) {
  Vec3d worldPose = arucoPosMap[markerId];
  if (maybePrintArucoPos)
    std::cout << "Marker " << markerId << ": " << worldPose << std::endl;
  return worldPose;
}

std::set<int> ArucoFrontEnd::getLastArucoIds() const { return lastArucoIds; }

Mat ArucoFrontEnd::constructRotationMat(const Vec3d &R) {
  Mat x = (Mat_<double>(3, 3) << 1, 0, 0, 0, std::cos(R[0]), -std::sin(R[0]), 0,
           std::sin(R[0]), std::cos(R[0]));
  Mat y = (Mat_<double>(3, 3) << std::cos(R[1]), 0, std::sin(R[1]), 0, 1, 0,
           -std::sin(R[1]), 0, std::cos(R[1]));
  Mat z = (Mat_<double>(3, 3) << std::cos(R[2]), -std::sin(R[2]), 0,
           std::sin(R[2]), std::cos(R[2]), 0, 0, 0, 1);
  return x * y * z;
}
