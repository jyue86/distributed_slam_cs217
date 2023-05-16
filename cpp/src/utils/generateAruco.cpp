#include "generateAruco.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <string>

void generateArucoTag(int id, std::string filePath) {
  Mat markerImage;
  aruco::Dictionary dictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  aruco::generateImageMarker(dictionary, id, 200, markerImage, 1);
  imwrite(filePath + "aruco_tag" + std::to_string(id) + ".jpg", markerImage);
}

void generateArucoBoard(int id, std::string filePath) {
  int markersX = 6;
  int markersY = 8;
  float markerLength = 0.04;
  float markerSeparation = 0.01;
  aruco::Dictionary dictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  cv::aruco::Board board = aruco::GridBoard(
      Size(markersX, markersY), markerLength, markerSeparation, dictionary);

  Mat boardImage;
  board.generateImage(Size(600, 500), boardImage, 10, 1);
  imwrite(filePath + "aruco_board" + std::to_string(id) + ".jpg", boardImage);
}

void generateCharuoBoard(std::string filePath) {
  int markersX = 8;
  int markersY = 6;
  float squareLength = 1;
  float markerLength = 0.8;
  aruco::Dictionary dictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  Ptr<aruco::CharucoBoard> board = new cv::aruco::CharucoBoard(
      Size(markersX, markersY), squareLength, markerLength, dictionary);

  Mat boardImg;
  board->generateImage(Size(2000, 2000), boardImg);
  imwrite(filePath + "charuco_board.jpg", boardImg);
}

int main(int argc, char *argv[]) {
  // for (int i = 0; i < 15; i++) {
  //   generateArucoTag(i, "/home/jyue86/distributed_slam/cpp/aruco_tags/");
  // }

  // for (int i = 0; i < 15; i++) {
  //   generateArucoBoard(i, "/home/jyue86/distributed_slam/cpp/aruco_boards/");
  // }
  //

  generateCharuoBoard("/home/jyue86/distributed_slam/cpp/aruco_boards");
  return 0;
}
