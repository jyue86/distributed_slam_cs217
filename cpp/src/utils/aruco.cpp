#include "aruco.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

void generateArucoTag(std::string filePath) {
  Mat markerImage;
  aruco::Dictionary dictionary =
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  aruco::generateImageMarker(dictionary, 2, 200, markerImage, 1);
  imwrite(filePath, markerImage);
}

void generateCharuoBoard(std::string filePath) {}

int main(int argc, char *argv[]) {
  generateArucoTag("/home/jyue86/distributed_slam/aruco_tags/aruco_tag4.jpg");
  return 0;
}
