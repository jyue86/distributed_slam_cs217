#ifndef ARUCO
#define ARUCO

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

void generateArucoTag(std::string filePath);
void generateCharuoBoard(int id, std::string filePath);

#endif
