#ifndef FEATUREMATCHER
#define FEATUREMATCHER

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

class FeatureMatcher {
public:
  FeatureMatcher();
  Mat orbFeatureMatch(Mat image1, Mat image2);

private:
  Ptr<FeatureDetector> detector;
  Ptr<DescriptorExtractor> descriptor;
  Ptr<DescriptorMatcher> matcher;

  std::vector<Point2f> runRansac();
};

#endif
