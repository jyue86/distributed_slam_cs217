#include "featureMatcher.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

FeatureMatcher::FeatureMatcher() {
  detector = ORB::create();
  descriptor = ORB::create();
  matcher = DescriptorMatcher::create("BruteForce-Hamming");
}

Mat FeatureMatcher::orbFeatureMatch(Mat image1, Mat image2) {
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  detector->detect(image1, keypoints1);
  detector->detect(image2, keypoints2);

  descriptor->compute(image1, keypoints1, descriptors1);
  descriptor->compute(image2, keypoints2, descriptors2);

  Mat output;
  drawKeypoints(image2, keypoints1, output, Scalar(0, 255, 0),
                DrawMatchesFlags::DEFAULT);
  return output;
}
