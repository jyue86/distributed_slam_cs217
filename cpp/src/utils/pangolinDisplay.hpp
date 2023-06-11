#ifndef PANGOLIN_DISPLAY
#define PANGOLIN_DISPLAY

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <vector>

using namespace Eigen;

class TrajectoryVisualizer {
public:
  void addPose(const cv::Vec3d &R, const cv::Vec3d &t);
  void drawTrajectory();

private:
  std::vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
};

#endif
