#ifndef PANGOLIN_DISPLAY
#define PANGOLIN_DISPLAY

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/nonlinear/Values.h>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <vector>

using namespace Eigen;
using namespace gtsam;

class TrajectoryVisualizer {
public:
  void addPose(const cv::Vec3d &R, const cv::Vec3d &t);
  void readPoseFromISAM(const Values &currentEstimate);
  void drawTrajectory();

private:
  std::vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
  double calculateDistance(double x1, double y1, double z1, double x2,
                           double y2, double z2);
};

#endif
