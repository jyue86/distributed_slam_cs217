#ifndef POSEGRAPH
#define POSEGRAPH

// In planar SLAM example we use Pose2 variables (x, y, theta) to represent the
// robot poses
#include <gtsam/geometry/Pose3.h>

// We will use simple integer Keys to refer to the robot poses.
#include <gtsam/inference/Key.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common
// factors have been provided with the library for solving robotics/SLAM/Bundle
// Adjustment problems. Here we will use Between factors for the relative motion
// described by odometry measurements. We will also use a Between Factor to
// encode the loop closure constraint Also, we will initialize the robot at the
// origin using a Prior factor.
#include <gtsam/slam/BetweenFactor.h>

// When the factors are created, we will add them to a Factor Graph. As the
// factors we are using are nonlinear factors, we will need a Nonlinear Factor
// Graph.
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

// Finally, once all of the factors have been added to our factor graph, we will
// want to solve/optimize to graph to find the best (Maximum A Posteriori) set
// of variable values. GTSAM includes several nonlinear optimizers to perform
// this step. Here we will use the a Gauss-Newton solver
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearISAM.h>

// Once the optimized values have been calculated, we can also calculate the
// marginal covariance of desired variables
#include <gtsam/nonlinear/Marginals.h>

// The nonlinear solvers within GTSAM are iterative solvers, meaning they
// linearize the nonlinear functions around an initial linearization point, then
// solve the linear system to update the linearization point. This happens
// repeatedly until the solver converges to a consistent set of variable values.
// This requires us to specify an initial guess for each variable, held in a
// Values container.
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Values.h>

#include <map>
#include <opencv2/opencv.hpp>
#include <utility>

using namespace cv;

using namespace gtsam;

class PoseGraph {
public:
  PoseGraph();
  void addPose(std::set<int> arucoids, const Vec3d &R, const Vec3d &t);
  void lmOptimize();
  void isamOptimize();
  Values getCurrentEstimate();

private:
  bool maybeLoopClose;
  unsigned int graphId;
  std::set<int> lastArucoIds;
  Vec3d lastPose;
  Vec3d lastRot;

  Values initialPosesEstimate;
  Values currentPosesEstimate;
  NonlinearFactorGraph graph;
  std::map<int, std::map<int, std::pair<Vec3d, Vec3d>>> poseDb;
  ISAM2 isam;
  // int relinearizeInterval = 3;
  // NonlinearISAM nonlinearISAM(relinearizeInterval);

  Eigen::Matrix3d constructRotationMat(const Vec3d &R);
  double calculateDistance(const Vec3d &t1, const Vec3d &t2);
};

#endif
