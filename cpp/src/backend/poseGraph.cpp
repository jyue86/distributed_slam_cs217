#include "poseGraph.hpp"
#include <cmath>
#include <limits>
#include <opencv2/core/matx.hpp>

PoseGraph::PoseGraph() { graphId = 1; }

void PoseGraph::addPose(int arucoId, const Vec3d &R, const Vec3d &t) {
  if (graphId == 1) {
    Pose3 priorMean = Pose3::Create(Rot3::identity(), Point3(0, 0, 0));
    // noiseModel::Diagonal::shared_ptr priorNoise =
    //     noiseModel::Diagonal::Variances(
    //         Vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4));
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas(
        (Vector(6) << Vector3::Constant(1e-6), Vector3::Constant(1e-4))
            .finished());
    graph.add(PriorFactor<Pose3>(1, Pose3(), priorNoise));
    lastRot = R;
    lastPose = t;
    initialPosesEstimate.insert(
        graphId,
        Pose3::Create(Rot3(constructRotationMat(R)), Point3(t[0], t[1], t[2])));
    std::cout << "Prior Added graph id " << graphId << std::endl;
    currentPosesEstimate = initialPosesEstimate;
    graphId += 1;
    return;
  }

  std::cout << "Calculating dist: " << calculateDistance(t, lastPose)
            << std::endl;
  double dist = calculateDistance(t, lastPose);
  if (dist < 1e-2 || dist > 0.5) {
    return;
  }
  Vec3d rDiff = R - lastRot;
  Vec3d tDiff = t - lastPose;

  Rot3 rotation = Rot3(constructRotationMat(rDiff));
  Point3 translation(tDiff[0], tDiff[1], tDiff[2]);
  // Pose2 odometry(worldTranslation[0], worldTranslation[1], theta);
  Pose3 odometry = Pose3::Create(rotation, translation);
  noiseModel::Diagonal::shared_ptr odometryNoise =
      noiseModel::Diagonal::Variances(
          Vector6(1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2));
  graph.add(
      BetweenFactor<Pose3>(graphId - 1, graphId, odometry, odometryNoise));
  std::cout << "Between Added graph id " << graphId << std::endl;
  initialPosesEstimate.insert(
      graphId,
      Pose3::Create(Rot3(constructRotationMat(R)), Point3(t[0], t[1], t[2])));

  // Check for loop closure
  if (arucoId != lastArucoId && poseDb.count(arucoId)) {
    std::cout << "Loop closure!!!" << std::endl;
    double minDistance = std::numeric_limits<double>::max();
    int correspondingGraphId;
    for (auto item : poseDb[arucoId]) {
      double dist = calculateDistance(t, item.second.second);
      if (dist < minDistance) {
        minDistance = dist;
        correspondingGraphId = item.first;
      }
    }

    rDiff = R - poseDb[arucoId][correspondingGraphId].first;
    tDiff = t - poseDb[arucoId][correspondingGraphId].second;
    rotation = Rot3(constructRotationMat(rDiff));
    translation = Point3{tDiff[0], tDiff[1], tDiff[2]};
    Pose3 loopClosureOdometry = Pose3::Create(rotation, translation);
    graph.add(BetweenFactor<Pose3>(graphId, correspondingGraphId,
                                   loopClosureOdometry, odometryNoise));
  }
  poseDb[arucoId][graphId] = std::pair<Vec3d, Vec3d>{R, t};

  // Optimize w/ isam
  isamOptimize();

  Pose3 lastOptimizedPose = currentPosesEstimate.at<Pose3>(graphId);
  Vector3 opsRotation = lastOptimizedPose.rotation().xyz();
  Vector3 opsTranslation = lastOptimizedPose.translation();
  double tx = opsTranslation.x(), ty = opsTranslation.y(),
         tz = opsTranslation.z(), rx = opsRotation.x(), ry = opsRotation.y(),
         rz = opsRotation.z();

  lastArucoId = arucoId;
  // lastRot = Vec3d(tx, ty, tz);
  // lastPose = Vec3d(rx, ry, rz);
  lastRot = R;
  lastPose = t;
  graphId += 1;
}

void PoseGraph::lmOptimize() {
  initialPosesEstimate =
      LevenbergMarquardtOptimizer(graph, initialPosesEstimate).optimize();
  initialPosesEstimate.print();
  // TODO: update graph with new poses from result
}

void PoseGraph::isamOptimize() {
  graph.print("Graph");
  // isam.print("ISAM");
  std::cout << "---------" << std::endl;
  isam.update(graph, initialPosesEstimate);
  currentPosesEstimate = isam.calculateEstimate();
  graph.resize(0);
  initialPosesEstimate.clear();
}

Values PoseGraph::getCurrentEstimate() {
  Values currentEstimate = isam.calculateEstimate();
  currentEstimate.print("Current Estimate:");
  return currentEstimate;
  // return initialPosesEstimate;
}

Eigen::Matrix3d PoseGraph::constructRotationMat(const Vec3d &R) {
  Eigen::Matrix3d x, y, z;
  x << 1, 0, 0, 0, std::cos(R[0]), -std::sin(R[0]), 0, std::sin(R[0]),
      std::cos(R[0]);
  y << std::cos(R[1]), 0, std::sin(R[1]), 0, 1, 0, -std::sin(R[1]), 0,
      std::cos(R[1]);
  z << std::cos(R[2]), -std::sin(R[2]), 0, std::sin(R[2]), std::cos(R[2]), 0, 0,
      0, 1;
  return x * y * z;
}

double PoseGraph::calculateDistance(const Vec3d &t1, const Vec3d &t2) {
  Vec3d diff = t1 - t2;
  return std::sqrt(std::pow(diff[0], 2) + std::pow(diff[1], 2) +
                   std::pow(diff[2], 2));
}
