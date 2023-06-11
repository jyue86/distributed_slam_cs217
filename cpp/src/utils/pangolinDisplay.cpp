#include "pangolinDisplay.hpp"

void TrajectoryVisualizer::addPose(const cv::Vec3d &R, const cv::Vec3d &t) {
  double x = R[0], y = R[1], z = R[2];
  Quaternion<double> qx(std::cos(x / 2.0), std::sin(x / 2.0), 0.0, 0.0);
  Quaternion<double> qy(std::cos(y / 2.0), 0.0, std::sin(y / 2.0), 0.0);
  Quaternion<double> qz(std::cos(z / 2.0), 0.0, 0.0, std::sin(z / 2.0));

  // Combine the individual quaternions to get the final quaternion
  Quaternion<double> q = qz * qy * qx;

  Isometry3d Twr(Quaterniond(q.w(), q.z(), q.y(), q.x()));
  Twr.pretranslate(Eigen::Vector3d(t[0], t[1], t[2]));
  poses.push_back(Twr);
}

void TrajectoryVisualizer::drawTrajectory() {
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 384.8156600568191, 385.6417742825542,
                                 324.0743160813822, 243.5872251288455, 0.1,
                                 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {
      // draw three axes of each pose
      Eigen::Vector3d Ow = poses[i].translation();
      Eigen::Vector3d Xw = poses[i] * (0.1 * Eigen::Vector3d(1, 0, 0));
      Eigen::Vector3d Yw = poses[i] * (0.1 * Eigen::Vector3d(0, 1, 0));
      Eigen::Vector3d Zw = poses[i] * (0.1 * Eigen::Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    // draw a connection
    for (size_t i = 0; i < poses.size() - 1; i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000); // sleep 5 ms
  }
}
