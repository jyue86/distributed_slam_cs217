#include "backend/poseGraph.hpp"
#include "frontend/arucoFrontEnd.hpp"
#include "frontend/featureMatcher.hpp"
#include "utils/pangolinDisplay.hpp"
#include "utils/realsense.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <fstream>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace rs2;

void recordVideo(const std::string &video) {
  VideoWriter videoWriter(video, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                          Size(640, 480));
  config cfg;
  pipeline pipe;
  char quitKey;
  configureCamera(cfg, pipe);
  namedWindow("Image", WINDOW_AUTOSIZE);

  while (true) {
    frameset frames = pipe.wait_for_frames();
    depth_frame depthFrame = frames.get_depth_frame();
    frame imageFrame = frames.get_color_frame();

    Mat img(Size(640, 480), CV_8UC3, (void *)imageFrame.get_data(),
            Mat::AUTO_STEP);
    videoWriter.write(img);
    imshow("Image", img);

    quitKey = waitKey(10);
    if (quitKey == 27) {
      break;
    }
  }

  destroyAllWindows();
  videoWriter.release();
}

void runSlam() {
  FeatureMatcher featureMatcher;
  config cfg;
  pipeline pipe;
  configureCamera(cfg, pipe);
  ArucoFrontEnd arucoFrontEnd;
  PoseGraph poseGraph;

  char quitKey;
  namedWindow("Aruco");

  while (true) {
    frameset frames = pipe.wait_for_frames();
    depth_frame depthFrame = frames.get_depth_frame();
    frame imageFrame = frames.get_color_frame();

    Mat img(Size(640, 480), CV_8UC3, (void *)imageFrame.get_data(),
            Mat::AUTO_STEP);

    arucoFrontEnd.detectAruco(img);
    Vec3d worldPose = arucoFrontEnd.getWorldPose();
    if (worldPose != Vec3d::zeros()) {
      poseGraph.addPose(1);
    }

    quitKey = waitKey(10);
    if (quitKey == 27)
      break;
  }
  destroyAllWindows();
}

void runSlamOnVideo(const std::string &video, const std::string &posePath) {
  ArucoFrontEnd arucoFrontEnd;
  PoseGraph poseGraph;
  VideoCapture cap(video);
  TrajectoryVisualizer tVisualizer;

  char quitKey;
  namedWindow("Aruco");

  std::ofstream poseOutputFile;
  poseOutputFile.open(posePath);

  while (true) {
    Mat img;
    cap >> img;
    if (img.empty())
      break;

    arucoFrontEnd.detectAruco(img);
    Vec3d t = arucoFrontEnd.getWorldPose();
    Vec3d R = arucoFrontEnd.getWorldRot();
    if (t != Vec3d::zeros() && R != Vec3d::zeros()) {
      tVisualizer.addPose(R, t);
    }

    for (int i = 0; i < 3; i++) {
      poseOutputFile << R[i] << " ";
    }
    for (int i = 0; i < 3; i++) {
      poseOutputFile << t[i] << " ";
    }
    poseOutputFile << std::endl;

    quitKey = waitKey(10);
    if (quitKey == 27)
      break;
  }
  poseOutputFile.close();
  destroyAllWindows();
  tVisualizer.drawTrajectory();
}

int main(int argc, char *argv[]) {
  std::string action, video, posePath;
  std::set<std::string> validActions{"record", "slam_video"};

  // Argument Parsing
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "action", po::value<std::string>(), "choose which action to take")(
      "video", po::value<std::string>(),
      "choose which path to write to or read from video")(
      "pose", po::value<std::string>(),
      "choose which path to write the pose data");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("action")) {
    action = vm["action"].as<std::string>();
    if (!validActions.count(action)) {
      std::cerr << "Invalid action given" << std::endl;
      return -1;
    }
  } else {
    action = "slam_video";
  }

  if (vm.count("video")) {
    video = vm["video"].as<std::string>();
  } else {
    std::cerr << "Video path was not given" << std::endl;
    return -1;
  }

  if (vm.count("pose")) {
    posePath = vm["pose"].as<std::string>();
  } else {
    std::cerr << "Pose path was not given" << std::endl;
    return -1;
  }

  if (action == "record") {
    recordVideo(video);
  } else if (action == "slam_video") {
    runSlamOnVideo(video, posePath);
  } else {
    runSlam();
  }
  return 0;
}
