#include "realsense.hpp"

void configureCamera(config &config, pipeline &pipeline) {
  config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
  config.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8, 30);
  config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
  pipeline.start(config);

  for (unsigned int i = 0; i < 30; ++i) {
    pipeline.wait_for_frames();
  }
}
