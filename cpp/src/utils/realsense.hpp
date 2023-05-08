#ifndef REALSENSE
#define REALSENSE

#include <librealsense2/rs.hpp>

using namespace rs2;

void configureCamera(config &config, pipeline &pipeline);

#endif
