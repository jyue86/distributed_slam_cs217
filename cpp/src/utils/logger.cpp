#include "logger.hpp"

void logInfo(std::string infoMessage) {
  std::cout << "Log INFO - " << infoMessage << std::endl;
}

void logError(std::string infoMessage) {
  std::cerr << "Log ERROR - " << infoMessage << std::endl;
}
