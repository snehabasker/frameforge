// M0 placeholder executable: proves the CMake configuration, compiler, and
// directory layout all work end to end before any real engine code exists.
// Replaced in spirit (not literally removed) once M4 adds the real CLI.
#include <iostream>

#include "frameforge/version.hpp"

int main() {
  std::cout << "frameforge engine skeleton " << frameforge::kVersionMajor
            << "." << frameforge::kVersionMinor << "."
            << frameforge::kVersionPatch << " (" << frameforge::kMilestone
            << ")\n";
  return 0;
}
