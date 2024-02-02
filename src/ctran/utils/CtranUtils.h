#include <sstream>
#include <vector>

template <typename T>
std::string vecToStr(const std::vector<T>& vec) {
  std::stringstream ss;
  bool first = true;
  for (auto it : vec) {
    if (!first) {
      ss << ", ";
    }
    ss << it;
    first = false;
  }
  return ss.str();
}
