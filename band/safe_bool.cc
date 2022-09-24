#include "band/safe_bool.h"

namespace Band {
void SafeBool::notify() {
  std::lock_guard<std::mutex> lock(m);
  flag = true;
  c.notify_one();
}

bool SafeBool::wait() {
  std::unique_lock<std::mutex> lock(m);
  while (!exit && !flag) {
    c.wait(lock);
  }
  flag = false;
  return exit;
}

void SafeBool::terminate() {
  std::lock_guard<std::mutex> lock(m);
  exit = true;
  c.notify_all();
}
} // namespace Band