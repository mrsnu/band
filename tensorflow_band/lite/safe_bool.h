#ifndef TENSORFLOW_LITE_SAFE_BOOL_H_
#define TENSORFLOW_LITE_SAFE_BOOL_H_

#include <mutex>
#include <condition_variable>

class SafeBool {
 public:
  SafeBool(void) : m(), c() {}

  ~SafeBool(void) {}

  void notify() {
    std::lock_guard<std::mutex> lock(m);
    flag = true;
    c.notify_one();
  }

  bool wait() {
    std::unique_lock<std::mutex> lock(m);
    while (!exit && !flag) {
      c.wait(lock);
    }
    flag = false;
    return exit;
  }

  void terminate() {
    std::lock_guard<std::mutex> lock(m);
    exit = true;
    c.notify_all();
  }

 private:
  mutable std::mutex m;
  bool flag = false;
  bool exit = false;
  std::condition_variable c;
};

#endif  // TENSORFLOW_LITE_SAFE_BOOL_H_
