#ifndef BAND_SAFE_BOOL_H_
#define BAND_SAFE_BOOL_H_

#include <condition_variable>
#include <mutex>
namespace band {
class SafeBool {
 public:
  SafeBool() = default;
  ~SafeBool() = default;

  void notify();
  bool wait();
  void terminate();

 private:
  mutable std::mutex m;
  bool flag = false;
  bool exit = false;
  std::condition_variable c;
};

}  // namespace band

#endif  // BAND_SAFE_BOOL_H_
