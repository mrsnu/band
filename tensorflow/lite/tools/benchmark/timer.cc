#include "tensorflow/lite/tools/benchmark/timer.h"

namespace tflite {
namespace benchmark {

void Timer::setTimeout(std::function<void(void)> function, int delay) {
  this->clear = false;
  std::thread t([=]() {
    if(this->clear) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    if(this->clear) return;
    function();
  });
  t.detach();
}

void Timer::setInterval(std::function<void(void)> function, int interval) {
  this->clear = false;
  std::thread t([=]() {
    while(true) {
      function();
      if(this->clear) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      if(this->clear) return;
    }
  });
  t.detach();
}

void Timer::stop() {
  this->clear = true;
}

} /// namespace benchmark
} /// namespace tflite
