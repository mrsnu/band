#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_TIMER_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_TIMER_H_

#include <iostream>
#include <chrono>
#include <thread>
#include <functional>

namespace tflite {
namespace benchmark {

class Timer {
    bool clear = false;
 
public:
    Timer() = default;
    ~Timer() = default;

    void setTimeout(std::function<void(void)> function, int delay);
    void setInterval(std::function<void(void)> function, int interval);
    void stop();
};

}   /// namespace benchmark
}   /// namespace tflite

#endif //TENSORFLOW_LITE_TOOLS_BENCHMARK_TIMER_H_
