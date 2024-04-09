#ifndef CHROME_TRACER_EVENT_H_
#define CHROME_TRACER_EVENT_H_

#include <chrono>
#include <iostream>
#include <string>

namespace chrome_tracer {

struct Event {
  enum class EventStatus {
    Running = 0,
    Finished = 1,
    Instantanous = 2,
  };

  Event(std::string name, EventStatus status = EventStatus::Running)
      : name(name), status_(status) {
    start = std::chrono::system_clock::now();
  }

  void Finish() {
    if (status_ != EventStatus::Running) {
      std::cerr << "Cannot finish the event not running";
      abort();
    }
    end = std::chrono::system_clock::now();
    status_ = EventStatus::Finished;
  }

  EventStatus GetStatus() const { return status_; }

  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  std::string name;
  std::string args;

 private:
  EventStatus status_;
};

}  // namespace chrome_tracer

#endif  // CHROME_TRACER_EVENT_H_