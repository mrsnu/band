#ifndef CHROME_TRACER_TRACER_H_
#define CHROME_TRACER_TRACER_H_

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "event.h"

namespace chrome_tracer {

class ChromeTracer {
 public:
  ChromeTracer();
  ChromeTracer(std::string name);

  bool HasStream(std::string stream);
  void AddStream(std::string stream);

  bool HasEvent(std::string stream, int32_t handle);
  void MarkEvent(std::string stream, std::string event);
  int32_t BeginEvent(std::string stream, std::string event);
  void EndEvent(std::string stream, int32_t handle, std::string args = "");

  bool Validate() const;

  std::string Dump() const;
  void Dump(std::string path) const;
  std::string Summary() const;

  void Clear();

 private:
  static size_t GetNextPId();
  std::string name_;
  std::map<std::string, std::map<int32_t, Event>> event_table_;
  std::chrono::system_clock::time_point anchor_;

  size_t count_;
  const size_t pid_;

  mutable std::mutex lock_;
};

}  // namespace chrome_tracer

#endif  // CHROME_TRACER_TRACER_H_