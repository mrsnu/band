#include "tracer.h"

#include <chrono>
#include <fstream>
#include <iostream>

namespace chrome_tracer {

namespace {

std::string GenerateInstantEvent(
    std::string name, int pid, int tid,
    std::chrono::system_clock::time_point timestamp,
    std::chrono::system_clock::time_point anchor) {
  std::string result = "{";
  result += "\"name\": \"" + name + "\", ";
  result += "\"ph\": \"i\", ";
  result +=
      "\"ts\": " +
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                         timestamp - anchor)
                         .count()) +
      ", ";
  result += "\"tid\": " + std::to_string(tid) + ", ";
  result += "\"pid\": " + std::to_string(pid);
  result += "}";
  return result;
}

std::string GenerateProcessMetaEvent(std::string name, std::string meta_name,
                                     int pid, int tid) {
  std::string result = "{";
  result += "\"name\": \"" + meta_name + "\", ";
  result += "\"ph\": \"M\", ";
  result += "\"pid\": " + std::to_string(pid) + ", ";
  result += "\"tid\": " + std::to_string(tid) + ", ";
  result += "\"args\": {";
  result += "\"name\": \"" + name + "\"";
  result += "}";
  result += "}";
  return result;
}

std::string GenerateBeginEvent(std::string name, int pid, int tid,
                               std::chrono::system_clock::time_point timestamp,
                               std::chrono::system_clock::time_point anchor) {
  std::string result = "{";
  result += "\"name\": \"" + name + "\", ";
  result += "\"ph\": \"B\", ";
  result +=
      "\"ts\": " +
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                         timestamp - anchor)
                         .count()) +
      ", ";
  result += "\"tid\": " + std::to_string(tid) + ", ";
  result += "\"pid\": " + std::to_string(pid);
  result += "}";
  return result;
}

std::string GenerateEndEvent(std::string name, int pid, int tid,
                             std::chrono::system_clock::time_point timestamp,
                             std::chrono::system_clock::time_point anchor,
                             std::string args) {
  std::string result = "{";
  result += "\"name\": \"" + name + "\", ";
  result += "\"ph\": \"E\", ";
  result +=
      "\"ts\": " +
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                         timestamp - anchor)
                         .count()) +
      ", ";
  result += "\"tid\": " + std::to_string(tid) + ", ";
  result += "\"pid\": " + std::to_string(pid);
  if (args != "") {
    result += ", \"args\": " + args;
  }
  result += "}";
  return result;
}

std::pair<std::string, std::string> GenerateDurationEvent(
    std::string name, int pid, int tid,
    std::pair<std::chrono::system_clock::time_point,
              std::chrono::system_clock::time_point>
        duration,
    std::chrono::system_clock::time_point anchor, std::string args) {
  return std::make_pair(
      GenerateBeginEvent(name, pid, tid, duration.first, anchor),
      GenerateEndEvent(name, pid, tid, duration.second, anchor, args));
}

}  // anonymous namespace

ChromeTracer::ChromeTracer()
    : anchor_(std::chrono::system_clock::now()),
      count_(0),
      pid_(GetNextPId()) {}

ChromeTracer::ChromeTracer(std::string name) : ChromeTracer() {
  this->name_ = name;
}

bool ChromeTracer::HasStream(std::string stream) {
  std::lock_guard<std::mutex> lock(lock_);
  if (event_table_.find(stream) == event_table_.end()) {
    return false;
  }
  return true;
}
void ChromeTracer::AddStream(std::string stream) {
  if (HasStream(stream)) {
    std::cerr << "The given stream already exists." << std::endl;
    abort();
  }

  std::lock_guard<std::mutex> lock(lock_);
  if (!event_table_.emplace(stream, std::map<int32_t, Event>()).second) {
    std::cerr << "Failed to add a stream.";
    abort();
  }
}

bool ChromeTracer::HasEvent(std::string stream, int32_t handle) {
  if (!HasStream(stream)) {
    std::cerr << "The given stream does not exists." << std::endl;
    abort();
  }

  std::lock_guard<std::mutex> lock(lock_);
  auto& events = event_table_[stream];
  if (events.find(handle) == events.end()) {
    return false;
  }
  return true;
}

void ChromeTracer::MarkEvent(std::string stream, std::string name) {
  std::lock_guard<std::mutex> lock(lock_);

  auto& events = event_table_[stream];
  if (!events.emplace(count_, Event(name, Event::EventStatus::Instantanous))
           .second) {
    std::cerr << "Failed to start an event." << std::endl;
    abort();
  }
  count_++;
}

int32_t ChromeTracer::BeginEvent(std::string stream, std::string name) {
  std::lock_guard<std::mutex> lock(lock_);

  auto& events = event_table_[stream];
  if (!events.emplace(count_, Event(name)).second) {
    std::cerr << "Failed to start an event." << std::endl;
    abort();
  }
  return count_++;
}

void ChromeTracer::EndEvent(std::string stream, int32_t handle,
                            std::string args) {
  if (!HasEvent(stream, handle)) {
    std::cerr << "The given event does not exists." << std::endl;
    abort();
  }

  std::lock_guard<std::mutex> lock(lock_);
  auto events = event_table_.find(stream)->second;
  auto new_event = events.find(handle)->second;
  new_event.args = args;
  new_event.Finish();
  events.erase(handle);
  events.emplace(handle, new_event);
  event_table_[stream] = events;
}

bool ChromeTracer::Validate() const {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto const& stream : event_table_) {
    for (auto const& events : stream.second) {
      if (events.second.GetStatus() == Event::EventStatus::Running) {
        std::cerr << stream.first << " " << events.second.name;
        return false;
      }
    }
  }
  return true;
}

// Returns the json string.
std::string ChromeTracer::Dump() const {
  if (!Validate()) {
    std::cerr << "There is unfinished event." << std::endl;
    abort();
  }

  std::lock_guard<std::mutex> lock(lock_);

  std::map<std::string, int> stream_tid_map;
  int i = 1;
  for (auto stream_name : event_table_) {
    stream_tid_map[stream_name.first] = i;
    i++;
  }

  std::string result = "{";
  std::string trace_events = "[";
  // 1. Start event per stream
  for (auto const& stream : event_table_) {
    std::string stream_name = stream.first;
    trace_events +=
        GenerateInstantEvent("Start", pid_, stream_tid_map[stream_name],
                             anchor_, anchor_) +
        ",";
  }

  // 2. Metadata event per process and thread (stream)
  trace_events +=
      GenerateProcessMetaEvent(name_, "process_name", pid_, 0) + ",";
  for (auto const& stream : event_table_) {
    std::string stream_name = stream.first;
    trace_events += GenerateProcessMetaEvent(stream_name, "thread_name", pid_,
                                             stream_tid_map[stream_name]) +
                    ",";
  }

  // 3. Duration event per events
  for (auto const& stream : event_table_) {
    for (auto const& event : stream.second) {
      switch (event.second.GetStatus()) {
        case Event::EventStatus::Finished: {
          auto dur_events = GenerateDurationEvent(
              event.second.name, pid_, stream_tid_map[stream.first],
              std::make_pair(event.second.start, event.second.end), anchor_,
              event.second.args);
          trace_events += dur_events.first + ",";
          trace_events += dur_events.second + ",";
        } break;
        case Event::EventStatus::Instantanous: {
          auto inst_event = GenerateInstantEvent(event.second.name, pid_,
                                                 stream_tid_map[stream.first],
                                                 event.second.start, anchor_);
          trace_events += inst_event + ",";
        } break;
        default: {
          std::cerr << "Invalid event state.";
          abort();
        }
      }
    }
  }
  trace_events = trace_events.substr(0, trace_events.size() - 1) + "]";

  result += "\"traceEvents\": " + trace_events;

  result += "}";

  return result;
}

// Dump the json string to the file path.
void ChromeTracer::Dump(std::string path) const {
  std::ofstream fout;
  fout.open(path);
  if (fout.is_open() == false) {
    std::cerr << "Cannot open the file path." << std::endl;
    abort();
  }
  fout << Dump();
  fout.close();
}

std::string ChromeTracer::Summary() const { return ""; }

void ChromeTracer::Clear() {
  for (auto& stream : event_table_) {
    stream.second.clear();
  }
  anchor_ = std::chrono::system_clock::now();
}

size_t ChromeTracer::GetNextPId() {
  static size_t pid = 0;
  return pid++;
}

}  // namespace chrome_tracer