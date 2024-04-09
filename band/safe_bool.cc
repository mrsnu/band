#include "band/safe_bool.h"

/**
 * @brief The `band` namespace contains classes and functions related to the band module.
 */
namespace band {
/**
 * @brief Notifies the waiting thread that a condition has occurred.
 * 
 * This function is used to notify the waiting thread that a condition has occurred.
 * It sets the flag to true and notifies one waiting thread using the condition variable.
 */
void SafeBool::notify() {
  std::lock_guard<std::mutex> lock(m);
  flag = true;
  c.notify_one();
}

/**
 * @brief Waits until a condition is met or the termination flag is set.
 * 
 * This function waits until either the exit flag is set to true or the condition flag is set to true.
 * It uses a condition variable to wait for the condition to be met.
 * The function will return the value of the exit flag after the wait.
 * 
 * @return The value of the exit flag.
 */
bool SafeBool::wait() {
  std::unique_lock<std::mutex> lock(m);
  while (!exit && !flag) {
    c.wait(lock);
  }
  flag = false;
  return exit;
}

/**
 * @brief Terminates the waiting thread.
 * 
 * This function is used to terminate the waiting thread.
 * It sets the exit flag to true and notifies all waiting threads using the condition variable.
 */
void SafeBool::terminate() {
  std::lock_guard<std::mutex> lock(m);
  exit = true;
  c.notify_all();
}
}  // namespace band