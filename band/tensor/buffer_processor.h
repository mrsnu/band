#ifndef BAND_TENSOR_BUFFER_PROCESSOR_H
#define BAND_TENSOR_BUFFER_PROCESSOR_H

namespace band {
namespace tensor {

class IOperation;
class BufferProcessor {
 public:
 private:
  BufferProcessor() = default;
  ~BufferProcessor() = default;
  BufferProcessor(const BufferProcessor&) = delete;
  BufferProcessor& operator=(const BufferProcessor&) = delete;
};

}  // namespace tensor
}  // namespace band

#endif  // BAND_TENSOR_BUFFER_PROCESSOR_H