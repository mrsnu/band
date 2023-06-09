#ifndef BAND_TENSOR_BUFFER_PROCESSOR_H
#define BAND_TENSOR_BUFFER_PROCESSOR_H

namespace band {
namespace tensor {

class IOperation;


class BufferProcessorBuilder {
 public:
  BufferProcessorBuilder() = default;
  ~BufferProcessorBuilder() = default;
  BufferProcessorBuilder(const BufferProcessorBuilder&) = delete;
  BufferProcessorBuilder& operator=(const BufferProcessorBuilder&) = delete;

  BufferProcessorBuilder& AddOperation(const IOperation& operation);

  BufferProcessor Build();

 private:
  std::vector<const IOperation*> operations_;
};

class BufferProcessor {
 public:

  static 


 private:
  BufferProcessor() = default;
  ~BufferProcessor() = default;
  BufferProcessor(const BufferProcessor&) = delete;
  BufferProcessor& operator=(const BufferProcessor&) = delete;



};

}  // namespace tensor
}  // namespace band

#endif  // BAND_TENSOR_BUFFER_PROCESSOR_H