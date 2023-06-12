#ifndef BAND_BUFFER_IMAGE_OPERATION_H_
#define BAND_BUFFER_IMAGE_OPERATION_H_

#include "absl/status/status.h"
#include "band/buffer/buffer.h"
#include "band/buffer/common_operator.h"
#include "band/buffer/operator.h"

namespace band {
namespace buffer {

class Crop : public IBufferOperator {
 public:
  Crop(int x0, int y0, int x1, int y1) : x0_(x0), y0_(y0), x1_(x1), y1_(y1) {}

  virtual IBufferOperator* Clone() const override;
  virtual Type GetOpType() const override;

  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

 private:
  int x0_, y0_, x1_, y1_;
};

class Resize : public IBufferOperator {
 public:
  // width: -1 for auto, height: -1 for auto
  Resize(int width = -1, int height = -1) : dims_({width, height}) {}
  Resize(const std::vector<int>& dims) : dims_(dims) {}

  virtual IBufferOperator* Clone() const override;
  virtual Type GetOpType() const override;

  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

 private:
  bool IsAuto(size_t dim) const { return dims_[dim] == -1; }
  std::vector<int> dims_;
};

// Rotate rotates the input buffer by the specified angle in degrees
// (counter-clockwise). The output buffer will be created automatically.
class Rotate : public IBufferOperator {
 public:
  Rotate(int angle_deg) : angle_deg_(angle_deg) {}

  virtual IBufferOperator* Clone() const override;
  virtual Type GetOpType() const override;

  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

 private:
  int angle_deg_;
};

class Flip : public IBufferOperator {
 public:
  // horizontal: true for horizontal flip, false for vertical flip
  Flip(bool horizontal) : horizontal_(horizontal) {}

  virtual IBufferOperator* Clone() const override;
  virtual Type GetOpType() const override;

  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

 private:
  bool horizontal_;
};

class ColorSpaceConvert : public IBufferOperator {
 public:
  ColorSpaceConvert()
      : output_format_(BufferFormat::kRaw), is_format_specified_(false) {}
  ColorSpaceConvert(BufferFormat buffer_format)
      : output_format_(buffer_format), is_format_specified_(true) {}

  virtual IBufferOperator* Clone() const override;
  virtual Type GetOpType() const override;

  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

 private:
  BufferFormat output_format_;
  bool is_format_specified_;
};

// AutoConvert automatically converts the internal data type to output data
// type regardless of the dimension and format of the input buffer.
// It is equivalent to ColorSpaceConvert() + Resize(-1, -1) + DataTypeConvert()

// TODO(dostos): this operator could be removed if we can support automatic
// propagation of parameters across operators. Currently, a last operator
// is the only once that can infer the parameters from the output buffer.
// This operator requires the propagation of the target color space and the
// target dimension from bottom to top.
class AutoConvert : public IBufferOperator {
 public:
  virtual ~AutoConvert() override;
  virtual IBufferOperator* Clone() const override;
  virtual Type GetOpType() const override;
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;
  virtual void SetOutput(Buffer* output) override;

 private:
  bool RequiresColorSpaceConvert(const Buffer& input) const;
  bool RequiresResize(const Buffer& input) const;
  bool RequiresDataTypeConvert(const Buffer& input) const;

  ColorSpaceConvert color_space_convert_;
  Resize resize_;
  DataTypeConvert data_type_convert_;
};

}  // namespace buffer
}  // namespace band

#endif  // BAND_BUFFER_IMAGE_OPERATION_H_