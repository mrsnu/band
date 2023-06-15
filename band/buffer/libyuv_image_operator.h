/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
Heavily modified from the original source code:
tflite-support/tensorflow_lite_support/cc/task/vision/utils
/libyuv_frame_buffer_utils.h
by Jingyu Lee <dostos10@gmail.com>
*/

#ifndef BAND_BUFFER_LIBYUV_IMAGE_OPERATOR_H_
#define BAND_BUFFER_LIBYUV_IMAGE_OPERATOR_H_

#include "absl/status/status.h"
#include "band/buffer/buffer.h"

namespace band {

class LibyuvImageOperator {
 public:
  // Crops input `buffer` to the specified subregions and resizes the cropped
  // region to the target image resolution defined by the `output_buffer`.
  //
  // (x0, y0) represents the top-left point of the buffer.
  // (x1, y1) represents the bottom-right point of the buffer.
  //
  // Crop region dimensions must be equal or smaller than input `buffer`
  // dimensions.
  static absl::Status Crop(const Buffer& buffer, int x0, int y0, int x1, int y1,
                           Buffer& output_buffer);

  // Resizes `buffer` to the size of the given `output_buffer`.
  static absl::Status Resize(const Buffer& buffer, Buffer& output_buffer);

  // Rotates `buffer` counter-clockwise by the given `angle_deg` (in degrees).
  //
  // The given angle must be a multiple of 90 degrees.
  static absl::Status Rotate(const Buffer& buffer, int angle_deg,
                             Buffer& output_buffer);

  // Flips `buffer` horizontally.
  static absl::Status FlipHorizontally(const Buffer& buffer,
                                       Buffer& output_buffer);

  // Flips `buffer` vertically.
  static absl::Status FlipVertically(const Buffer& buffer,
                                     Buffer& output_buffer);

  // Converts `buffer`'s format to the format of the given `output_buffer`.
  //
  // Grayscale format cannot be converted to other formats.
  static absl::Status ColorSpaceConvert(const Buffer& buffer,
                                        Buffer& output_buffer);
};
}  // namespace band

#endif  // BAND_BUFFER_LIBYUV_IMAGE_OPERATOR_H_
