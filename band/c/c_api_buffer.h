/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_C_C_API_BUFFER_H_
#define BAND_C_C_API_BUFFER_H_

#include "c_api.h"
#include "c_api_type.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An external buffer interface for Band. This is used to pass user-owned
// buffers to Band. Band will not take ownership of the buffer, and the buffer
// must outlive the BandBuffer object.
typedef struct BandBuffer BandBuffer;

// ImageProcessorBuilder is used to build an ImageProcessor. ImageProcessor
// defines a series of operations to be applied to a BandBuffer and convert
// it to a BandTensor. Supported operations are:
// - Crop (int x0, int y0, int x1, int y1) - crop from top-left corner,
// inclusive
// - Resize (int width, int height) - resize to a new size
// - Rotate (float angle) - counter-clockwise, between 0 and 360 in multiples of
// 90
// - Flip (bool horizontal, bool vertical)
// - Convert color space (BandBufferFormat target_format) - convert the color
// space
// - Normalize (float mean, float std)
// - DataTypeConvert () convert the data type to the output data type
// E.g., convert from 8-bit RGB to 32-bit float RGB (tensor).
//
// By default, builder without any operation will create a ImageProcessor
// provides a direct mapping from BandBuffer to BandTensor without
// normalization. E.g., automated color space conversion, resize to the
// output tensor shape, and data type conversion.

typedef struct BandImageProcessor BandImageProcessor;
typedef struct BandImageProcessorBuilder BandImageProcessorBuilder;

BAND_CAPI_EXPORT extern BandBuffer* BandBufferCreate();
BAND_CAPI_EXPORT extern void BandBufferDelete(BandBuffer* buffer);

// Set buffer from raw image data. Supported formats are:
// - RGB (3 channels - 8 bits per channel, interleaved)
// - RGBA (4 channels - 8 bits per channel, interleaved)
// - GRAY (1 channel - 8 bits per channel)
// - NV21 (YUV 4:2:0 - 8 bits per channel, interleaved)
// - NV12 (YUV 4:2:0 - 8 bits per channel, interleaved)
// - YV12 (YUV 4:2:0 - 8 bits per channel, planar)
// - YV21 (YUV 4:2:0 - 8 bits per channel, planar)
BAND_CAPI_EXPORT extern BandStatus BandBufferSetFromRawData(
    BandBuffer* buffer, const void* data, size_t width, size_t height,
    BandBufferFormat format);

// Set buffer from YUV data. Supported formats are:
// - NV21 (YUV 4:2:0 - 8 bits per channel, interleaved)
// - NV12 (YUV 4:2:0 - 8 bits per channel, interleaved)
// - YV12 (YUV 4:2:0 - 8 bits per channel, planar)
// - YV21 (YUV 4:2:0 - 8 bits per channel, planar)
BAND_CAPI_EXPORT extern BandStatus BandBufferSetFromYUVData(
    BandBuffer* buffer, const void* y_data, const void* u_data,
    const void* v_data, size_t width, size_t height, size_t row_stride_y,
    size_t row_stride_uv, size_t pixel_stride_uv,
    BandBufferFormat buffer_format);

BAND_CAPI_EXPORT extern BandImageProcessorBuilder*
BandImageProcessorBuilderCreate();
BAND_CAPI_EXPORT extern void BandImageProcessorBuilderDelete(
    BandImageProcessorBuilder* builder);
BAND_CAPI_EXPORT extern BandImageProcessor* BandImageProcessorBuilderBuild(
    BandImageProcessorBuilder* builder);

// Add an operator to the builder. The order of the operators will be the
// order of the operations applied to the input buffer.
// E.g., BandAddOperator(builder, BAND_IMAGE_PROCESSOR_CROP, 4, 0, 0, 100, 100);
// will crop the input buffer from (0, 0) to (100, 100).
// This will return kBandErr if the given variadic arguments are invalid.
BAND_CAPI_EXPORT extern BandStatus BandAddOperator(
    BandImageProcessorBuilder* b, BandImageProcessorBuilderField field,
    int count, ...);

BAND_CAPI_EXPORT extern BandStatus BandImageProcessorProcess(
    BandImageProcessor* image_processor, BandBuffer* buffer,
    BandTensor* target_tensor);
BAND_CAPI_EXPORT extern void BandImageProcessorDelete(
    BandImageProcessor* processor);

typedef BandBuffer* (*PFN_BandBufferCreate)();
typedef void (*PFN_BandBufferDelete)(BandBuffer*);
typedef BandStatus (*PFN_BandBufferSetFromRawData)(BandBuffer*, const void*,
                                                   size_t, size_t,
                                                   BandBufferFormat);
typedef BandStatus (*PFN_BandBufferSetFromYUVData)(BandBuffer*, const void*,
                                                   const void*, const void*,
                                                   size_t, size_t, size_t,
                                                   size_t, size_t,
                                                   BandBufferFormat);
typedef BandImageProcessorBuilder* (*PFN_BandImageProcessorBuilderCreate)();
typedef void (*PFN_BandImageProcessorBuilderDelete)(BandImageProcessorBuilder*);
typedef BandImageProcessor* (*PFN_BandImageProcessorBuilderBuild)(
    BandImageProcessorBuilder*);
typedef BandStatus (*PFN_BandAddOperator)(BandImageProcessorBuilder*,
                                          BandImageProcessorBuilderField, int,
                                          ...);
typedef BandStatus (*PFN_BandImageProcessorProcess)(BandImageProcessor*,
                                                    BandBuffer*, BandTensor*);
typedef void (*PFN_BandImageProcessorDelete)(BandImageProcessor*);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif