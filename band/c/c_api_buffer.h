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
typedef struct BandBuffer;

// A preprocessing pipeline for BandBuffer. This is used to apply a series of
// preprocessing operations to a BandBuffer then convert it to a BandTensor.
// Supported operations are:
// - Crop (x0, y0, x1, y1 - crop from top-left corner, inclusive)
// - Resize (width, height - resize to a new size, if -1 is given, the operation
// automatically calculates the new size based on the output tensor or buffer
// shape)
// - Rotate (angle - counter-clockwise, between 0 and 360 in multiples of 90)
// - Flip (boolean - horizontal or vertical)
// - Convert color space
typedef struct BandBufferProcessor;
// A builder for BandBufferProcessor. This is used to build a
// BandBufferProcessor. By default, the BandBufferProcessorBuilder will create a
// BandBufferProcessor with single Resize
typedef struct BandBufferProcessorBuilder;

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

BAND_CAPI_EXPORT extern BandBufferProcessorBuilder*
BandBufferProcessorBuilderCreate();
BAND_CAPI_EXPORT extern void BandBufferProcessorBuilderDelete(
    BandBufferProcessorBuilder* builder);
BAND_CAPI_EXPORT extern void BandBufferProcessorBuilderAddCrop(
    BandBufferProcessorBuilder* builder, int x0, int y0, int x1, int y1);
BAND_CAPI_EXPORT extern void BandBufferProcessorBuilderAddResize(
    BandBufferProcessorBuilder* builder, int width, int height);
BAND_CAPI_EXPORT extern void BandBufferProcessorBuilderAddRotate(
    BandBufferProcessorBuilder* builder, int angle);
BAND_CAPI_EXPORT extern void BandBufferProcessorBuilderAddFlip(
    BandBufferProcessorBuilder* builder, bool horizontal);
BAND_CAPI_EXPORT extern void BandBufferProcessorBuilderAddConvert(
    BandBufferProcessorBuilder* builder, BandBufferFormat format);
BAND_CAPI_EXPORT extern BandBufferProcessor* BandBufferProcessorBuilderBuild(
    BandBufferProcessorBuilder* builder);

BAND_CAPI_EXPORT extern BandBufferProcessor* BandBufferProcessorCreate();
BAND_CAPI_EXPORT extern void BandBufferProcessorDelete(
    BandBufferProcessor* processor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif