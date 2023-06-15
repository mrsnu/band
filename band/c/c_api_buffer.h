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
// - Crop (x0, y0, x1, y1 - crop from top-left corner, inclusive)
// - Resize (width, height - resize to a new size)
// - Rotate (angle - counter-clockwise, between 0 and 360 in multiples of
// 90)
// - Flip (boolean - horizontal or vertical)
// - Convert color space (format - target format)
// - Normalize (mean, std - normalize the buffer with mean and std)
// - DataTypeConvert (convert the data type to the output data type)
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

BAND_CAPI_EXPORT extern void BandImageProcessorBuilderAddCrop(
    BandImageProcessorBuilder* builder, int x0, int y0, int x1, int y1);
BAND_CAPI_EXPORT extern void BandImageProcessorBuilderAddResize(
    BandImageProcessorBuilder* builder, int width, int height);
BAND_CAPI_EXPORT extern void BandImageProcessorBuilderAddRotate(
    BandImageProcessorBuilder* builder, int angle);
BAND_CAPI_EXPORT extern void BandImageProcessorBuilderAddFlip(
    BandImageProcessorBuilder* builder, bool horizontal, bool vertical);
BAND_CAPI_EXPORT extern void BandImageProcessorBuilderAddColorSpaceConvert(
    BandImageProcessorBuilder* builder, BandBufferFormat format);
BAND_CAPI_EXPORT extern void BandImageProcessorBuilderAddNormalize(
    BandImageProcessorBuilder* builder, float mean, float std);
BAND_CAPI_EXPORT extern void BandImageProcessorBuilderAddDataTypeConvert(
    BandImageProcessorBuilder* builder);

BAND_CAPI_EXPORT extern BandStatus BandImageProcessorProcess(
    BandImageProcessor* image_processor, BandBuffer* buffer,
    BandTensor* target_tensor);
BAND_CAPI_EXPORT extern void BandImageProcessorDelete(
    BandImageProcessor* processor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif