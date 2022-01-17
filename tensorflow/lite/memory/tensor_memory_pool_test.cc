#include "tensorflow/lite/memory/tensor_memory_pool.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

TEST(TensorMemoryPoolTest, BasicPoolOperations) {
  ErrorReporter* reporter = DefaultErrorReporter();
  TensorMemoryPool pool(reporter);

  pool.Allocate(1024, 0);
  pool.Allocate(1024, 1);
  pool.Allocate(1024, 2);
  pool.Allocate(1024, 3);
  pool.Allocate(1024, 4);

  EXPECT_EQ(pool.GetHead(), 1024 * 5);
}

TEST(TensorMemoryPoolTest, PoolResize) {
  ErrorReporter* reporter = DefaultErrorReporter();
  TensorMemoryPool pool(reporter, 1 << 12);

  pool.Allocate(1024, 0);
  pool.Allocate(1024, 1);
  pool.Allocate(1024, 2);
  pool.Allocate(1024, 3);
  EXPECT_EQ(pool.GetBufferSize(), 1 << 12);
  pool.Allocate(1024, 4);
  EXPECT_EQ(pool.GetBufferSize(), 1 << 13);
}

TEST(TensorMemoryPoolTest, PoolDeallocate) {
  ErrorReporter* reporter = DefaultErrorReporter();
  TensorMemoryPool pool(reporter);

  pool.Allocate(1024, 0);
  pool.Allocate(512, 1);
  pool.Allocate(512, 2);
  pool.Allocate(256, 3);
  EXPECT_EQ(pool.GetHead(), 2048 + 256);
  EXPECT_EQ(pool.Deallocate(2), kTfLiteOk);
  // |1024|512|x(512)|256|
  EXPECT_EQ(pool.GetHead(), 2048 + 256);
  EXPECT_EQ(pool.Deallocate(3), kTfLiteOk);
  // |1024|512|
  EXPECT_EQ(pool.GetHead(), 1024 + 512);
  EXPECT_EQ(pool.Deallocate(0), kTfLiteOk);
  EXPECT_EQ(pool.Deallocate(1), kTfLiteOk);
  // |
  EXPECT_EQ(pool.GetHead(), 0);
  EXPECT_EQ(pool.Deallocate(1), kTfLiteError);
}

TEST(TensorMemoryPoolTest, PoolCaching) {
  ErrorReporter* reporter = DefaultErrorReporter();
  TensorMemoryPool pool(reporter);

  pool.Allocate(1024, 0);
  pool.Allocate(512, 1);
  pool.Allocate(512, 2);
  pool.Allocate(256, 3);
  EXPECT_EQ(pool.GetHead(), 2048 + 256);
  EXPECT_EQ(pool.Deallocate(2), kTfLiteOk);
  // |1024|512|x(512)|256|
  pool.Allocate(256, 4);
  // |1024|512|x(512)|256|256|
  EXPECT_EQ(pool.GetOffset(4), 2048 + 256);
  EXPECT_EQ(pool.GetHead(), 2048 + 256 + 256);
  pool.Allocate(512, 5);
  // |1024|512|512|256|256|
  EXPECT_EQ(pool.GetHead(), 2048 + 256 + 256);
  EXPECT_EQ(pool.GetOffset(5), 1024 + 512);
}

class PoolTensorCopyTest
    : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(PoolTensorCopyTest, PoolTensorCopy) {
  const int num_tensors = std::get<0>(GetParam());
  const int offset_0 = std::get<1>(GetParam());
  const int offset_1 = std::get<2>(GetParam());

  ErrorReporter* reporter = DefaultErrorReporter();
  TensorMemoryPool pool(reporter);

  EXPECT_EQ(pool.Allocate(1 << 14, 0), kTfLiteOk);
  EXPECT_EQ(pool.Allocate(1 << 14, 1), kTfLiteOk);

  std::vector<TfLiteTensor*> tensors_0;
  std::vector<TfLiteTensor*> out_tensors_0;

  for (int i = 0; i < num_tensors; i++) {
    TfLiteTensor* tensor = TfLiteTensorCreate();
    tensor->allocation_type = kTfLiteDynamic;
    TfLiteTensorRealloc(offset_0 + i * 32, tensor);
    memset(tensor->data.raw, 48 + i, offset_0 + i * 32);
    tensors_0.push_back(tensor);
  }

  for (int i = 0; i < num_tensors; i++) {
    TfLiteTensor* tensor = TfLiteTensorCreate();
    tensor->allocation_type = kTfLiteDynamic;
    TfLiteTensorRealloc(offset_0 + i * 32, tensor);
    out_tensors_0.push_back(tensor);
  }

  std::vector<TfLiteTensor*> tensors_1;
  std::vector<TfLiteTensor*> out_tensors_1;

  for (int i = 0; i < num_tensors; i++) {
    TfLiteTensor* tensor = TfLiteTensorCreate();
    tensor->allocation_type = kTfLiteDynamic;
    TfLiteTensorRealloc(offset_1 + i * 64, tensor);
    memset(tensor->data.raw, 48 + i, offset_1 + i * 64);
    tensors_1.push_back(tensor);
  }

  for (int i = 0; i < num_tensors; i++) {
    TfLiteTensor* tensor = TfLiteTensorCreate();
    tensor->allocation_type = kTfLiteDynamic;
    TfLiteTensorRealloc(offset_1 + i * 64, tensor);
    out_tensors_1.push_back(tensor);
  }

  for (int i = 0; i < num_tensors; i++) {
    EXPECT_EQ(pool.PutTensorToHandle(tensors_0[i], 0, i), kTfLiteOk);
    EXPECT_EQ(pool.PutTensorToHandle(tensors_1[i], 1, i), kTfLiteOk);
  }


  for (int i = 0; i < num_tensors; i++) {
    EXPECT_EQ(pool.GetTensorFromHandle(out_tensors_0[i], 0, i), kTfLiteOk);
    EXPECT_EQ(pool.GetTensorFromHandle(out_tensors_1[i], 1, i), kTfLiteOk);

    EXPECT_EQ(memcmp(tensors_0[i]->data.raw, out_tensors_0[i]->data.raw,
                     offset_0 + i * 32),
              0);
    EXPECT_EQ(memcmp(tensors_1[i]->data.raw, out_tensors_1[i]->data.raw,
                     offset_1 + i * 64),
              0);
  }

  for (TfLiteTensor* tensor : tensors_0) {
    TfLiteTensorDelete(tensor);
  }

  for (TfLiteTensor* tensor : tensors_1) {
    TfLiteTensorDelete(tensor);
  }

  for (TfLiteTensor* tensor : out_tensors_0) {
    TfLiteTensorDelete(tensor);
  }

  for (TfLiteTensor* tensor : out_tensors_1) {
    TfLiteTensorDelete(tensor);
  }
}

INSTANTIATE_TEST_SUITE_P(TensorMemoryPoolTest, PoolTensorCopyTest,
                         ::testing::Combine(  
                             // number of tensors
                             ::testing::Values(1, 3, 5),
                             // offset for tensors 1
                             ::testing::Values(5, 32, 64, 67),
                             // offset for tensors 2
                             ::testing::Values(2, 32, 64, 67)));

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
 