#include "tensorflow/lite/free_tree_allocator.h"

#include <gtest/gtest.h>

#include <cstddef>  // max_align_t
#include <random>
#include <string>
#include <thread>
#include <unordered_map>

namespace tflite {

size_t calculate_offset(void* base, void* offset) {
  return (char*)offset - (char*)base;
}

static unsigned int SelectRandomly(unsigned int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<std::mt19937::result_type> dis(0, max);
  return dis(gen);
}

static void FillRandomBytes(
    std::unordered_map<void*, std::vector<unsigned char>>& addresses,
    void* address, std::size_t count) {
  if (address == nullptr) return;
  addresses[address] = std::vector<unsigned char>(count);
  for (unsigned int i = 0; i < count; ++i) {
    unsigned char byte = (unsigned char)SelectRandomly(255);
    addresses[address][i] = byte;
    *(reinterpret_cast<char*>(address) + i) = byte;
  }
}

static void RandomAllocate(FreeTreeAllocator* alloc, size_t num_iteration) {
  std::vector<std::size_t> randomSizes;
  randomSizes.reserve(num_iteration);
  for (std::size_t i = 0; i < randomSizes.capacity(); ++i) {
    randomSizes.push_back(SelectRandomly(512) + 1);
  }

  std::unordered_map<void*, std::vector<unsigned char>> addresses;
  addresses.reserve(num_iteration);

  for (auto s : randomSizes) {
    void* address = alloc->Allocate(s);
    FillRandomBytes(addresses, address, s);
  }
  for (auto a : addresses) {
    void* address = a.first;
    for (unsigned int i = 0; i < a.second.size(); ++i) {
      ASSERT_EQ(a.second[i], *(reinterpret_cast<unsigned char*>(address) + i));
    }
  }
}

static void RandomAllocateDeallocate(FreeTreeAllocator* alloc, size_t num_iteration) {
  std::vector<std::size_t> randomSizes;
  randomSizes.reserve(num_iteration);
  for (std::size_t i = 0; i < randomSizes.capacity(); ++i) {
    randomSizes.push_back(SelectRandomly(512) + 1);
  }

  std::unordered_map<void*, std::vector<unsigned char>> addresses;
  addresses.reserve(num_iteration);

  std::vector<void*> addressesToDeallocate;
  for (auto s : randomSizes) {
    void* address = alloc->Allocate(s);
    FillRandomBytes(addresses, address, s);
    if (SelectRandomly(100) <= 50) addressesToDeallocate.push_back(address);
  }
  for (auto a : addressesToDeallocate) {
    alloc->Deallocate(a);
    addresses.erase(a);
  }
  for (unsigned int i = 0; i < addressesToDeallocate.size(); ++i) {
    void* address = alloc->Allocate(randomSizes[i]);
    FillRandomBytes(addresses, address, randomSizes[i]);
  }
  for (auto a : addresses) {
    void* address = a.first;
    for (unsigned int i = 0; i < a.second.size(); ++i) {
      ASSERT_EQ(a.second[i], *(reinterpret_cast<unsigned char*>(address) + i));
    }
  }
}

static void RandomAllocateDeallocateLinearly(FreeTreeAllocator* alloc, size_t num_iteration) {
  std::vector<std::size_t> randomSizes;
  randomSizes.reserve(num_iteration);
  for (unsigned int i = 0; i < randomSizes.capacity(); ++i) {
    randomSizes.push_back(SelectRandomly(512) + 1);
  }

  std::unordered_map<void*, std::vector<unsigned char>> addresses;
  addresses.reserve(num_iteration);

  std::vector<void*> addressesToDeallocate;
  addressesToDeallocate.reserve(SelectRandomly(100));
  for (unsigned int i = 0; i < randomSizes.size(); ++i) {
    void* address = alloc->Allocate(randomSizes[i]);
    FillRandomBytes(addresses, address, randomSizes[i]);
    if (i >= randomSizes.size() - addressesToDeallocate.capacity()) {
      addressesToDeallocate.push_back(address);
    }
  }
  for (auto a : addressesToDeallocate) {
    alloc->Deallocate(a);
    addresses.erase(a);
  }
  for (unsigned int i = 0; i < addressesToDeallocate.size(); ++i) {
    void* address = alloc->Allocate(randomSizes[i]);
    FillRandomBytes(addresses, address, randomSizes[i]);
  }
  for (auto a : addresses) {
    void* address = a.first;
    for (int i = 0; i < a.second.size(); ++i) {
      ASSERT_EQ(a.second[i], *(reinterpret_cast<unsigned char*>(address) + i));
    }
  }
}

static void FixedAllocateDeallocate(FreeTreeAllocator* alloc,
                                    const std::size_t size, 
                                    size_t num_iteration) {
  std::unordered_map<void*, std::vector<unsigned char>> addresses;
  addresses.reserve(num_iteration);

  std::vector<void*> addressesToDeallocate;
  for (unsigned int i = 0; i < num_iteration; ++i) {
    void* address = alloc->Allocate(size);
    FillRandomBytes(addresses, address, size);
    if (SelectRandomly(100) <= 50) addressesToDeallocate.push_back(address);
  }
  for (auto a : addressesToDeallocate) {
    alloc->Deallocate(a);
    addresses.erase(a);
  }
  for (unsigned int i = 0; i < addressesToDeallocate.size(); ++i) {
    void* address = alloc->Allocate(size);
    FillRandomBytes(addresses, address, size);
  }
  for (auto a : addresses) {
    void* address = a.first;
    for (unsigned int i = 0; i < a.second.size(); ++i) {
      ASSERT_EQ(a.second[i], *(reinterpret_cast<unsigned char*>(address) + i));
    }
  }
}
constexpr std::size_t mb = 1024 * 1024;

TEST(FreeTreeAllocator, RandomAllocate) {
  FreeTreeAllocator allocator(mb);
  RandomAllocate(&allocator, 1000);
}

TEST(FreeTreeAllocator, RandomAllocateMultiThreaded) {
  FreeTreeAllocator allocator(mb);

  int num_threads = 5;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread([&]() { RandomAllocate(&allocator, 100); }));
  }

  for (std::thread& t : threads) {
    t.join();
  }
}

TEST(FreeTreeAllocator, RandomAllocateDeallocate) {
  FreeTreeAllocator allocator(mb);
  RandomAllocateDeallocate(&allocator, 1000);
}

TEST(FreeTreeAllocator, RandomAllocateDeallocateMultiThreaded) {
  FreeTreeAllocator allocator(mb);

  int num_threads = 5;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread([&]() { RandomAllocateDeallocate(&allocator, 100); }));
  }

  for (std::thread& t : threads) {
    t.join();
  }
}

TEST(FreeTreeAllocator, RandomAllocateDeallocateLinearly) {
  FreeTreeAllocator allocator(mb);
  RandomAllocateDeallocateLinearly(&allocator, 1000);
}

TEST(FreeTreeAllocator, RandomAllocateDeallocateLinearlyMultiThreaded) {
  FreeTreeAllocator allocator(mb);

  int num_threads = 5;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread([&]() { RandomAllocateDeallocateLinearly(&allocator, 100); }));
  }

  for (std::thread& t : threads) {
    t.join();
  }
}

TEST(FreeTreeAllocator, FixedAllocateDeallocate) {
  FreeTreeAllocator allocator(mb);
  FixedAllocateDeallocate(&allocator, 512, 1000);
}

TEST(FreeTreeAllocator, FixedAllocateDeallocateMultiThreaded) {
  FreeTreeAllocator allocator(mb);

  int num_threads = 5;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread([&]() { FixedAllocateDeallocate(&allocator, 512, 100); }));
  }

  for (std::thread& t : threads) {
    t.join();
  }
}
}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}