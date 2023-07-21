#include "band/tool/benchmark_graph.h"

#include <algorithm>
#include <queue>
#include <random>

#include "absl/strings/str_format.h"
#include "band/engine.h"
#include "band/model.h"
#include "band/tensor.h"
#include "band/tool/engine_runner.h"

namespace band {
namespace tool {

// motivated from /tensorflow/lite/tools/benchmark
template <typename T, typename Distribution>
void CreateRandomTensorData(void* target_ptr, int num_elements,
                            Distribution distribution) {
  std::mt19937 random_engine;
  T* target_head = static_cast<T*>(target_ptr);
  std::generate_n(target_head, num_elements, [&]() {
    return static_cast<T>(distribution(random_engine));
  });
}

BenchmarkGraph::Vertex::~Vertex() {
  auto delete_tensors = [](Tensors& tensors) {
    for (auto t : tensors) {
      delete t;
    }
  };

  for (auto request_inputs : model_request_inputs) {
    delete_tensors(request_inputs);
  }

  for (auto request_outputs : model_request_outputs) {
    delete_tensors(request_outputs);
  }

  delete_tensors(model_inputs);
}

absl::Status BenchmarkGraph::Vertex::PrepareInput() {
  for (int batch_index = 0; batch_index < model_request_inputs.size();
       batch_index++) {
    for (int input_index = 0; input_index < model_inputs.size();
         input_index++) {
      auto status =
          model_request_inputs[batch_index][input_index]->CopyDataFrom(
              model_inputs[input_index]);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

void BenchmarkGraph::Vertex::InitializeContext(Engine& engine) {
  const int model_id = model.GetId();
  const auto input_indices = engine.GetInputTensorIndices(model_id);
  const auto output_indices = engine.GetOutputTensorIndices(model_id);

  for (int i = 0; i < batch_size; i++) {
    // pre-allocate tensors
    Tensors inputs, outputs;
    for (int input_index : input_indices) {
      inputs.push_back(engine.CreateTensor(model_id, input_index));
    }

    for (int output_index : output_indices) {
      outputs.push_back(engine.CreateTensor(model_id, output_index));
    }

    model_request_inputs.push_back(inputs);
    model_request_outputs.push_back(outputs);
  }

  model_ids = std::vector<ModelId>(batch_size, model_id);
  request_options = std::vector<RequestOption>(batch_size, GetRequestOption());

  // pre-allocate random input tensor to feed in run-time requests
  for (int input_index : input_indices) {
    interface::ITensor* input_tensor =
        engine.CreateTensor(model_id, input_index);
    // random value ranges borrowed from tensorflow/lite/tools/benchmark
    switch (input_tensor->GetType()) {
      case DataType::kUInt8:
        CreateRandomTensorData<uint8_t>(
            input_tensor->GetData(), input_tensor->GetNumElements(),
            std::uniform_int_distribution<int32_t>(0, 254));
        break;
      case DataType::kInt8:
        CreateRandomTensorData<int8_t>(
            input_tensor->GetData(), input_tensor->GetNumElements(),
            std::uniform_int_distribution<int32_t>(-127, 127));
        break;
      case DataType::kInt16:
        CreateRandomTensorData<int16_t>(
            input_tensor->GetData(), input_tensor->GetNumElements(),
            std::uniform_int_distribution<int16_t>(0, 99));
        break;
      case DataType::kInt32:
        CreateRandomTensorData<int32_t>(
            input_tensor->GetData(), input_tensor->GetNumElements(),
            std::uniform_int_distribution<int32_t>(0, 99));
        break;
      case DataType::kInt64:
        CreateRandomTensorData<int64_t>(
            input_tensor->GetData(), input_tensor->GetNumElements(),
            std::uniform_int_distribution<int64_t>(0, 99));
        break;
      case DataType::kFloat32:
        CreateRandomTensorData<float>(
            input_tensor->GetData(), input_tensor->GetNumElements(),
            std::uniform_real_distribution<float>(-0.5, 0.5));
        break;
      case DataType::kFloat64:
        CreateRandomTensorData<double>(
            input_tensor->GetData(), input_tensor->GetNumElements(),
            std::uniform_real_distribution<double>(-0.5, 0.5));
        break;

      default:
        break;
    }
    model_inputs.push_back(input_tensor);
  }
}

const RequestOption BenchmarkGraph::Vertex::GetRequestOption() const {
  RequestOption option = RequestOption::GetDefaultOption();
  if (worker_id >= 0) {
    option.target_worker = worker_id;
  }
  return option;
}

BenchmarkGraph::~BenchmarkGraph() {
  for (auto vertex : vertices_) {
    delete vertex;
  }
}

absl::Status BenchmarkGraph::Initialize(const Json::Value& root,
                                        const EngineRunner& engine_runner) {
  for (auto vertex_key : root["vertices"].getMemberNames()) {
    if (!root["vertices"][vertex_key].isObject()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Please check if model config for model %s is "
                          "given",
                          vertex_key.c_str()));
    }

    if (!json::Validate(root["vertices"][vertex_key], {"name"})) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Please check if model name for vertex %s is "
                          "given",
                          vertex_key.c_str()));
    }

    const std::string model_key =
        root["vertices"][vertex_key]["name"].asString();
    auto model = engine_runner.GetModel(model_key);
    if (!model.ok()) {
      return model.status();
    }

    size_t batch_size = 1;
    int worker_id = -1;
    if (json::AssignIfValid(batch_size, root["vertices"][vertex_key],
                            "batch_size")) {
      if (batch_size <= 0) {
        return absl::InvalidArgumentError(
            "Please check if argument batch_size >= 0");
      }
    }

    if (json::AssignIfValid(worker_id, root["vertices"][vertex_key],
                            "worker_id")) {
      if ((worker_id < 0) ||
          (worker_id >= engine_runner.GetEngine().GetNumWorkers())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Please check if argument worker_id is valid (0 ~ %zu)",
            engine_runner.GetEngine().GetNumWorkers() - 1));
      }
    }

    size_t vertex_id = vertices_.size();
    vertex_names_.push_back(vertex_key);
    vertices_.push_back(
        new Vertex(model.value(), batch_size, worker_id, vertex_id));
  }

  for (auto edge : root["vertices"]["edges"]) {
    // make sure edge is two string array
    if (!edge.isArray() || edge.size() != 2) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Please check if edge %s is valid", edge.toStyledString().c_str()));
    }

    if (!edge[0].isString() || !edge[1].isString()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Please check if edge %s is valid", edge.toStyledString().c_str()));
    }

    const std::string from_vertex_key = edge[0].asString();
    const std::string to_vertex_key = edge[1].asString();

    // check if vertex exists
    if (std::find(vertex_names_.begin(), vertex_names_.end(),
                  from_vertex_key) == vertex_names_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Please check if vertex %s exists", from_vertex_key.c_str()));
    }

    if (std::find(vertex_names_.begin(), vertex_names_.end(), to_vertex_key) ==
        vertex_names_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Please check if vertex %s exists", to_vertex_key.c_str()));
    }

    // add edge
    size_t from_vertex_id =
        std::find(vertex_names_.begin(), vertex_names_.end(), from_vertex_key) -
        vertex_names_.begin();
    size_t to_vertex_id =
        std::find(vertex_names_.begin(), vertex_names_.end(), to_vertex_key) -
        vertex_names_.begin();

    edges_.emplace_back(std::make_pair(from_vertex_id, to_vertex_id));
  }

  return CheckCycles();
}

void BenchmarkGraph::InitializeExecutionContext() {
  std::lock_guard<std::mutex> lock(mutex_);
  finished_vertices_.clear();
}

std::vector<const BenchmarkGraph::Vertex*> BenchmarkGraph::GetNextVertices()
    const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<const Vertex*> vertices;
  std::set<size_t> resolved_vertex_ids = GetResolvedVertexIds();
  std::set<size_t> executable_vertex_ids;
  // executable_vertex_ids = resolved_vertex_ids - finished_vertices_
  std::set_difference(
      resolved_vertex_ids.begin(), resolved_vertex_ids.end(),
      finished_vertices_.begin(), finished_vertices_.end(),
      std::inserter(executable_vertex_ids, executable_vertex_ids.begin()));
  for (auto vertex_id : executable_vertex_ids) {
    vertices.push_back(vertices_[vertex_id]);
  }
  return vertices;
}

void BenchmarkGraph::OnVertexFinished(size_t vertex_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  finished_vertices_.insert(vertex_id);
}

bool BenchmarkGraph::IsFinished() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return finished_vertices_.size() == vertices_.size();
}

std::set<size_t> BenchmarkGraph::GetResolvedVertexIds() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::set<size_t> resolved_vertices;
  // A vertex is resolved if all its incoming edges are resolved
  for (size_t i = 0; i < vertices_.size(); ++i) {
    bool is_resolved = true;
    for (size_t j = 0; j < edges_.size(); ++j) {
      if (edges_[j].second == i && finished_vertices_.find(edges_[j].first) ==
                                       finished_vertices_.end()) {
        is_resolved = false;
        break;
      }
    }
    if (is_resolved) {
      resolved_vertices.insert(i);
    }
  }
  return resolved_vertices;
}

absl::Status BenchmarkGraph::CheckCycles() const {
  std::vector<size_t> num_incoming_edges(vertices_.size(), 0);
  std::queue<size_t> queue;
  size_t num_visited_vertices = 0;
  // Initialize num_incoming_edges per vertex
  for (size_t i = 0; i < edges_.size(); ++i) {
    ++num_incoming_edges[edges_[i].second];
  }
  // Start with vertices with no incoming edges
  for (size_t i = 0; i < num_incoming_edges.size(); ++i) {
    if (num_incoming_edges[i] == 0) {
      queue.push(i);
    }
  }
  // BFS
  while (!queue.empty()) {
    size_t vertex_id = queue.front();
    queue.pop();
    ++num_visited_vertices;

    for (size_t i = 0; i < edges_.size(); ++i) {
      if (edges_[i].first == vertex_id) {
        --num_incoming_edges[edges_[i].second];
        if (num_incoming_edges[edges_[i].second] == 0) {
          queue.push(edges_[i].second);
        }
      }
    }
  }

  return num_visited_vertices == vertices_.size()
             ? absl::OkStatus()
             : absl::InternalError("Cycles detected");
}
}  // namespace tool
}  // namespace band