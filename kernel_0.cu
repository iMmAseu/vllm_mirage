#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace mirage::runtime;
size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

void construct_task_graph(int num_gpus,
                          int my_gpu_id,
                          std::vector<FullTaskDesc> &all_tasks,
                          std::vector<EventDesc> &all_events,
                          std::vector<TaskId> &first_tasks,
                          std::map<std::string, void*> const &all_tensors) {
  std::filesystem::path file_path(__FILE__);
  std::ifstream json_file(file_path.parent_path().string()+"/task_graph.json");
  nlohmann::json json_task_graph;
  json_file >> json_task_graph;
  for (json const &task : json_task_graph["all_tasks"]) {
    FullTaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    task_desc.request_id = task.at("request_id").get<int>();
    if (task.at("trigger_event").is_number_integer()) {
      task_desc.trigger_event = task.at("trigger_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    task_desc.num_inputs = 0;
    for (json const &tensor : task["inputs"]) {
      TensorDesc input;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      input.num_dims = tensor.at("dims").size();
      input.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < input.num_dims; i++) {
        input.dim[i] = tensor["dims"][i].get<int>();
        input.stride[i] = tensor["strides"][i].get<int>();
      }
      task_desc.inputs[task_desc.num_inputs++] = input;
    }
    task_desc.num_outputs = 0;
    for (json const &tensor : task["outputs"]) {
      TensorDesc output;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      output.num_dims = tensor.at("dims").size();
      output.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < output.num_dims; i++) {
        output.dim[i] = tensor["dims"][i];
        output.stride[i] = tensor["strides"][i];
      }
      task_desc.outputs[task_desc.num_outputs++] = output;
    }
    #ifdef MPK_ENABLE_TMA
    if (task.at("task_type") > TASK_HOPPER_TASK_BEGIN && task.at("task_type") < TASK_HOPPER_TASK_END) {
      create_tma_desc_by_task(task_desc);
    }
    if (task.at("task_type") > TASK_SM100_TASK_BEGIN && task.at("task_type") < TASK_ATTN_SM100) {
      create_tma_desc_by_task(task_desc);
    }
    #endif
    all_tasks.push_back(task_desc);
  }
  for (json const &e : json_task_graph["all_events"]) {
    EventType event_type = static_cast<EventType>(e.at("event_type").get<int>());
    int num_triggers = e.at("num_triggers").get<int>();
    int first_task_id = e.at("first_task_id").get<int>();
    int last_task_id = e.at("last_task_id").get<int>();
    all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));
  }
  for (json const &t : json_task_graph["first_tasks"]) {
    first_tasks.push_back(t.get<int>());
  }
}

static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                  std::vector<TaskId> &first_tasks,
                                  int num_gpus,
                                  int my_gpu_id) {
  assert(num_gpus = 1);
  std::map<std::string, void*> all_tensors;
  char *input_token = (char*)(0x1001f22b400);
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = (char*)(0x1000a820000);
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = (char*)(0x1000d400000);
  all_tensors["sin_position_embedding"] = sin_position_embedding;
  void *embed_out;
  cudaMalloc(&embed_out, 16384);
  all_tensors["embed_out"] = embed_out;
  void *rmsnorm_out;
  cudaMalloc(&rmsnorm_out, 16384);
  all_tensors["rmsnorm_out"] = rmsnorm_out;
  void *attn_in;
  cudaMalloc(&attn_in, 65536);
  all_tensors["attn_in"] = attn_in;
  void *attn_out;
  cudaMalloc(&attn_out, 32768);
  all_tensors["attn_out"] = attn_out;
  void *attn_proj_out;
  cudaMalloc(&attn_proj_out, 16384);
  all_tensors["attn_proj_out"] = attn_proj_out;
  void *mlp_mid;
  cudaMalloc(&mlp_mid, 98304);
  all_tensors["mlp_mid"] = mlp_mid;
  void *silu_mul_out;
  cudaMalloc(&silu_mul_out, 49152);
  all_tensors["silu_mul_out"] = silu_mul_out;
  void *mlp_out;
  cudaMalloc(&mlp_out, 16384);
  all_tensors["mlp_out"] = mlp_out;
  void *argmax_in;
  cudaMalloc(&argmax_in, 2457600);
  all_tensors["argmax_in"] = argmax_in;
  void *argmax_part_value;
  cudaMalloc(&argmax_part_value, 1536);
  all_tensors["argmax_part_value"] = argmax_part_value;
  void *argmax_part_index;
  cudaMalloc(&argmax_part_index, 6144);
  all_tensors["argmax_part_index"] = argmax_part_index;
  char *output_token = (char*)(0x1001f22b600);
  all_tensors["output_token"] = output_token;
  char *embed_tokens = (char*)(0x101e0000000);
  all_tensors["embed_tokens"] = embed_tokens;
  char *layer_0_input_layernorm = (char*)(0x1001f200800);
  all_tensors["layer_0_input_layernorm"] = layer_0_input_layernorm;
  char *layer_0_qkv_proj;
  cudaMalloc(&layer_0_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x101f2a00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x101f2e00000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x101f3000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_0_qkv_proj"] = layer_0_qkv_proj;
  char *layer_0_q_norm = (char*)(0x1001f200200);
  all_tensors["layer_0_q_norm"] = layer_0_q_norm;
  char *layer_0_k_norm = (char*)(0x1001f200400);
  all_tensors["layer_0_k_norm"] = layer_0_k_norm;
  char *layer_0_k_cache = (char*)(0x10020000000);
  all_tensors["layer_0_k_cache"] = layer_0_k_cache;
  char *layer_0_v_cache = (char*)(0x10100000000);
  all_tensors["layer_0_v_cache"] = layer_0_v_cache;
  char *layer_0_o_proj = (char*)(0x101f3200000);
  all_tensors["layer_0_o_proj"] = layer_0_o_proj;
  char *layer_0_post_attn_layernorm = (char*)(0x1001f201000);
  all_tensors["layer_0_post_attn_layernorm"] = layer_0_post_attn_layernorm;
  char *layer_0_gatedup_proj;
  cudaMalloc(&layer_0_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x101f3600000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x101f4000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_0_gatedup_proj"] = layer_0_gatedup_proj;
  char *layer_0_down_proj = (char*)(0x101f4600000);
  all_tensors["layer_0_down_proj"] = layer_0_down_proj;
  char *layer_1_input_layernorm = (char*)(0x1001f201e00);
  all_tensors["layer_1_input_layernorm"] = layer_1_input_layernorm;
  char *layer_1_qkv_proj;
  cudaMalloc(&layer_1_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x101f4c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x101f3c00000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x101f5000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_1_qkv_proj"] = layer_1_qkv_proj;
  char *layer_1_q_norm = (char*)(0x1001f201800);
  all_tensors["layer_1_q_norm"] = layer_1_q_norm;
  char *layer_1_k_norm = (char*)(0x1001f201a00);
  all_tensors["layer_1_k_norm"] = layer_1_k_norm;
  char *layer_1_k_cache = (char*)(0x10028000000);
  all_tensors["layer_1_k_cache"] = layer_1_k_cache;
  char *layer_1_v_cache = (char*)(0x10108000000);
  all_tensors["layer_1_v_cache"] = layer_1_v_cache;
  char *layer_1_o_proj = (char*)(0x101f6000000);
  all_tensors["layer_1_o_proj"] = layer_1_o_proj;
  char *layer_1_post_attn_layernorm = (char*)(0x1001f202600);
  all_tensors["layer_1_post_attn_layernorm"] = layer_1_post_attn_layernorm;
  char *layer_1_gatedup_proj;
  cudaMalloc(&layer_1_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x101f6400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x101f6a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_1_gatedup_proj"] = layer_1_gatedup_proj;
  char *layer_1_down_proj = (char*)(0x101f8000000);
  all_tensors["layer_1_down_proj"] = layer_1_down_proj;
  char *layer_2_input_layernorm = (char*)(0x1001f203400);
  all_tensors["layer_2_input_layernorm"] = layer_2_input_layernorm;
  char *layer_2_qkv_proj;
  cudaMalloc(&layer_2_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x101f7000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x101f5200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x101f8600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_2_qkv_proj"] = layer_2_qkv_proj;
  char *layer_2_q_norm = (char*)(0x1001f202e00);
  all_tensors["layer_2_q_norm"] = layer_2_q_norm;
  char *layer_2_k_norm = (char*)(0x1001f203000);
  all_tensors["layer_2_k_norm"] = layer_2_k_norm;
  char *layer_2_k_cache = (char*)(0x10030000000);
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = (char*)(0x10110000000);
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_o_proj = (char*)(0x101f8800000);
  all_tensors["layer_2_o_proj"] = layer_2_o_proj;
  char *layer_2_post_attn_layernorm = (char*)(0x1001f203c00);
  all_tensors["layer_2_post_attn_layernorm"] = layer_2_post_attn_layernorm;
  char *layer_2_gatedup_proj;
  cudaMalloc(&layer_2_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x101f8c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x101fa000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_2_gatedup_proj"] = layer_2_gatedup_proj;
  char *layer_2_down_proj = (char*)(0x101fa600000);
  all_tensors["layer_2_down_proj"] = layer_2_down_proj;
  char *layer_3_input_layernorm = (char*)(0x1001f204a00);
  all_tensors["layer_3_input_layernorm"] = layer_3_input_layernorm;
  char *layer_3_qkv_proj;
  cudaMalloc(&layer_3_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x101fac00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x101f9200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x101fb000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_3_qkv_proj"] = layer_3_qkv_proj;
  char *layer_3_q_norm = (char*)(0x1001f204400);
  all_tensors["layer_3_q_norm"] = layer_3_q_norm;
  char *layer_3_k_norm = (char*)(0x1001f204600);
  all_tensors["layer_3_k_norm"] = layer_3_k_norm;
  char *layer_3_k_cache = (char*)(0x10038000000);
  all_tensors["layer_3_k_cache"] = layer_3_k_cache;
  char *layer_3_v_cache = (char*)(0x10118000000);
  all_tensors["layer_3_v_cache"] = layer_3_v_cache;
  char *layer_3_o_proj = (char*)(0x101fc000000);
  all_tensors["layer_3_o_proj"] = layer_3_o_proj;
  char *layer_3_post_attn_layernorm = (char*)(0x1001f205200);
  all_tensors["layer_3_post_attn_layernorm"] = layer_3_post_attn_layernorm;
  char *layer_3_gatedup_proj;
  cudaMalloc(&layer_3_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x101fc400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x101fca00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_3_gatedup_proj"] = layer_3_gatedup_proj;
  char *layer_3_down_proj = (char*)(0x101fe000000);
  all_tensors["layer_3_down_proj"] = layer_3_down_proj;
  char *layer_4_input_layernorm = (char*)(0x1001f206000);
  all_tensors["layer_4_input_layernorm"] = layer_4_input_layernorm;
  char *layer_4_qkv_proj;
  cudaMalloc(&layer_4_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x101fd000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x101fb200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x101fe600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_4_qkv_proj"] = layer_4_qkv_proj;
  char *layer_4_q_norm = (char*)(0x1001f205a00);
  all_tensors["layer_4_q_norm"] = layer_4_q_norm;
  char *layer_4_k_norm = (char*)(0x1001f205c00);
  all_tensors["layer_4_k_norm"] = layer_4_k_norm;
  char *layer_4_k_cache = (char*)(0x10040000000);
  all_tensors["layer_4_k_cache"] = layer_4_k_cache;
  char *layer_4_v_cache = (char*)(0x10120000000);
  all_tensors["layer_4_v_cache"] = layer_4_v_cache;
  char *layer_4_o_proj = (char*)(0x101fe800000);
  all_tensors["layer_4_o_proj"] = layer_4_o_proj;
  char *layer_4_post_attn_layernorm = (char*)(0x1001f206800);
  all_tensors["layer_4_post_attn_layernorm"] = layer_4_post_attn_layernorm;
  char *layer_4_gatedup_proj;
  cudaMalloc(&layer_4_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x101fec00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10200000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_4_gatedup_proj"] = layer_4_gatedup_proj;
  char *layer_4_down_proj = (char*)(0x10200600000);
  all_tensors["layer_4_down_proj"] = layer_4_down_proj;
  char *layer_5_input_layernorm = (char*)(0x1001f207600);
  all_tensors["layer_5_input_layernorm"] = layer_5_input_layernorm;
  char *layer_5_qkv_proj;
  cudaMalloc(&layer_5_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10200c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x101ff200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10201000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_5_qkv_proj"] = layer_5_qkv_proj;
  char *layer_5_q_norm = (char*)(0x1001f207000);
  all_tensors["layer_5_q_norm"] = layer_5_q_norm;
  char *layer_5_k_norm = (char*)(0x1001f207200);
  all_tensors["layer_5_k_norm"] = layer_5_k_norm;
  char *layer_5_k_cache = (char*)(0x10048000000);
  all_tensors["layer_5_k_cache"] = layer_5_k_cache;
  char *layer_5_v_cache = (char*)(0x10128000000);
  all_tensors["layer_5_v_cache"] = layer_5_v_cache;
  char *layer_5_o_proj = (char*)(0x10202000000);
  all_tensors["layer_5_o_proj"] = layer_5_o_proj;
  char *layer_5_post_attn_layernorm = (char*)(0x1001f207e00);
  all_tensors["layer_5_post_attn_layernorm"] = layer_5_post_attn_layernorm;
  char *layer_5_gatedup_proj;
  cudaMalloc(&layer_5_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10202400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10202a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_5_gatedup_proj"] = layer_5_gatedup_proj;
  char *layer_5_down_proj = (char*)(0x10204000000);
  all_tensors["layer_5_down_proj"] = layer_5_down_proj;
  char *layer_6_input_layernorm = (char*)(0x1001f208c00);
  all_tensors["layer_6_input_layernorm"] = layer_6_input_layernorm;
  char *layer_6_qkv_proj;
  cudaMalloc(&layer_6_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10203000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10201200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10204600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_6_qkv_proj"] = layer_6_qkv_proj;
  char *layer_6_q_norm = (char*)(0x1001f208600);
  all_tensors["layer_6_q_norm"] = layer_6_q_norm;
  char *layer_6_k_norm = (char*)(0x1001f208800);
  all_tensors["layer_6_k_norm"] = layer_6_k_norm;
  char *layer_6_k_cache = (char*)(0x10050000000);
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = (char*)(0x10130000000);
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_o_proj = (char*)(0x10204800000);
  all_tensors["layer_6_o_proj"] = layer_6_o_proj;
  char *layer_6_post_attn_layernorm = (char*)(0x1001f209400);
  all_tensors["layer_6_post_attn_layernorm"] = layer_6_post_attn_layernorm;
  char *layer_6_gatedup_proj;
  cudaMalloc(&layer_6_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10204c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10206000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_6_gatedup_proj"] = layer_6_gatedup_proj;
  char *layer_6_down_proj = (char*)(0x10206600000);
  all_tensors["layer_6_down_proj"] = layer_6_down_proj;
  char *layer_7_input_layernorm = (char*)(0x1001f20a200);
  all_tensors["layer_7_input_layernorm"] = layer_7_input_layernorm;
  char *layer_7_qkv_proj;
  cudaMalloc(&layer_7_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10206c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10205200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10207000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_7_qkv_proj"] = layer_7_qkv_proj;
  char *layer_7_q_norm = (char*)(0x1001f209c00);
  all_tensors["layer_7_q_norm"] = layer_7_q_norm;
  char *layer_7_k_norm = (char*)(0x1001f209e00);
  all_tensors["layer_7_k_norm"] = layer_7_k_norm;
  char *layer_7_k_cache = (char*)(0x10058000000);
  all_tensors["layer_7_k_cache"] = layer_7_k_cache;
  char *layer_7_v_cache = (char*)(0x10138000000);
  all_tensors["layer_7_v_cache"] = layer_7_v_cache;
  char *layer_7_o_proj = (char*)(0x10208000000);
  all_tensors["layer_7_o_proj"] = layer_7_o_proj;
  char *layer_7_post_attn_layernorm = (char*)(0x1001f20aa00);
  all_tensors["layer_7_post_attn_layernorm"] = layer_7_post_attn_layernorm;
  char *layer_7_gatedup_proj;
  cudaMalloc(&layer_7_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10208400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10208a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_7_gatedup_proj"] = layer_7_gatedup_proj;
  char *layer_7_down_proj = (char*)(0x1020a000000);
  all_tensors["layer_7_down_proj"] = layer_7_down_proj;
  char *layer_8_input_layernorm = (char*)(0x1001f20b800);
  all_tensors["layer_8_input_layernorm"] = layer_8_input_layernorm;
  char *layer_8_qkv_proj;
  cudaMalloc(&layer_8_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10209000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10207200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1020a600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_8_qkv_proj"] = layer_8_qkv_proj;
  char *layer_8_q_norm = (char*)(0x1001f20b200);
  all_tensors["layer_8_q_norm"] = layer_8_q_norm;
  char *layer_8_k_norm = (char*)(0x1001f20b400);
  all_tensors["layer_8_k_norm"] = layer_8_k_norm;
  char *layer_8_k_cache = (char*)(0x10060000000);
  all_tensors["layer_8_k_cache"] = layer_8_k_cache;
  char *layer_8_v_cache = (char*)(0x10140000000);
  all_tensors["layer_8_v_cache"] = layer_8_v_cache;
  char *layer_8_o_proj = (char*)(0x1020a800000);
  all_tensors["layer_8_o_proj"] = layer_8_o_proj;
  char *layer_8_post_attn_layernorm = (char*)(0x1001f20c000);
  all_tensors["layer_8_post_attn_layernorm"] = layer_8_post_attn_layernorm;
  char *layer_8_gatedup_proj;
  cudaMalloc(&layer_8_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1020ac00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1020c000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_8_gatedup_proj"] = layer_8_gatedup_proj;
  char *layer_8_down_proj = (char*)(0x1020c600000);
  all_tensors["layer_8_down_proj"] = layer_8_down_proj;
  char *layer_9_input_layernorm = (char*)(0x1001f20ce00);
  all_tensors["layer_9_input_layernorm"] = layer_9_input_layernorm;
  char *layer_9_qkv_proj;
  cudaMalloc(&layer_9_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1020cc00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1020b200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1020d000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_9_qkv_proj"] = layer_9_qkv_proj;
  char *layer_9_q_norm = (char*)(0x1001f20c800);
  all_tensors["layer_9_q_norm"] = layer_9_q_norm;
  char *layer_9_k_norm = (char*)(0x1001f20ca00);
  all_tensors["layer_9_k_norm"] = layer_9_k_norm;
  char *layer_9_k_cache = (char*)(0x10068000000);
  all_tensors["layer_9_k_cache"] = layer_9_k_cache;
  char *layer_9_v_cache = (char*)(0x10148000000);
  all_tensors["layer_9_v_cache"] = layer_9_v_cache;
  char *layer_9_o_proj = (char*)(0x1020e000000);
  all_tensors["layer_9_o_proj"] = layer_9_o_proj;
  char *layer_9_post_attn_layernorm = (char*)(0x1001f20d600);
  all_tensors["layer_9_post_attn_layernorm"] = layer_9_post_attn_layernorm;
  char *layer_9_gatedup_proj;
  cudaMalloc(&layer_9_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1020e400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1020ea00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_9_gatedup_proj"] = layer_9_gatedup_proj;
  char *layer_9_down_proj = (char*)(0x10210000000);
  all_tensors["layer_9_down_proj"] = layer_9_down_proj;
  char *layer_10_input_layernorm = (char*)(0x1001f20e400);
  all_tensors["layer_10_input_layernorm"] = layer_10_input_layernorm;
  char *layer_10_qkv_proj;
  cudaMalloc(&layer_10_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1020f000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1020d200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10210600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_10_qkv_proj"] = layer_10_qkv_proj;
  char *layer_10_q_norm = (char*)(0x1001f20de00);
  all_tensors["layer_10_q_norm"] = layer_10_q_norm;
  char *layer_10_k_norm = (char*)(0x1001f20e000);
  all_tensors["layer_10_k_norm"] = layer_10_k_norm;
  char *layer_10_k_cache = (char*)(0x10070000000);
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = (char*)(0x10150000000);
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_o_proj = (char*)(0x10210800000);
  all_tensors["layer_10_o_proj"] = layer_10_o_proj;
  char *layer_10_post_attn_layernorm = (char*)(0x1001f20ec00);
  all_tensors["layer_10_post_attn_layernorm"] = layer_10_post_attn_layernorm;
  char *layer_10_gatedup_proj;
  cudaMalloc(&layer_10_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10210c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10212000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_10_gatedup_proj"] = layer_10_gatedup_proj;
  char *layer_10_down_proj = (char*)(0x10212600000);
  all_tensors["layer_10_down_proj"] = layer_10_down_proj;
  char *layer_11_input_layernorm = (char*)(0x1001f20fa00);
  all_tensors["layer_11_input_layernorm"] = layer_11_input_layernorm;
  char *layer_11_qkv_proj;
  cudaMalloc(&layer_11_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10212c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10211200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10213000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_11_qkv_proj"] = layer_11_qkv_proj;
  char *layer_11_q_norm = (char*)(0x1001f20f400);
  all_tensors["layer_11_q_norm"] = layer_11_q_norm;
  char *layer_11_k_norm = (char*)(0x1001f20f600);
  all_tensors["layer_11_k_norm"] = layer_11_k_norm;
  char *layer_11_k_cache = (char*)(0x10078000000);
  all_tensors["layer_11_k_cache"] = layer_11_k_cache;
  char *layer_11_v_cache = (char*)(0x10158000000);
  all_tensors["layer_11_v_cache"] = layer_11_v_cache;
  char *layer_11_o_proj = (char*)(0x10214000000);
  all_tensors["layer_11_o_proj"] = layer_11_o_proj;
  char *layer_11_post_attn_layernorm = (char*)(0x1001f210200);
  all_tensors["layer_11_post_attn_layernorm"] = layer_11_post_attn_layernorm;
  char *layer_11_gatedup_proj;
  cudaMalloc(&layer_11_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10214400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10214a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_11_gatedup_proj"] = layer_11_gatedup_proj;
  char *layer_11_down_proj = (char*)(0x10216000000);
  all_tensors["layer_11_down_proj"] = layer_11_down_proj;
  char *layer_12_input_layernorm = (char*)(0x1001f211000);
  all_tensors["layer_12_input_layernorm"] = layer_12_input_layernorm;
  char *layer_12_qkv_proj;
  cudaMalloc(&layer_12_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10215000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10213200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10216600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_12_qkv_proj"] = layer_12_qkv_proj;
  char *layer_12_q_norm = (char*)(0x1001f210a00);
  all_tensors["layer_12_q_norm"] = layer_12_q_norm;
  char *layer_12_k_norm = (char*)(0x1001f210c00);
  all_tensors["layer_12_k_norm"] = layer_12_k_norm;
  char *layer_12_k_cache = (char*)(0x10080000000);
  all_tensors["layer_12_k_cache"] = layer_12_k_cache;
  char *layer_12_v_cache = (char*)(0x10160000000);
  all_tensors["layer_12_v_cache"] = layer_12_v_cache;
  char *layer_12_o_proj = (char*)(0x10216800000);
  all_tensors["layer_12_o_proj"] = layer_12_o_proj;
  char *layer_12_post_attn_layernorm = (char*)(0x1001f211800);
  all_tensors["layer_12_post_attn_layernorm"] = layer_12_post_attn_layernorm;
  char *layer_12_gatedup_proj;
  cudaMalloc(&layer_12_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10216c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10218000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_12_gatedup_proj"] = layer_12_gatedup_proj;
  char *layer_12_down_proj = (char*)(0x10218600000);
  all_tensors["layer_12_down_proj"] = layer_12_down_proj;
  char *layer_13_input_layernorm = (char*)(0x1001f212600);
  all_tensors["layer_13_input_layernorm"] = layer_13_input_layernorm;
  char *layer_13_qkv_proj;
  cudaMalloc(&layer_13_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10218c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10217200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10219000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_13_qkv_proj"] = layer_13_qkv_proj;
  char *layer_13_q_norm = (char*)(0x1001f212000);
  all_tensors["layer_13_q_norm"] = layer_13_q_norm;
  char *layer_13_k_norm = (char*)(0x1001f212200);
  all_tensors["layer_13_k_norm"] = layer_13_k_norm;
  char *layer_13_k_cache = (char*)(0x10088000000);
  all_tensors["layer_13_k_cache"] = layer_13_k_cache;
  char *layer_13_v_cache = (char*)(0x10168000000);
  all_tensors["layer_13_v_cache"] = layer_13_v_cache;
  char *layer_13_o_proj = (char*)(0x1021a000000);
  all_tensors["layer_13_o_proj"] = layer_13_o_proj;
  char *layer_13_post_attn_layernorm = (char*)(0x1001f212e00);
  all_tensors["layer_13_post_attn_layernorm"] = layer_13_post_attn_layernorm;
  char *layer_13_gatedup_proj;
  cudaMalloc(&layer_13_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1021a400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1021aa00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_13_gatedup_proj"] = layer_13_gatedup_proj;
  char *layer_13_down_proj = (char*)(0x1021c000000);
  all_tensors["layer_13_down_proj"] = layer_13_down_proj;
  char *layer_14_input_layernorm = (char*)(0x1001f213c00);
  all_tensors["layer_14_input_layernorm"] = layer_14_input_layernorm;
  char *layer_14_qkv_proj;
  cudaMalloc(&layer_14_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1021b000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10219200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1021c600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_14_qkv_proj"] = layer_14_qkv_proj;
  char *layer_14_q_norm = (char*)(0x1001f213600);
  all_tensors["layer_14_q_norm"] = layer_14_q_norm;
  char *layer_14_k_norm = (char*)(0x1001f213800);
  all_tensors["layer_14_k_norm"] = layer_14_k_norm;
  char *layer_14_k_cache = (char*)(0x10090000000);
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = (char*)(0x10170000000);
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_o_proj = (char*)(0x1021c800000);
  all_tensors["layer_14_o_proj"] = layer_14_o_proj;
  char *layer_14_post_attn_layernorm = (char*)(0x1001f214400);
  all_tensors["layer_14_post_attn_layernorm"] = layer_14_post_attn_layernorm;
  char *layer_14_gatedup_proj;
  cudaMalloc(&layer_14_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1021cc00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1021e000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_14_gatedup_proj"] = layer_14_gatedup_proj;
  char *layer_14_down_proj = (char*)(0x1021e600000);
  all_tensors["layer_14_down_proj"] = layer_14_down_proj;
  char *layer_15_input_layernorm = (char*)(0x1001f215200);
  all_tensors["layer_15_input_layernorm"] = layer_15_input_layernorm;
  char *layer_15_qkv_proj;
  cudaMalloc(&layer_15_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1021ec00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1021d200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1021f000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_15_qkv_proj"] = layer_15_qkv_proj;
  char *layer_15_q_norm = (char*)(0x1001f214c00);
  all_tensors["layer_15_q_norm"] = layer_15_q_norm;
  char *layer_15_k_norm = (char*)(0x1001f214e00);
  all_tensors["layer_15_k_norm"] = layer_15_k_norm;
  char *layer_15_k_cache = (char*)(0x10098000000);
  all_tensors["layer_15_k_cache"] = layer_15_k_cache;
  char *layer_15_v_cache = (char*)(0x10178000000);
  all_tensors["layer_15_v_cache"] = layer_15_v_cache;
  char *layer_15_o_proj = (char*)(0x10220000000);
  all_tensors["layer_15_o_proj"] = layer_15_o_proj;
  char *layer_15_post_attn_layernorm = (char*)(0x1001f215a00);
  all_tensors["layer_15_post_attn_layernorm"] = layer_15_post_attn_layernorm;
  char *layer_15_gatedup_proj;
  cudaMalloc(&layer_15_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10220400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10220a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_15_gatedup_proj"] = layer_15_gatedup_proj;
  char *layer_15_down_proj = (char*)(0x10222000000);
  all_tensors["layer_15_down_proj"] = layer_15_down_proj;
  char *layer_16_input_layernorm = (char*)(0x1001f216800);
  all_tensors["layer_16_input_layernorm"] = layer_16_input_layernorm;
  char *layer_16_qkv_proj;
  cudaMalloc(&layer_16_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10221000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1021f200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10222600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_16_qkv_proj"] = layer_16_qkv_proj;
  char *layer_16_q_norm = (char*)(0x1001f216200);
  all_tensors["layer_16_q_norm"] = layer_16_q_norm;
  char *layer_16_k_norm = (char*)(0x1001f216400);
  all_tensors["layer_16_k_norm"] = layer_16_k_norm;
  char *layer_16_k_cache = (char*)(0x100a0000000);
  all_tensors["layer_16_k_cache"] = layer_16_k_cache;
  char *layer_16_v_cache = (char*)(0x10180000000);
  all_tensors["layer_16_v_cache"] = layer_16_v_cache;
  char *layer_16_o_proj = (char*)(0x10222800000);
  all_tensors["layer_16_o_proj"] = layer_16_o_proj;
  char *layer_16_post_attn_layernorm = (char*)(0x1001f217000);
  all_tensors["layer_16_post_attn_layernorm"] = layer_16_post_attn_layernorm;
  char *layer_16_gatedup_proj;
  cudaMalloc(&layer_16_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10222c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10224000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_16_gatedup_proj"] = layer_16_gatedup_proj;
  char *layer_16_down_proj = (char*)(0x10224600000);
  all_tensors["layer_16_down_proj"] = layer_16_down_proj;
  char *layer_17_input_layernorm = (char*)(0x1001f217e00);
  all_tensors["layer_17_input_layernorm"] = layer_17_input_layernorm;
  char *layer_17_qkv_proj;
  cudaMalloc(&layer_17_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10224c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10223200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10225000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_17_qkv_proj"] = layer_17_qkv_proj;
  char *layer_17_q_norm = (char*)(0x1001f217800);
  all_tensors["layer_17_q_norm"] = layer_17_q_norm;
  char *layer_17_k_norm = (char*)(0x1001f217a00);
  all_tensors["layer_17_k_norm"] = layer_17_k_norm;
  char *layer_17_k_cache = (char*)(0x100a8000000);
  all_tensors["layer_17_k_cache"] = layer_17_k_cache;
  char *layer_17_v_cache = (char*)(0x10188000000);
  all_tensors["layer_17_v_cache"] = layer_17_v_cache;
  char *layer_17_o_proj = (char*)(0x10226000000);
  all_tensors["layer_17_o_proj"] = layer_17_o_proj;
  char *layer_17_post_attn_layernorm = (char*)(0x1001f218600);
  all_tensors["layer_17_post_attn_layernorm"] = layer_17_post_attn_layernorm;
  char *layer_17_gatedup_proj;
  cudaMalloc(&layer_17_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10226400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10226a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_17_gatedup_proj"] = layer_17_gatedup_proj;
  char *layer_17_down_proj = (char*)(0x10228000000);
  all_tensors["layer_17_down_proj"] = layer_17_down_proj;
  char *layer_18_input_layernorm = (char*)(0x1001f219400);
  all_tensors["layer_18_input_layernorm"] = layer_18_input_layernorm;
  char *layer_18_qkv_proj;
  cudaMalloc(&layer_18_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10227000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10225200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10228600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_18_qkv_proj"] = layer_18_qkv_proj;
  char *layer_18_q_norm = (char*)(0x1001f218e00);
  all_tensors["layer_18_q_norm"] = layer_18_q_norm;
  char *layer_18_k_norm = (char*)(0x1001f219000);
  all_tensors["layer_18_k_norm"] = layer_18_k_norm;
  char *layer_18_k_cache = (char*)(0x100b0000000);
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = (char*)(0x10190000000);
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_o_proj = (char*)(0x10228800000);
  all_tensors["layer_18_o_proj"] = layer_18_o_proj;
  char *layer_18_post_attn_layernorm = (char*)(0x1001f219c00);
  all_tensors["layer_18_post_attn_layernorm"] = layer_18_post_attn_layernorm;
  char *layer_18_gatedup_proj;
  cudaMalloc(&layer_18_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10228c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1022a000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_18_gatedup_proj"] = layer_18_gatedup_proj;
  char *layer_18_down_proj = (char*)(0x1022a600000);
  all_tensors["layer_18_down_proj"] = layer_18_down_proj;
  char *layer_19_input_layernorm = (char*)(0x1001f21aa00);
  all_tensors["layer_19_input_layernorm"] = layer_19_input_layernorm;
  char *layer_19_qkv_proj;
  cudaMalloc(&layer_19_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1022ac00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10229200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1022b000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_19_qkv_proj"] = layer_19_qkv_proj;
  char *layer_19_q_norm = (char*)(0x1001f21a400);
  all_tensors["layer_19_q_norm"] = layer_19_q_norm;
  char *layer_19_k_norm = (char*)(0x1001f21a600);
  all_tensors["layer_19_k_norm"] = layer_19_k_norm;
  char *layer_19_k_cache = (char*)(0x100b8000000);
  all_tensors["layer_19_k_cache"] = layer_19_k_cache;
  char *layer_19_v_cache = (char*)(0x10198000000);
  all_tensors["layer_19_v_cache"] = layer_19_v_cache;
  char *layer_19_o_proj = (char*)(0x1022c000000);
  all_tensors["layer_19_o_proj"] = layer_19_o_proj;
  char *layer_19_post_attn_layernorm = (char*)(0x1001f21b200);
  all_tensors["layer_19_post_attn_layernorm"] = layer_19_post_attn_layernorm;
  char *layer_19_gatedup_proj;
  cudaMalloc(&layer_19_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1022c400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1022ca00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_19_gatedup_proj"] = layer_19_gatedup_proj;
  char *layer_19_down_proj = (char*)(0x1022e000000);
  all_tensors["layer_19_down_proj"] = layer_19_down_proj;
  char *layer_20_input_layernorm = (char*)(0x1001f21c000);
  all_tensors["layer_20_input_layernorm"] = layer_20_input_layernorm;
  char *layer_20_qkv_proj;
  cudaMalloc(&layer_20_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1022d000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1022b200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1022e600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_20_qkv_proj"] = layer_20_qkv_proj;
  char *layer_20_q_norm = (char*)(0x1001f21ba00);
  all_tensors["layer_20_q_norm"] = layer_20_q_norm;
  char *layer_20_k_norm = (char*)(0x1001f21bc00);
  all_tensors["layer_20_k_norm"] = layer_20_k_norm;
  char *layer_20_k_cache = (char*)(0x100c0000000);
  all_tensors["layer_20_k_cache"] = layer_20_k_cache;
  char *layer_20_v_cache = (char*)(0x101a0000000);
  all_tensors["layer_20_v_cache"] = layer_20_v_cache;
  char *layer_20_o_proj = (char*)(0x1022e800000);
  all_tensors["layer_20_o_proj"] = layer_20_o_proj;
  char *layer_20_post_attn_layernorm = (char*)(0x1001f21c800);
  all_tensors["layer_20_post_attn_layernorm"] = layer_20_post_attn_layernorm;
  char *layer_20_gatedup_proj;
  cudaMalloc(&layer_20_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1022ec00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10230000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_20_gatedup_proj"] = layer_20_gatedup_proj;
  char *layer_20_down_proj = (char*)(0x10230600000);
  all_tensors["layer_20_down_proj"] = layer_20_down_proj;
  char *layer_21_input_layernorm = (char*)(0x1001f21d600);
  all_tensors["layer_21_input_layernorm"] = layer_21_input_layernorm;
  char *layer_21_qkv_proj;
  cudaMalloc(&layer_21_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10230c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1022f200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10231000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_21_qkv_proj"] = layer_21_qkv_proj;
  char *layer_21_q_norm = (char*)(0x1001f21d000);
  all_tensors["layer_21_q_norm"] = layer_21_q_norm;
  char *layer_21_k_norm = (char*)(0x1001f21d200);
  all_tensors["layer_21_k_norm"] = layer_21_k_norm;
  char *layer_21_k_cache = (char*)(0x100c8000000);
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = (char*)(0x101a8000000);
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_o_proj = (char*)(0x10232000000);
  all_tensors["layer_21_o_proj"] = layer_21_o_proj;
  char *layer_21_post_attn_layernorm = (char*)(0x1001f21de00);
  all_tensors["layer_21_post_attn_layernorm"] = layer_21_post_attn_layernorm;
  char *layer_21_gatedup_proj;
  cudaMalloc(&layer_21_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10232400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10232a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_21_gatedup_proj"] = layer_21_gatedup_proj;
  char *layer_21_down_proj = (char*)(0x10234000000);
  all_tensors["layer_21_down_proj"] = layer_21_down_proj;
  char *layer_22_input_layernorm = (char*)(0x1001f21ec00);
  all_tensors["layer_22_input_layernorm"] = layer_22_input_layernorm;
  char *layer_22_qkv_proj;
  cudaMalloc(&layer_22_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10233000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10231200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10234600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_22_qkv_proj"] = layer_22_qkv_proj;
  char *layer_22_q_norm = (char*)(0x1001f21e600);
  all_tensors["layer_22_q_norm"] = layer_22_q_norm;
  char *layer_22_k_norm = (char*)(0x1001f21e800);
  all_tensors["layer_22_k_norm"] = layer_22_k_norm;
  char *layer_22_k_cache = (char*)(0x100d0000000);
  all_tensors["layer_22_k_cache"] = layer_22_k_cache;
  char *layer_22_v_cache = (char*)(0x101b0000000);
  all_tensors["layer_22_v_cache"] = layer_22_v_cache;
  char *layer_22_o_proj = (char*)(0x10234800000);
  all_tensors["layer_22_o_proj"] = layer_22_o_proj;
  char *layer_22_post_attn_layernorm = (char*)(0x1001f21f400);
  all_tensors["layer_22_post_attn_layernorm"] = layer_22_post_attn_layernorm;
  char *layer_22_gatedup_proj;
  cudaMalloc(&layer_22_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10234c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10236000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_22_gatedup_proj"] = layer_22_gatedup_proj;
  char *layer_22_down_proj = (char*)(0x10236600000);
  all_tensors["layer_22_down_proj"] = layer_22_down_proj;
  char *layer_23_input_layernorm = (char*)(0x1001f220200);
  all_tensors["layer_23_input_layernorm"] = layer_23_input_layernorm;
  char *layer_23_qkv_proj;
  cudaMalloc(&layer_23_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10236c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10235200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10237000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_23_qkv_proj"] = layer_23_qkv_proj;
  char *layer_23_q_norm = (char*)(0x1001f21fc00);
  all_tensors["layer_23_q_norm"] = layer_23_q_norm;
  char *layer_23_k_norm = (char*)(0x1001f21fe00);
  all_tensors["layer_23_k_norm"] = layer_23_k_norm;
  char *layer_23_k_cache = (char*)(0x100d8000000);
  all_tensors["layer_23_k_cache"] = layer_23_k_cache;
  char *layer_23_v_cache = (char*)(0x101b8000000);
  all_tensors["layer_23_v_cache"] = layer_23_v_cache;
  char *layer_23_o_proj = (char*)(0x10238000000);
  all_tensors["layer_23_o_proj"] = layer_23_o_proj;
  char *layer_23_post_attn_layernorm = (char*)(0x1001f220a00);
  all_tensors["layer_23_post_attn_layernorm"] = layer_23_post_attn_layernorm;
  char *layer_23_gatedup_proj;
  cudaMalloc(&layer_23_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10238400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10238a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_23_gatedup_proj"] = layer_23_gatedup_proj;
  char *layer_23_down_proj = (char*)(0x1023a000000);
  all_tensors["layer_23_down_proj"] = layer_23_down_proj;
  char *layer_24_input_layernorm = (char*)(0x1001f221800);
  all_tensors["layer_24_input_layernorm"] = layer_24_input_layernorm;
  char *layer_24_qkv_proj;
  cudaMalloc(&layer_24_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10239000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10237200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1023a600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_24_qkv_proj"] = layer_24_qkv_proj;
  char *layer_24_q_norm = (char*)(0x1001f221200);
  all_tensors["layer_24_q_norm"] = layer_24_q_norm;
  char *layer_24_k_norm = (char*)(0x1001f221400);
  all_tensors["layer_24_k_norm"] = layer_24_k_norm;
  char *layer_24_k_cache = (char*)(0x100e0000000);
  all_tensors["layer_24_k_cache"] = layer_24_k_cache;
  char *layer_24_v_cache = (char*)(0x101c0000000);
  all_tensors["layer_24_v_cache"] = layer_24_v_cache;
  char *layer_24_o_proj = (char*)(0x1023a800000);
  all_tensors["layer_24_o_proj"] = layer_24_o_proj;
  char *layer_24_post_attn_layernorm = (char*)(0x1001f222000);
  all_tensors["layer_24_post_attn_layernorm"] = layer_24_post_attn_layernorm;
  char *layer_24_gatedup_proj;
  cudaMalloc(&layer_24_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1023ac00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1023c000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_24_gatedup_proj"] = layer_24_gatedup_proj;
  char *layer_24_down_proj = (char*)(0x1023c600000);
  all_tensors["layer_24_down_proj"] = layer_24_down_proj;
  char *layer_25_input_layernorm = (char*)(0x1001f222e00);
  all_tensors["layer_25_input_layernorm"] = layer_25_input_layernorm;
  char *layer_25_qkv_proj;
  cudaMalloc(&layer_25_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1023cc00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1023b200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x1023d000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_25_qkv_proj"] = layer_25_qkv_proj;
  char *layer_25_q_norm = (char*)(0x1001f222800);
  all_tensors["layer_25_q_norm"] = layer_25_q_norm;
  char *layer_25_k_norm = (char*)(0x1001f222a00);
  all_tensors["layer_25_k_norm"] = layer_25_k_norm;
  char *layer_25_k_cache = (char*)(0x100e8000000);
  all_tensors["layer_25_k_cache"] = layer_25_k_cache;
  char *layer_25_v_cache = (char*)(0x101c8000000);
  all_tensors["layer_25_v_cache"] = layer_25_v_cache;
  char *layer_25_o_proj = (char*)(0x1023e000000);
  all_tensors["layer_25_o_proj"] = layer_25_o_proj;
  char *layer_25_post_attn_layernorm = (char*)(0x1001f223600);
  all_tensors["layer_25_post_attn_layernorm"] = layer_25_post_attn_layernorm;
  char *layer_25_gatedup_proj;
  cudaMalloc(&layer_25_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x1023e400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x1023ea00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_25_gatedup_proj"] = layer_25_gatedup_proj;
  char *layer_25_down_proj = (char*)(0x10240000000);
  all_tensors["layer_25_down_proj"] = layer_25_down_proj;
  char *layer_26_input_layernorm = (char*)(0x1001f224400);
  all_tensors["layer_26_input_layernorm"] = layer_26_input_layernorm;
  char *layer_26_qkv_proj;
  cudaMalloc(&layer_26_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x1023f000000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x1023d200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10240600000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_26_qkv_proj"] = layer_26_qkv_proj;
  char *layer_26_q_norm = (char*)(0x1001f223e00);
  all_tensors["layer_26_q_norm"] = layer_26_q_norm;
  char *layer_26_k_norm = (char*)(0x1001f224000);
  all_tensors["layer_26_k_norm"] = layer_26_k_norm;
  char *layer_26_k_cache = (char*)(0x100f0000000);
  all_tensors["layer_26_k_cache"] = layer_26_k_cache;
  char *layer_26_v_cache = (char*)(0x101d0000000);
  all_tensors["layer_26_v_cache"] = layer_26_v_cache;
  char *layer_26_o_proj = (char*)(0x10240800000);
  all_tensors["layer_26_o_proj"] = layer_26_o_proj;
  char *layer_26_post_attn_layernorm = (char*)(0x1001f224c00);
  all_tensors["layer_26_post_attn_layernorm"] = layer_26_post_attn_layernorm;
  char *layer_26_gatedup_proj;
  cudaMalloc(&layer_26_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10240c00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10242000000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_26_gatedup_proj"] = layer_26_gatedup_proj;
  char *layer_26_down_proj = (char*)(0x10242600000);
  all_tensors["layer_26_down_proj"] = layer_26_down_proj;
  char *layer_27_input_layernorm = (char*)(0x1001f225a00);
  all_tensors["layer_27_input_layernorm"] = layer_27_input_layernorm;
  char *layer_27_qkv_proj;
  cudaMalloc(&layer_27_qkv_proj, 8388608);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 0), 1048576, reinterpret_cast<const void *>(0x10242c00000), 524288, 524288, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 524288), 1048576, reinterpret_cast<const void *>(0x10241200000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 786432), 1048576, reinterpret_cast<const void *>(0x10243000000), 262144, 262144, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_27_qkv_proj"] = layer_27_qkv_proj;
  char *layer_27_q_norm = (char*)(0x1001f225400);
  all_tensors["layer_27_q_norm"] = layer_27_q_norm;
  char *layer_27_k_norm = (char*)(0x1001f225600);
  all_tensors["layer_27_k_norm"] = layer_27_k_norm;
  char *layer_27_k_cache = (char*)(0x100f8000000);
  all_tensors["layer_27_k_cache"] = layer_27_k_cache;
  char *layer_27_v_cache = (char*)(0x101d8000000);
  all_tensors["layer_27_v_cache"] = layer_27_v_cache;
  char *layer_27_o_proj = (char*)(0x10244000000);
  all_tensors["layer_27_o_proj"] = layer_27_o_proj;
  char *layer_27_post_attn_layernorm = (char*)(0x1001f226200);
  all_tensors["layer_27_post_attn_layernorm"] = layer_27_post_attn_layernorm;
  char *layer_27_gatedup_proj;
  cudaMalloc(&layer_27_gatedup_proj, 12582912);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_gatedup_proj + 0), 262144, reinterpret_cast<const void *>(0x10244400000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_gatedup_proj + 131072), 262144, reinterpret_cast<const void *>(0x10244a00000), 131072, 131072, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_27_gatedup_proj"] = layer_27_gatedup_proj;
  char *layer_27_down_proj = (char*)(0x10246000000);
  all_tensors["layer_27_down_proj"] = layer_27_down_proj;
  char *model_norm_weight = (char*)(0x1001f226a00);
  all_tensors["model_norm_weight"] = model_norm_weight;
  char *lm_head = (char*)(0x10262000000);
  all_tensors["lm_head"] = lm_head;
  all_tensors["nullptr"] = nullptr;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
  cudaDeviceSynchronize();
}

__device__ __forceinline__
void _execute_task(TaskDesc const* task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_EMBEDDING && task_desc->variant_id == 0) {
      kernel::embedding_kernel<bfloat16, 8, 1024, 1024>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0]);

  }
  else if (task_desc->task_type == TASK_LINEAR_WITH_RESIDUAL && task_desc->variant_id == 0) {
      kernel::linear_kernel<bfloat16, 8, 64, 2048, 1024>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      runtime_config.my_gpu_id == 0);

  }
  else if (task_desc->task_type == TASK_LINEAR_WITH_RESIDUAL && task_desc->variant_id == 1) {
      kernel::linear_kernel<bfloat16, 8, 64, 3072, 1024>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      runtime_config.my_gpu_id == 0);

  }
  else if (task_desc->task_type == TASK_ARGMAX_PARTIAL && task_desc->variant_id == 0) {
      kernel::argmax_partial_kernel<bfloat16, 8, 1600, 96>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_ARGMAX_REDUCE && task_desc->variant_id == 0) {
      kernel::argmax_reduce_kernel<bfloat16, 8, 1600, 96>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_PAGED_ATTENTION_1 && task_desc->variant_id == 0) {
      kernel::multitoken_paged_attention_task_impl<bfloat16, 2, 1, 1024, 4096, 2048, 128, 512, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      task_desc->request_id,
      true,
      true,
      task_desc->input_ptrs[3],
      task_desc->input_ptrs[4],
      task_desc->input_ptrs[5],
      task_desc->input_ptrs[6],
      1e-6f,
      1e-6f);

  }
  else if (task_desc->task_type == TASK_SILU_MUL && task_desc->variant_id == 0) {
      kernel::silu_mul_task_impl<bfloat16, 8, 64, 6144, 3072>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_RMS_NORM && task_desc->variant_id == 0) {
      kernel::rms_norm_impl<bfloat16, 1, 1024>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      1e-6f);

  }
  else if (task_desc->task_type == TASK_LINEAR && task_desc->variant_id == 0) {
      kernel::linear_kernel<bfloat16, 8, 64, 1024, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      nullptr,
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      false/*residual*/);

  }
  else if (task_desc->task_type == TASK_LINEAR && task_desc->variant_id == 1) {
      kernel::linear_kernel<bfloat16, 8, 64, 1024, 6144>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      nullptr,
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      false/*residual*/);

  }
  else if (task_desc->task_type == TASK_LINEAR && task_desc->variant_id == 2) {
      kernel::linear_kernel<bfloat16, 8, 1600, 1024, 153600>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      nullptr,
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      false/*residual*/);

  }
}
