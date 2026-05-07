#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#ifndef USE_HIPDNN
namespace at {
namespace native {

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  TORCH_CHECK(false, "PyTorch was not compiled with hipDNN enabled!");
}

void run_cudnn_SDP_fprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  TORCH_CHECK(false, "PyTorch was not compiled with hipDNN enabled!");
}

void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  TORCH_CHECK(false, "PyTorch was not compiled with hipDNN enabled!");
}

void run_cudnn_SDP_bprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,

    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  TORCH_CHECK(false, "PyTorch was not compiled with hipDNN enabled!");
}

} // namespace native
} // namespace at

#else // USE_HIPDNN

// TODO: drop this define once hipDNN exposes SDPA unconditionally and
// pytorch's LoadHIP.cmake propagates hipdnn_frontend's
// INTERFACE_COMPILE_DEFINITIONS.
#define HIPDNN_ENABLE_SDPA
#include <ATen/hipdnn/Exceptions.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/hipdnn-wrapper.h>
#include <ATen/native/cudnn/MHA.h>
#include <ATen/native/transformers/sdp_utils.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/hip/HIPCachingAllocator.h>

namespace at::native {

namespace fe = hipdnn_frontend;

constexpr uint8_t MAX_MHA_DIM = 4;

// TODO: replace with shared hipDNN dtype util once one exists.
static fe::DataType_t to_fe_data_type(c10::ScalarType t) {
  switch (t) {
    case kHalf:
      return fe::DataType_t::HALF;
    case kBFloat16:
      return fe::DataType_t::BFLOAT16;
    case kFloat:
      return fe::DataType_t::FLOAT;
    case kDouble:
      return fe::DataType_t::DOUBLE;
    default:
      TORCH_CHECK(false, "hipDNN SDPA: unexpected tensor dtype ", t);
  }
}

static void check_tensor_matches_graph(
    const std::unordered_map<
        int64_t,
        std::shared_ptr<fe::graph::TensorAttributes>>& graph_tensors,
    int64_t uid,
    const Tensor& t) {
  auto it = graph_tensors.find(uid);
  TORCH_CHECK(it != graph_tensors.end());
  const auto& attrs = it->second;
  TORCH_CHECK(t.sizes() == IntArrayRef(attrs->get_dim()));
  TORCH_CHECK(t.strides() == IntArrayRef(attrs->get_stride()));
  TORCH_CHECK(to_fe_data_type(t.scalar_type()) == attrs->get_data_type());
}

struct MHAParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  float scaling_factor;
  std::array<int64_t, MAX_MHA_DIM> q_dim;
  std::array<int64_t, MAX_MHA_DIM> k_dim;
  std::array<int64_t, MAX_MHA_DIM> v_dim;
  std::array<int64_t, MAX_MHA_DIM> q_stride;
  std::array<int64_t, MAX_MHA_DIM> k_stride;
  std::array<int64_t, MAX_MHA_DIM> v_stride;
  std::array<int64_t, MAX_MHA_DIM> bias_dim;
  std::array<int64_t, MAX_MHA_DIM> bias_stride;
  int64_t b;
  int64_t h;
  int64_t s_q;
  int64_t s_kv;
  int64_t d_qk;
  int64_t d_v;
  double dropout_probability;
  bool is_causal;
  bool return_softmaxstats;
  bool has_attn_bias;
};

static void setMHAParams(
    MHAParams& params,
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    double dropout_probability,
    bool is_causal,
    bool return_softmaxstats) {
  memset(&params, 0, sizeof(MHAParams));
  params.device_id = at::cuda::current_device();
  params.dataType = to_fe_data_type(q.scalar_type());
  params.scaling_factor = scaling_factor;
  params.b = b;
  params.h = h;
  params.d_qk = d_qk;
  params.d_v = d_v;
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.dropout_probability = dropout_probability;
  params.is_causal = is_causal;
  params.return_softmaxstats = return_softmaxstats;
  params.has_attn_bias = attn_bias.has_value();
  std::copy(q.sizes().begin(), q.sizes().end(), params.q_dim.begin());
  std::copy(q.strides().begin(), q.strides().end(), params.q_stride.begin());
  std::copy(k.sizes().begin(), k.sizes().end(), params.k_dim.begin());
  std::copy(k.strides().begin(), k.strides().end(), params.k_stride.begin());
  std::copy(v.sizes().begin(), v.sizes().end(), params.v_dim.begin());
  std::copy(v.strides().begin(), v.strides().end(), params.v_stride.begin());
  // uninit is OK as the struct is memset 0'd
  if (params.has_attn_bias) {
    std::copy(
        attn_bias.value().sizes().begin(),
        attn_bias.value().sizes().end(),
        params.bias_dim.begin());
    std::copy(
        attn_bias.value().strides().begin(),
        attn_bias.value().strides().end(),
        params.bias_stride.begin());
  }
}

// TODO: lift to a shared hipDNN graph cache utility.
struct MHACacheKeyWrapper : ParamsWrapper<MHAParams> {
  MHACacheKeyWrapper(
      int64_t b,
      int64_t h,
      int64_t s_q,
      int64_t s_kv,
      int64_t d_qk,
      int64_t d_v,
      float scaling_factor,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      const std::optional<Tensor>& attn_bias,
      double dropout_probability,
      bool is_causal,
      bool return_softmaxstats) {
    setMHAParams(
        this->pod,
        b,
        h,
        s_q,
        s_kv,
        d_qk,
        d_v,
        scaling_factor,
        q,
        k,
        v,
        attn_bias,
        dropout_probability,
        is_causal,
        return_softmaxstats);
  }
};

struct MHAGraphCache {
  using KeyType = MHACacheKeyWrapper;
  using ValueType = std::unique_ptr<fe::graph::Graph>;
  using MapType =
      std::unordered_map<KeyType, ValueType, ParamsWrapperHash<KeyType>>;
  using iterator = typename MapType::iterator;
  using const_iterator = typename MapType::const_iterator;

  MapType engine_cache;
  int count = 0;
  int hits = 0;

  // No mutexes — the cache is thread-local (see getMHAGraphCache_).
  iterator find(const KeyType& key) {
    static bool flag =
        c10::utils::check_env("TORCH_CUDNN_SDPA_CACHE_DEBUG") == true;
    if (flag && count) {
      TORCH_WARN(
          "SDPA Cache Called ",
          count,
          " times. Hit rate: ",
          100 * hits / count,
          "%");
    }
    count++;
    auto it = engine_cache.find(key);
    if (it != engine_cache.end()) {
      hits++;
    }
    return it;
  }

  const_iterator end() const {
    return engine_cache.end();
  }

  template <typename... Args>
  std::pair<iterator, bool> try_emplace(const KeyType& key, Args&&... args) {
    return engine_cache.try_emplace(key, std::forward<Args>(args)...);
  }
};

// Use thread local caches to avoid potential thread safety issues.
static MHAGraphCache& getMHAGraphCache_() {
  thread_local MHAGraphCache instance;
  return instance;
}

static MHAGraphCache& getMHAGraphBackwardCache_() {
  thread_local MHAGraphCache instance;
  return instance;
}

namespace {

enum UIDS {
  Q,
  K,
  V,
  O,
  BIAS,
  SEED,
  OFFSET,
  LSE,
  DO,
  DQ,
  DK,
  DV,
};

} // namespace

static std::unique_ptr<fe::graph::Graph> build_graph_structure(
    const MHAParams& params) {
  auto mha_graph = std::make_unique<fe::graph::Graph>();
  mha_graph->set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto scaled_dot_product_flash_attention_options =
      fe::graph::SDPA_attributes()
          .set_name("HIPDNN_SDPA")
          .set_generate_stats(params.return_softmaxstats)
          .set_causal_mask(params.is_causal)
          .set_attn_scale_value(params.scaling_factor);
  if (params.dropout_probability != 0.0f) {
    // Seed/offset dtype is hardcoded INT64 here, to match the allocation done
    // in 'attention.cu'.
    auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_uid(SEED)
                                      .set_name("Seed")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT64));
    auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_uid(OFFSET)
                                        .set_name("Offset")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT64));
    scaled_dot_product_flash_attention_options.set_dropout(
        params.dropout_probability, seed, offset);
  }
  auto Q_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(Q)
          .set_name("Q")
          .set_dim(std::vector(params.q_dim.begin(), params.q_dim.end()))
          .set_stride(std::vector(params.q_stride.begin(), params.q_stride.end()))
          .set_data_type(params.dataType));
  auto K_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(K)
          .set_name("K")
          .set_dim(std::vector(params.k_dim.begin(), params.k_dim.end()))
          .set_stride(std::vector(params.k_stride.begin(), params.k_stride.end()))
          .set_data_type(params.dataType));
  auto V_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(V)
          .set_name("V")
          .set_dim(std::vector(params.v_dim.begin(), params.v_dim.end()))
          .set_stride(std::vector(params.v_stride.begin(), params.v_stride.end()))
          .set_data_type(params.dataType));
  if (params.has_attn_bias) {
    scaled_dot_product_flash_attention_options.set_bias(mha_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_uid(BIAS)
            .set_name("bias")
            .set_dim(std::vector(params.bias_dim.begin(), params.bias_dim.end()))
            .set_stride(std::vector(params.bias_stride.begin(), params.bias_stride.end()))
            .set_data_type(params.dataType)));
  }

  auto [O_, Stats] =
      mha_graph->sdpa(Q_, K_, V_, scaled_dot_product_flash_attention_options);

  // Output metadata here matches the allocation done in 'run_cudnn_SDP_fprop'.
  std::vector<int64_t> o_sizes = {params.b, params.h, params.s_q, params.d_v};
  auto o_strides = compute_matching_strides(params.q_dim, params.q_stride, o_sizes);
  O_->set_uid(O)
      .set_output(true)
      .set_data_type(params.dataType)
      .set_dim(o_sizes)
      .set_stride(o_strides);
  if (Stats) {
    Stats->set_uid(LSE)
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({params.b, params.h, params.s_q, 1})
        .set_stride({params.h * params.s_q, params.s_q, 1, 1});
  }

  return mha_graph;
}

static std::unique_ptr<fe::graph::Graph> build_graph(
    const MHAParams& params,
    hipdnnHandle_t& handle) {
  auto mha_graph = build_graph_structure(params);
  HIPDNN_FE_CHECK(mha_graph->validate());
  HIPDNN_FE_CHECK(mha_graph->build_operation_graph(handle));
  HIPDNN_FE_CHECK(mha_graph->create_execution_plans({fe::HeurMode_t::FALLBACK}));
  HIPDNN_FE_CHECK(mha_graph->check_support());
  HIPDNN_FE_CHECK(mha_graph->build_plans());
  return mha_graph;
}

static std::unique_ptr<fe::graph::Graph> build_graph_backward_structure(
    const MHAParams& params) {
  auto mha_graph = std::make_unique<fe::graph::Graph>();
  mha_graph->set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                   .set_name("HIPDNN_SDPA_BACKWARD")
                                   .set_causal_mask(params.is_causal)
                                   .set_attn_scale_value(params.scaling_factor);

  auto Q_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(Q)
          .set_name("Q")
          .set_dim(std::vector(params.q_dim.begin(), params.q_dim.end()))
          .set_stride(std::vector(params.q_stride.begin(), params.q_stride.end()))
          .set_data_type(params.dataType));
  auto K_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(K)
          .set_name("K")
          .set_dim(std::vector(params.k_dim.begin(), params.k_dim.end()))
          .set_stride(std::vector(params.k_stride.begin(), params.k_stride.end()))
          .set_data_type(params.dataType));
  auto V_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(V)
          .set_name("V")
          .set_dim(std::vector(params.v_dim.begin(), params.v_dim.end()))
          .set_stride(std::vector(params.v_stride.begin(), params.v_stride.end()))
          .set_data_type(params.dataType));
  if (params.has_attn_bias) {
    sdpa_backward_options.set_bias(mha_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_uid(BIAS)
            .set_name("bias")
            .set_dim(std::vector(params.bias_dim.begin(), params.bias_dim.end()))
            .set_stride(std::vector(params.bias_stride.begin(), params.bias_stride.end()))
            .set_data_type(params.dataType)));
  }

  // Metadata for seed/offset/O/stats are hardcoded here, based on the
  // allocation done during forward execution.
  if (params.dropout_probability != 0.0f) {
    auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_uid(SEED)
                                      .set_name("Seed")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT64));
    auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_uid(OFFSET)
                                        .set_name("Offset")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT64));
    sdpa_backward_options.set_dropout(params.dropout_probability, seed, offset);
  }

  std::vector<int64_t> o_sizes = {params.b, params.h, params.s_q, params.d_v};
  auto o_strides =
      compute_matching_strides(params.q_dim, params.q_stride, o_sizes);
  auto O_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(O)
          .set_name("O")
          .set_dim(o_sizes)
          .set_stride(o_strides)
          .set_data_type(params.dataType));
  auto Stats = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(LSE)
          .set_name("Stats")
          .set_dim({params.b, params.h, params.s_q, 1})
          .set_stride({params.h * params.s_q, params.s_q, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT));
  auto Do = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(DO)
          .set_name("DO")
          .set_dim(o_sizes)
          .set_stride(o_strides)
          .set_data_type(params.dataType));
  auto [Dq, Dk, Dv] = mha_graph->sdpa_backward(
      Q_, K_, V_, O_, Do, Stats, sdpa_backward_options);
  Dq->set_uid(DQ)
      .set_output(true)
      .set_data_type(params.dataType)
      .set_dim(std::vector(params.q_dim.begin(), params.q_dim.end()))
      .set_stride(std::vector(params.q_stride.begin(), params.q_stride.end()));
  Dk->set_uid(DK)
      .set_output(true)
      .set_data_type(params.dataType)
      .set_dim(std::vector(params.k_dim.begin(), params.k_dim.end()))
      .set_stride(std::vector(params.k_stride.begin(), params.k_stride.end()));
  Dv->set_uid(DV)
      .set_output(true)
      .set_data_type(params.dataType)
      .set_dim(std::vector(params.v_dim.begin(), params.v_dim.end()))
      .set_stride(std::vector(params.v_stride.begin(), params.v_stride.end()));

  return mha_graph;
}

static std::unique_ptr<fe::graph::Graph> build_graph_backward(
    const MHAParams& params,
    hipdnnHandle_t& handle) {
  auto mha_graph = build_graph_backward_structure(params);
  HIPDNN_FE_CHECK(mha_graph->validate());
  HIPDNN_FE_CHECK(mha_graph->build_operation_graph(handle));
  HIPDNN_FE_CHECK(mha_graph->create_execution_plans({fe::HeurMode_t::FALLBACK}));
  HIPDNN_FE_CHECK(mha_graph->check_support());
  HIPDNN_FE_CHECK(mha_graph->build_plans());
  return mha_graph;
}

// TODO: cache the support result (and ideally the built graph) so we don't
// rebuild it on every dispatch. Currently runs end-to-end graph build +
// query for every can_use_cudnn_attention() call.
bool check_cudnn_sdpa_support(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias) {
  std::optional<Tensor> expanded_bias;
  if (attn_bias.has_value()) {
    // At this point we have the caller's original attention mask tensor, before
    // rank normalization (from 'attention.cu:_cudnn_attention_forward'), so we
    // need to mirror that logic here to determine what the actual metadata will
    // be at execution time.
    const auto rank = attn_bias.value().dim();
    TORCH_CHECK(
        rank == 2 || rank == 3 || rank == 4,
        "hipDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ",
        rank,
        "D");
    const int64_t h_bias = rank == 4 ? attn_bias.value().size(1) : 1;
    expanded_bias = attn_bias.value().expand({b, h_bias, s_q, s_kv});
  }
  MHAParams fwd_params;
  setMHAParams(
      fwd_params,
      b,
      h,
      s_q,
      s_kv,
      d_qk,
      d_v,
      scaling_factor,
      q,
      k,
      v,
      expanded_bias,
      dropout_probability,
      is_causal,
      return_softmaxstats);

  hipdnnHandle_t handle = getHipdnnHandle();
  std::unique_ptr<fe::graph::Graph> fwd_graph =
      build_graph_structure(fwd_params);
  if (!fwd_graph->is_supported_ext(handle).is_good())
    return false;

  // Check that backwards is also supported here if it might be needed, since
  // we'll be expected to handle it too if we accept the forward pass.
  if (return_softmaxstats) {
    std::unique_ptr<fe::graph::Graph> bwd_graph =
        build_graph_backward_structure(fwd_params);
    if (!bwd_graph->is_supported_ext(handle).is_good())
      return false;
  }

  return true;
}

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  // do nothing if we got 0-element tensors
  if (!q.numel() || !k.numel() || !v.numel()) {
    return;
  }

  TORCH_CHECK(
      !o.defined(), "hipDNN SDPA expects output tensor to be undefined");
  // q is passed to us in BHSD dim order
  alloc_with_matching_layout(q, o, {b, h, s_q, d_v});
  TORCH_CHECK(
      !softmaxstats.defined(),
      "hipDNN SDPA expects softmaxstats tensor to be undefined");
  if (return_softmaxstats) {
    softmaxstats = at::empty({b, h, s_q, 1}, q.options().dtype(kFloat));
  }

  hipdnnHandle_t handle = getHipdnnHandle();

  MHACacheKeyWrapper key(
      b,
      h,
      s_q,
      s_kv,
      d_qk,
      d_v,
      scaling_factor,
      q,
      k,
      v,
      attn_bias,
      dropout_probability,
      is_causal,
      return_softmaxstats);
  auto [cache_it, _] = getMHAGraphCache_().try_emplace(key, nullptr);
  if (cache_it->second == nullptr) {
    cache_it->second = build_graph(key.pod, handle);
  }
  const fe::graph::Graph& mha_graph = *cache_it->second;
  // Graph construction makes some assumptions based on constraints checked
  // earlier. Validate they hold by comparing metadata against tensors now that
  // they're available.
  auto graph_tensors = mha_graph.getTensorsByUid();
  check_tensor_matches_graph(graph_tensors, Q, q);
  check_tensor_matches_graph(graph_tensors, K, k);
  check_tensor_matches_graph(graph_tensors, V, v);
  check_tensor_matches_graph(graph_tensors, O, o);
  if (return_softmaxstats) {
    check_tensor_matches_graph(graph_tensors, LSE, softmaxstats);
  }
  if (attn_bias.has_value()) {
    check_tensor_matches_graph(graph_tensors, BIAS, attn_bias.value());
  }
  if (dropout_probability != 0.0f) {
    check_tensor_matches_graph(graph_tensors, SEED, dropoutseed);
    check_tensor_matches_graph(graph_tensors, OFFSET, dropoutoffset);
  }

  std::unordered_map<int64_t, void*> variant_pack = {
      {Q, q.mutable_data_ptr()},
      {K, k.mutable_data_ptr()},
      {V, v.mutable_data_ptr()},
      {O, o.mutable_data_ptr()}};
  if (return_softmaxstats) {
    variant_pack[LSE] = softmaxstats.mutable_data_ptr();
  }
  if (attn_bias.has_value()) {
    variant_pack[BIAS] = attn_bias.value().mutable_data_ptr();
  }
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = dropoutseed.mutable_data_ptr();
    variant_pack[OFFSET] = dropoutoffset.mutable_data_ptr();
  }
  // TODO: lift workspace allocation + execute to a shared util that takes a
  // graph and variant_pack.
  int64_t workspace_size = 0;
  HIPDNN_FE_CHECK(mha_graph.get_workspace_size(workspace_size));
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(
      mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
}

void run_cudnn_SDP_fprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  TORCH_CHECK(false, "hipDNN SDPA does not support nested tensors");
}

void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  // do nothing if we got 0-element tensors
  if (!q.numel() || !k.numel() || !v.numel() || !o.numel() || !dO.numel() ||
      !softmaxstats.numel()) {
    return;
  }

  hipdnnHandle_t handle = getHipdnnHandle();
  MHACacheKeyWrapper key(
      b,
      h,
      s_q,
      s_kv,
      d_qk,
      d_v,
      scaling_factor,
      q,
      k,
      v,
      attn_bias,
      dropout_probability,
      is_causal,
      /*return_softmaxstats=*/true);
  auto [cache_it, _] = getMHAGraphBackwardCache_().try_emplace(key, nullptr);
  if (cache_it->second == nullptr) {
    cache_it->second = build_graph_backward(key.pod, handle);
  }
  const fe::graph::Graph& mha_graph = *cache_it->second;
  // Graph construction makes some assumptions based on constraints checked
  // earlier. Validate they hold by comparing metadata against tensors now that
  // they're available.
  auto graph_tensors = mha_graph.getTensorsByUid();
  check_tensor_matches_graph(graph_tensors, Q, q);
  check_tensor_matches_graph(graph_tensors, K, k);
  check_tensor_matches_graph(graph_tensors, V, v);
  check_tensor_matches_graph(graph_tensors, O, o);
  check_tensor_matches_graph(graph_tensors, DO, dO);
  check_tensor_matches_graph(graph_tensors, LSE, softmaxstats);
  check_tensor_matches_graph(graph_tensors, DQ, dQ);
  check_tensor_matches_graph(graph_tensors, DK, dK);
  check_tensor_matches_graph(graph_tensors, DV, dV);
  if (attn_bias.has_value()) {
    check_tensor_matches_graph(graph_tensors, BIAS, attn_bias.value());
  }
  if (dropout_probability != 0.0f) {
    check_tensor_matches_graph(graph_tensors, SEED, dropoutseed);
    check_tensor_matches_graph(graph_tensors, OFFSET, dropoutoffset);
  }

  std::unordered_map<int64_t, void*> variant_pack = {
      // inputs
      {Q, q.mutable_data_ptr()},
      {K, k.mutable_data_ptr()},
      {V, v.mutable_data_ptr()},
      {O, o.mutable_data_ptr()},
      {DO, dO.mutable_data_ptr()},
      {LSE, softmaxstats.mutable_data_ptr()},
      // outputs
      {DQ, dQ.mutable_data_ptr()},
      {DK, dK.mutable_data_ptr()},
      {DV, dV.mutable_data_ptr()}};
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = dropoutseed.mutable_data_ptr();
    variant_pack[OFFSET] = dropoutoffset.mutable_data_ptr();
  }
  if (attn_bias.has_value()) {
    variant_pack[BIAS] = attn_bias.value().mutable_data_ptr();
  }

  int64_t workspace_size;
  HIPDNN_FE_CHECK(mha_graph.get_workspace_size(workspace_size));
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  TORCH_CHECK(
      mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
}

void run_cudnn_SDP_bprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  TORCH_CHECK(false, "hipDNN SDPA does not support nested tensors");
}

} // namespace at::native

#endif // USE_HIPDNN
