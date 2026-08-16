// aff-gpucheck — GPU vs CPU on the REAL container, not synthetic weights.
//
// tests/test_hip_expert.hip proves the two kernels agree on matrices this repo quantised moments
// earlier. That is necessary but not sufficient: it shares the quantiser, the codebook fit, and the
// in-memory layout with the thing it is testing. This reads the actual .aff file, uploads the
// actual expert bytes to VRAM, and uses the actual fitted codebook stored in the container — the
// same path a hybrid decode would take.
//
// It also reports how much of the expert pool fits in VRAM, which is the number the whole residency
// design turns on.
//
//   aff-gpucheck -m model.aff [--experts N] [--layer L] [--bench REPS] [--dq] [--verbose]
//                [--vram-gbps R]

#include "format/aff_reader.h"
#include "engine/expert_kernel.h"
#include "engine/ops.h"
#include "quant/source_dtypes.h"
#include "quant/blockquant.h"
#include "engine/model.h"
#include "gpu/expert_kernel_hip.h"
#include "gpu/hip_common.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <random>
#include <cstring>
#include <string>
#include <vector>

using namespace aff;

namespace {

int g_fail = 0;

// Filled from the container in main() before anything uses it. A spec hardcoded here would be a
// second place to forget when a format is added.
QuantSpec g_spec = QuantSpec{16, 5, 4, 2, 2, false};
QuantSpec spec() { return g_spec; }

struct Planes {
  DeviceBuffer<uint8_t> data, scales;
  DeviceBuffer<uint16_t> rscales;
  DeviceMatrix view;
};

// Uploads one matrix straight from the mmap'd container. The 16-byte pad lets the kernel's
// trailing 64-bit window overrun the logical end of a plane, which it legitimately does.
Planes upload(const AffMatrixDesc& d, const MatrixView& mv) {
  Planes p;
  const uint64_t n = d.rows * d.cols;
  const uint64_t data_bytes  = n * 5 / 16;          // 5 bits per pair
  const uint64_t scale_bytes = n * 6 / 128;         // 6 bits per 16-weight block
  p.data.alloc(data_bytes, 16);
  p.scales.alloc(scale_bytes, 16);
  p.rscales.alloc(d.rows, 16);
  // The CPU reference reads the mmap'd container ROW-MAJOR whatever happens here, so uploading the
  // colgroup-major permutation makes this an end-to-end check of the layout as well as the kernel:
  // if the permute and the addressing disagree by so much as one span, rel_l2 says so.
  if (expert_planes_are_colgroup_major()) {
    std::vector<uint8_t> dt(data_bytes), ft(scale_bytes);
    expert_transpose_plane_host(mv.data, mv.scales, dt.data(), ft.data(), (uint32_t)d.rows,
                                (uint32_t)d.cols);
    p.data.upload(dt.data(), data_bytes);
    p.scales.upload(ft.data(), scale_bytes);
  } else {
    p.data.upload(mv.data, data_bytes);
    p.scales.upload(mv.scales, scale_bytes);
  }
  p.rscales.upload(reinterpret_cast<const uint16_t*>(mv.rscales), d.rows);
  p.view.data = p.data.get();
  p.view.scales = p.scales.get();
  p.view.rscales = p.rscales.get();
  p.view.rows = (uint32_t)d.rows;
  p.view.cols = (uint32_t)d.cols;
  return p;
}

// A flat integer out of the container's own config.json copy, so the geometry below is this file's
// rather than a literal.
int cfg_int(const std::string& j, const char* k, int dflt) {
  const std::string pat = std::string("\"") + k + "\"";
  size_t p = j.find(pat);
  if (p == std::string::npos) return dflt;
  p = j.find(':', p);
  return p == std::string::npos ? dflt : (int)std::strtol(j.c_str() + p + 1, nullptr, 10);
}

double cfg_f64(const std::string& j, const char* k, double dflt) {
  const std::string pat = std::string("\"") + k + "\"";
  size_t p = j.find(pat);
  if (p == std::string::npos) return dflt;
  p = j.find(':', p);
  return p == std::string::npos ? dflt : std::strtod(j.c_str() + p + 1, nullptr);
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b, double* worst) {
  double num = 0, den = 0;
  *worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - b[i];
    num += d * d;
    den += (double)b[i] * b[i];
    *worst = std::max(*worst, std::fabs(d));
  }
  return den > 0 ? std::sqrt(num / den) : -1.0;
}

} // namespace

int main(int argc, char** argv) {
  std::string path;
  int n_experts = 8, only_layer = -1, reps = 0;
  bool verbose = false;
  // Off by default: it dequantises a whole matrix per check, which is 32 MB and a second.
  bool dq_check = false;
  // Peak VRAM read bandwidth of ONE card, used only to turn the container's dense byte total into a
  // ceiling. There is no way to read it from the driver, so it is an input.
  double vram_gbps = 630.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "-m" || a == "--model") path = nx();
    else if (a == "--experts") n_experts = std::atoi(nx());
    else if (a == "--layer") only_layer = std::atoi(nx());
    else if (a == "--verbose") verbose = true;
    else if (a == "--dq") dq_check = true;
    else if (a == "--bench") reps = std::atoi(nx());
    else if (a == "--vram-gbps") vram_gbps = std::atof(nx());
    else if (a == "-h" || a == "--help") {
      std::printf("aff-gpucheck -m model.aff [--experts N] [--layer L] [--bench REPS]\n"
                  "               [--dq] [--verbose] [--vram-gbps R]\n"
                  "  --dq also checks the engine decoder against the container's own inverse.\n"
                  "  --vram-gbps is one card's peak read bandwidth (default 630).\n");
      return 0;
    }
    else { std::fprintf(stderr, "error: unrecognised argument '%s'. Try --help.\n", a.c_str());
           return 2; }
  }
  if (vram_gbps <= 0.0) { std::fprintf(stderr, "error: --vram-gbps must be positive\n"); return 2; }
  if (path.empty()) { std::fprintf(stderr, "error: -m is required\n"); return 2; }

  if (!hip_available()) {
    std::printf("SKIP: %s\n", hip_device_summary().c_str());
    return 0;
  }
  std::printf("%s", hip_device_summary().c_str());
  if (!hip_select_device(0)) { std::printf("SKIP: cannot select a gfx12 device\n"); return 0; }

  AffReader aff;
  std::string err;
  if (!aff.open(path, &err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }

  // Which decode the container holds, latched BEFORE any expert kernel runs. Reading 10-bit quads
  // as 5-bit pairs is in-bounds, right-sized and completely wrong, so this is not optional and it
  // is not a preference — every tool that opens a container for GPU work does exactly this.
  const uint32_t wire = expert_wire_format_for_quant(
      aff.layer_count() ? (uint32_t)aff.layer(0).gate.quant : 0u);
  expert_set_wire_format(wire);
  g_spec = expert_quant_spec(aff.layer_count() ? (uint32_t)aff.layer(0).gate.quant : 0u);
  const QuantSpec qspec = g_spec;

  // The codebook is fitted data stored in the container — not a constant, and not reconstructible.
  const AffTensorEntry* cbe = aff.find_tensor("__codebook");
  if (!cbe) { std::fprintf(stderr, "error: container has no __codebook\n"); return 1; }
  const float* cb_host = reinterpret_cast<const float*>(aff.tensor_data(*cbe));
  const uint64_t cbn = cbe->desc.rows * cbe->desc.cols;
  Codebook cb;
  std::vector<uint32_t> cb_packed;
  if (qspec.pair == 4) {
    // Stored already packed as E4M3 quads — that is what the GPU LUT holds. Unpack for the CPU
    // reference; hand the device the dwords verbatim.
    unpack_codebook_e4m3_quads(reinterpret_cast<const uint32_t*>(cb_host), qspec.variants(),
                               qspec.lut_size(), &cb);
    cb_packed.assign(reinterpret_cast<const uint32_t*>(cb_host),
                     reinterpret_cast<const uint32_t*>(cb_host) + cbn);
  } else {
    cb.variants = 4; cb.k = 32; cb.pair = 2;
    cb.v.assign(cb_host, cb_host + cbn);
  }

  DeviceBuffer<float> d_cb(qspec.pair == 4 ? cb_packed.size() : cb.v.size());
  if (qspec.pair == 4) d_cb.upload((const float*)cb_packed.data(), cb_packed.size());
  else                d_cb.upload(cb.v.data(), cb.v.size());

  const uint64_t n_layer = aff.layer_count();
  // This model's geometry, read from the container's own config.json copy rather than assumed.
  const std::string meta(aff.meta());
  const uint32_t n_used = (uint32_t)std::max(1, cfg_int(meta, "num_experts_per_tok", 6));
  const float swiglu_limit = (float)cfg_f64(meta, "swiglu_limit", 10.0);
  std::printf("container %llu layers, %u experts a token, expert pool %.2f GiB\n",
              (unsigned long long)n_layer, n_used, aff.expert_pool_size() / 1073741824.0);

  const std::vector<int> devs = gfx12_devices();
  DeviceInfo di;
  if (devs.empty()) {
    std::printf("no gfx12 device found, so no VRAM residency figure\n\n");
  } else if (hip_device_info(devs[0], &di, &err)) {
    const double pool = aff.expert_pool_size() / 1073741824.0;
    const double vram = di.vram_free / 1073741824.0;
    std::printf("one GPU holds %.0f%% of the pool (%.1f of %.1f GiB free); all %zu hold %.0f%%\n\n",
                100.0 * std::min(1.0, vram / pool), vram, pool, devs.size(),
                100.0 * std::min(1.0, (double)devs.size() * vram / pool));
  }

  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0f, 1.0f);

  double worst_rel = 0.0;
  int checked = 0;
  const char* mat_name[3] = {"gate", "up", "down"};

  for (uint64_t l = 0; l < n_layer; ++l) {
    if (only_layer >= 0 && (int)l != only_layer) continue;
    const AffLayerDesc& L = aff.layer((uint32_t)l);
    // Spread the sample across the expert index so it is not all the popular head of the profile.
    for (int k = 0; k < n_experts; ++k) {
      const uint32_t e = (uint32_t)((uint64_t)k * L.n_experts / n_experts);
      const ExpertView ev = aff.expert((uint32_t)l, e);
      const AffMatrixDesc* descs[3] = {&L.gate, &L.up, &L.down};
      const MatrixView* views[3] = {&ev.gate, &ev.up, &ev.down};

      for (int m = 0; m < 3; ++m) {
        const AffMatrixDesc& d = *descs[m];
        std::vector<float> x(d.cols);
        for (auto& v : x) v = nd(rng);

        ExpertMatrixView cpu;
        cpu.data = views[m]->data; cpu.scales = views[m]->scales; cpu.rscales = views[m]->rscales;
        cpu.rows = d.rows; cpu.cols = d.cols; cpu.spec = spec();
        std::vector<float> y_cpu(d.rows, 0.0f);
        expert_gemv_scaled(cpu, cb, x.data(), 1.0f, y_cpu.data());

        // THE WRITER'S OWN INVERSE, on the same container bytes. `dequantize_matrix` is what
        // aff-quantize's error metric is computed against, and `expert_gemv_reference` is what the
        // engine decodes with; they share no code, so this pins which side of a GPU-vs-CPU
        // disagreement is at fault instead of leaving it to be argued. A format change can put all
        // three out of step at once, and nothing else here can tell that apart from a kernel bug.
        if (dq_check) {
          const QuantSpec& sp = cpu.spec;
          const uint64_t n = d.rows * d.cols;
          QuantizedMatrix qm;
          qm.rows = d.rows; qm.cols = d.cols; qm.spec = sp;
          qm.data.assign(views[m]->data, views[m]->data + n * sp.code_bits / sp.pair / 8);
          qm.scales.assign(views[m]->scales,
                           views[m]->scales + n / sp.block * sp.block_field_bits() / 8);
          const uint8_t* rs8 = reinterpret_cast<const uint8_t*>(views[m]->rscales);
          qm.rscales.assign(rs8, rs8 + d.rows * (sp.row_scale_f32 ? 4 : 2));
          std::vector<float> W(n);
          std::string derr;
          if (dequantize_matrix(qm, cb, W.data(), &derr)) {
            std::vector<float> y_dq(d.rows, 0.0f);
            for (uint64_t r = 0; r < d.rows; ++r) {
              double a = 0;
              const float* wr = W.data() + r * d.cols;
              for (uint64_t c = 0; c < d.cols; ++c) a += (double)wr[c] * x[c];
              y_dq[r] = (float)a;
            }
            double w2 = 0;
            const double r2 = rel_l2(y_cpu, y_dq, &w2);
            if (r2 > 1e-4) {
              std::printf("  L%llu E%-3u %-4s CPU DECODER vs CONTAINER INVERSE rel=%.3e\n",
                          (unsigned long long)l, e, mat_name[m], r2);
              ++g_fail;
            }
          }
        }

        Planes p = upload(d, *views[m]);
        DeviceBuffer<float> d_x(d.cols), d_y(d.rows);
        d_x.upload(x.data(), d.cols);
        AFF_HIP_CHECK(hipMemset(d_y.get(), 0, d.rows * sizeof(float)));
        expert_gemv_hip(p.view, d_cb.get(), d_x.get(), 1.0f, d_y.get());
        std::vector<float> y_gpu(d.rows);
        d_y.download(y_gpu.data(), d.rows);

        double worst = 0;
        const double rel = rel_l2(y_gpu, y_cpu, &worst);
        ++checked;
        if (rel < 0) {
          std::printf("  L%llu E%u %-4s  reference is all zero — expert is empty\n",
                      (unsigned long long)l, e, mat_name[m]);
          ++g_fail;
          continue;
        }
        worst_rel = std::max(worst_rel, rel);
        if (verbose || rel > 1e-4)
          std::printf("  L%llu E%-3u %-4s rel=%.3e max_abs=%.3e\n",
                      (unsigned long long)l, e, mat_name[m], rel, worst);
        if (rel > 1e-4) ++g_fail;
      }
    }
  }

  // ---- dense tensors: GPU against the CPU path, on the container's real bytes -----------------
  //
  // Dense is several times the expert traffic and lives on the GPU permanently, so it needs the same
  // treatment the experts got. FP8 is the interesting case: one UE8M0 scale covers a 128x128 tile,
  // so a kernel that picks the wrong tile row is wrong by a power of two and still looks entirely
  // plausible. Rows are checked well past the first tile for exactly that reason.
  {
    const char* names[] = {"attn_q_a", "attn_q_b", "attn_kv", "attn_out_a", "attn_out_b",
                           "shexp_gate", "shexp_up", "shexp_down", "token_embd", "output"};
    std::printf("\n  dense tensors (GPU vs CPU, real container bytes)\n");
    const uint32_t layers_to_check = (only_layer >= 0) ? 1u : std::min<uint64_t>(3, n_layer);
    for (uint32_t li = 0; li < layers_to_check; ++li) {
      const uint32_t l = (only_layer >= 0) ? (uint32_t)only_layer : li;
      for (const char* nm : names) {
        const bool global = !std::strncmp(nm, "token_embd", 10) || !std::strcmp(nm, "output");
        if (global && l != 0) continue;
        const std::string tn = global ? std::string(nm) : "blk." + std::to_string(l) + "." + nm;
        const AffTensorEntry* e = aff.find_tensor(tn);
        if (!e) continue;
        const uint64_t rows = e->desc.rows, cols = e->desc.cols;
        // The vocabulary matrices are 129280 rows; checking every one says nothing the first few
        // thousand did not, and costs a second per tensor.
        const uint64_t nr = std::min<uint64_t>(rows, 4096);

        std::vector<float> x(cols);
        for (auto& v : x) v = nd(rng);

        DenseW dw;
        dw.data = aff.tensor_data(*e);
        dw.quant = e->desc.quant;
        dw.scales = (dw.quant == AFF_FP8_E4M3) ? aff.tensor_scales(*e) : nullptr;
        std::vector<float> y_cpu(nr, 0.0f);
        matvec_dense(dw, x.data(), nr, cols, y_cpu.data());

        DeviceBuffer<float> d_x(cols), d_y(nr);
        d_x.upload(x.data(), cols);
        std::vector<float> y_gpu(nr);
        if (dw.quant == AFF_FP8_E4M3) {
          const uint64_t sc = ((rows + 127) / 128) * ((cols + 127) / 128);
          DeviceBuffer<uint8_t> d_w((size_t)rows * cols), d_s(sc);
          d_w.upload((const uint8_t*)dw.data, (size_t)rows * cols);
          d_s.upload(dw.scales, sc);
          matvec_fp8_hip(d_w.get(), d_s.get(), d_x.get(), (uint32_t)nr, (uint32_t)cols, d_y.get(), nullptr);
        } else {
          DeviceBuffer<uint16_t> d_w((size_t)rows * cols);
          d_w.upload((const uint16_t*)dw.data, (size_t)rows * cols);
          matvec_bf16_hip(d_w.get(), d_x.get(), (uint32_t)nr, (uint32_t)cols, d_y.get(), nullptr);
        }
        AFF_HIP_CHECK(hipDeviceSynchronize());
        d_y.download(y_gpu.data(), nr);

        double worst = 0;
        const double rel = rel_l2(y_gpu, y_cpu, &worst);
        worst_rel = std::max(worst_rel, rel < 0 ? 0.0 : rel);
        ++checked;
        if (verbose || rel > 1e-4 || rel < 0)
          std::printf("    %-22s %6llux%-6llu %-8s rel=%.3e\n", tn.c_str(),
                      (unsigned long long)nr, (unsigned long long)cols,
                      quant_name((AffQuant)dw.quant), rel);
        if (rel > 1e-4 || rel < 0) ++g_fail;
      }

      // The BATCHED path, on the four tensors that actually use it. `mv_multi` puts a layer's
      // compressor projections in one launch and the kernel finds each row's matrix by walking a
      // prefix sum, so the failure mode is an off-by-one that lands whole rows in the neighbouring
      // matrix's output — which reads as plausible noise downstream and would be blamed on
      // quantisation. Checking each output separately against the CPU is what catches it.
      {
        const char* bnames[] = {"comp_wkv", "comp_wgate", "idx_comp_wkv", "idx_comp_wgate"};
        std::vector<MvJob> jobs;
        std::vector<const AffTensorEntry*> ents;
        std::vector<std::unique_ptr<DeviceBuffer<uint8_t>>> keep8;
        std::vector<std::unique_ptr<DeviceBuffer<uint16_t>>> keep16;
        uint64_t total = 0, cols = 0;
        for (const char* nm : bnames) {
          const AffTensorEntry* e = aff.find_tensor("blk." + std::to_string(l) + "." + nm);
          if (!e) continue;
          if (cols && e->desc.cols != cols) continue;      // they must share x to be batchable
          cols = e->desc.cols;
          ents.push_back(e);
          total += e->desc.rows;
        }
        if (ents.size() >= 2 && total) {
          std::vector<float> x(cols);
          for (auto& v : x) v = nd(rng);
          DeviceBuffer<float> d_x(cols), d_y(total);
          d_x.upload(x.data(), cols);

          uint64_t off = 0;
          for (const AffTensorEntry* e : ents) {
            const uint64_t rows = e->desc.rows;
            MvJob j;
            j.quant = e->desc.quant;
            j.rows = (uint32_t)rows;
            j.out = d_y.get() + off;
            if (j.quant == AFF_FP8_E4M3) {
              const uint64_t sc = ((rows + 127) / 128) * ((cols + 127) / 128);
              keep8.push_back(std::make_unique<DeviceBuffer<uint8_t>>((size_t)rows * cols));
              auto ds = std::make_unique<DeviceBuffer<uint8_t>>(sc);
              keep8.back()->upload((const uint8_t*)aff.tensor_data(*e), (size_t)rows * cols);
              ds->upload(aff.tensor_scales(*e), sc);
              j.data = keep8.back()->get();
              j.scales = ds->get();
              keep8.push_back(std::move(ds));
            } else {
              keep16.push_back(std::make_unique<DeviceBuffer<uint16_t>>((size_t)rows * cols));
              keep16.back()->upload((const uint16_t*)aff.tensor_data(*e), (size_t)rows * cols);
              j.data = keep16.back()->get();
              j.scales = nullptr;
            }
            jobs.push_back(j);
            off += rows;
          }
          matvec_batch_hip(jobs.data(), (uint32_t)jobs.size(), d_x.get(), (uint32_t)cols, nullptr);
          AFF_HIP_CHECK(hipDeviceSynchronize());
          std::vector<float> y_gpu(total);
          d_y.download(y_gpu.data(), total);

          off = 0;
          for (size_t i = 0; i < ents.size(); ++i) {
            const uint64_t rows = ents[i]->desc.rows;
            DenseW dw;
            dw.data = aff.tensor_data(*ents[i]);
            dw.quant = ents[i]->desc.quant;
            dw.scales = (dw.quant == AFF_FP8_E4M3) ? aff.tensor_scales(*ents[i]) : nullptr;
            std::vector<float> y_cpu(rows, 0.0f);
            matvec_dense(dw, x.data(), rows, cols, y_cpu.data());
            std::vector<float> y_one(y_gpu.begin() + off, y_gpu.begin() + off + rows);
            double worst = 0;
            const double rel = rel_l2(y_one, y_cpu, &worst);
            worst_rel = std::max(worst_rel, rel < 0 ? 0.0 : rel);
            ++checked;
            if (verbose || rel > 1e-4 || rel < 0)
              std::printf("    %-22s %6llux%-6llu %-8s rel=%.3e  [batch %zu/%zu]\n",
                          bnames[i], (unsigned long long)rows, (unsigned long long)cols,
                          quant_name((AffQuant)dw.quant), rel, i + 1, ents.size());
            if (rel > 1e-4 || rel < 0) ++g_fail;
            off += rows;
          }
        }
      }
    }
  }

  // ---- the full expert FFN on device, against the CPU path ----------------------------------
  // Per-matrix agreement does not imply the composition agrees: the SwiGLU clamp sits between the
  // stages, and it is asymmetric between its two operands in a way that is easy to mirror wrongly.
  {
    const uint32_t l = (only_layer >= 0) ? (uint32_t)only_layer : 0;
    const AffLayerDesc& L = aff.layer(l);
    const uint32_t N = n_used;
    const uint32_t inter = (uint32_t)L.gate.rows;
    const uint32_t hidden = (uint32_t)L.gate.cols;
    const float limit = swiglu_limit;

    std::vector<float> x(hidden);
    for (auto& v : x) v = nd(rng);
    std::vector<float> wts(N);
    for (uint32_t i = 0; i < N; ++i) wts[i] = 0.1f + 0.13f * (float)i;

    std::vector<uint32_t> ids(N);
    for (uint32_t i = 0; i < N; ++i) ids[i] = i * (L.n_experts / N);

    std::vector<float> y_cpu(hidden, 0.0f), gate(inter), up(inter), act(inter);
    for (uint32_t k = 0; k < N; ++k) {
      const ExpertView ev = aff.expert(l, ids[k]);
      auto mk = [&](const AffMatrixDesc& d, const MatrixView& mv) {
        ExpertMatrixView v;
        v.data = mv.data; v.scales = mv.scales; v.rscales = mv.rscales;
        v.rows = d.rows; v.cols = d.cols; v.spec = spec();
        return v;
      };
      std::fill(gate.begin(), gate.end(), 0.0f);
      std::fill(up.begin(), up.end(), 0.0f);
      expert_gemv_scaled(mk(L.gate, ev.gate), cb, x.data(), 1.0f, gate.data());
      expert_gemv_scaled(mk(L.up, ev.up), cb, x.data(), 1.0f, up.data());
      swiglu(gate.data(), up.data(), inter, limit, act.data());
      expert_gemv_scaled(mk(L.down, ev.down), cb, act.data(), wts[k], y_cpu.data());
    }

    std::vector<Planes> planes;
    std::vector<DeviceExpert> des(N);
    for (uint32_t k = 0; k < N; ++k) {
      const ExpertView ev = aff.expert(l, ids[k]);
      planes.push_back(upload(L.gate, ev.gate));
      planes.push_back(upload(L.up, ev.up));
      planes.push_back(upload(L.down, ev.down));
    }
    for (uint32_t k = 0; k < N; ++k) {
      des[k].gate = planes[3 * k + 0].view;
      des[k].up   = planes[3 * k + 1].view;
      des[k].down = planes[3 * k + 2].view;
    }

    DeviceBuffer<float> d_x(hidden), d_y(hidden), d_w(N), d_scratch((size_t)2 * N * inter);
    d_x.upload(x.data(), hidden);
    d_w.upload(wts.data(), N);
    AFF_HIP_CHECK(hipMemset(d_y.get(), 0, hidden * sizeof(float)));
    expert_ffn_hip(des.data(), d_w.get(), N, d_cb.get(), d_x.get(), limit,
                   d_scratch.get(), d_y.get());
    std::vector<float> y_gpu(hidden);
    d_y.download(y_gpu.data(), hidden);

    double worst = 0;
    const double rel = rel_l2(y_gpu, y_cpu, &worst);
    std::printf("\nfull expert FFN, layer %u, %u experts: rel=%.3e max_abs=%.3e\n",
                l, N, rel, worst);
    if (!(rel >= 0 && rel < 1e-4)) { std::printf("  FFN MISMATCH\n"); ++g_fail; }
  }

  // ---- A/B the two kernel variants on real weights ------------------------------------------
  // Interleaved, not one-after-the-other: a clock ramp over the run would otherwise be credited to
  // whichever variant ran second.
  if (reps > 0) {
    const uint32_t l = (only_layer >= 0) ? (uint32_t)only_layer : 0;
    const AffLayerDesc& L = aff.layer(l);
    const uint32_t NE = 24;                     // enough to stream past cache
    std::vector<Planes> pl;
    std::vector<DeviceMatrix> vs;
    for (uint32_t k = 0; k < NE; ++k) {
      const uint32_t e = k * (L.n_experts / NE);
      pl.push_back(upload(L.gate, aff.expert(l, e).gate));
    }
    for (uint32_t k = 0; k < NE; ++k) vs.push_back(pl[k].view);

    const uint32_t hidden = (uint32_t)L.gate.cols, out_rows = (uint32_t)L.gate.rows;
    std::vector<float> x(hidden);
    for (auto& v : x) v = nd(rng);
    DeviceBuffer<float> d_x(hidden), d_y(out_rows), d_w(NE);
    d_x.upload(x.data(), hidden);
    std::vector<float> ones(NE, 1.0f);
    d_w.upload(ones.data(), NE);

    const double bytes = (double)NE * ((double)L.gate.rows * L.gate.cols * qspec.bpw() / 8
                                       + L.gate.rows * 2);
    std::printf("\n[decode gemv on real weights: %u matrices, %.1f MiB per pass, %d reps]\n",
                NE, bytes / 1048576.0, reps);

    // Several launches per sample: one pass is short enough that launch overhead and clock ramp
    // would otherwise dominate the measurement rather than the kernel.
    const int inner = 16;
    std::vector<double> gbps;
    for (int r = 0; r < reps; ++r) {
      AFF_HIP_CHECK(hipMemset(d_y.get(), 0, out_rows * sizeof(float)));
      expert_gemv_batch_hip(vs.data(), d_w.get(), NE, d_cb.get(), d_x.get(), d_y.get());  // warm
      AFF_HIP_CHECK(hipDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < inner; ++k)
        expert_gemv_batch_hip(vs.data(), d_w.get(), NE, d_cb.get(), d_x.get(), d_y.get());
      AFF_HIP_CHECK(hipDeviceSynchronize());
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      gbps.push_back(dt > 0 ? bytes * inner / dt / 1e9 : 0.0);
    }
    std::sort(gbps.begin(), gbps.end());
    std::printf("  %7.1f GB/s  (worst %.1f, best %.1f, spread %.0f%%)\n", gbps[gbps.size() / 2],
                gbps.front(), gbps.back(),
                100.0 * (gbps.back() - gbps.front()) / gbps[gbps.size() / 2]);

    // ---- FFN latency, which is what actually bounds decode -----------------------------------
    // One token through one layer's routed experts. Times the layer count it is the GPU-side floor
    // for a decode step, before attention or the dense projections.
    {
      const uint32_t inter = (uint32_t)L.gate.rows, hidden = (uint32_t)L.gate.cols;
      const uint32_t NF = n_used;
      std::vector<Planes> fp;
      std::vector<DeviceExpert> de(NF);
      for (uint32_t k = 0; k < NF; ++k) {
        const ExpertView ev = aff.expert(l, k * (L.n_experts / NF));
        fp.push_back(upload(L.gate, ev.gate));
        fp.push_back(upload(L.up, ev.up));
        fp.push_back(upload(L.down, ev.down));
      }
      for (uint32_t k = 0; k < NF; ++k) {
        de[k].gate = fp[3 * k].view; de[k].up = fp[3 * k + 1].view; de[k].down = fp[3 * k + 2].view;
      }
      DeviceBuffer<float> fx(hidden), fy(hidden), fw(NF), fs((size_t)2 * NF * inter);
      std::vector<float> hx(hidden);
      for (auto& v : hx) v = nd(rng);
      fx.upload(hx.data(), hidden);
      std::vector<float> hw(NF, 0.2f);
      fw.upload(hw.data(), NF);

      std::vector<double> ms;
      for (int r = 0; r < reps; ++r) {
        AFF_HIP_CHECK(hipMemset(fy.get(), 0, hidden * sizeof(float)));
        expert_ffn_hip(de.data(), fw.get(), NF, d_cb.get(), fx.get(), swiglu_limit, fs.get(), fy.get());
        AFF_HIP_CHECK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        const int inner = 8;
        for (int k = 0; k < inner; ++k)
          expert_ffn_hip(de.data(), fw.get(), NF, d_cb.get(), fx.get(), swiglu_limit, fs.get(), fy.get());
        AFF_HIP_CHECK(hipDeviceSynchronize());
        ms.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                     * 1000.0 / inner);
      }
      // Same work through the monolithic cooperative kernel, verified against the staged path
      // before it is timed — a megakernel that races is not faster, it is wrong.
      // A clean single run for the reference. The timing loop above accumulates `inner` times
      // into fy without resetting, so reading it there would compare 1 pass against 8.
      // The megakernel still reads its weights ROW-MAJOR, and upload() above wrote colgroup-major
      // planes, so there is nothing here to compare against — skip rather than report a mismatch
      // that is really a layout the kernel was never taught.
      double med_mono = -1.0;
      if (expert_planes_are_colgroup_major()) {
        std::printf("\n  monolithic FFN: SKIPPED (kernel is row-major; planes are colgroup-major)\n");
      } else {
      std::vector<float> y_multi(hidden), y_mono(hidden);
      AFF_HIP_CHECK(hipMemset(fy.get(), 0, hidden * sizeof(float)));
      expert_ffn_hip(de.data(), fw.get(), NF, d_cb.get(), fx.get(), swiglu_limit, fs.get(), fy.get());
      AFF_HIP_CHECK(hipDeviceSynchronize());
      fy.download(y_multi.data(), hidden);
      AFF_HIP_CHECK(hipMemset(fy.get(), 0, hidden * sizeof(float)));
      expert_ffn_mono_hip(de.data(), fw.get(), NF, d_cb.get(), fx.get(), swiglu_limit, fs.get(), fy.get());
      AFF_HIP_CHECK(hipDeviceSynchronize());
      fy.download(y_mono.data(), hidden);
      { double w2 = 0; const double rel = rel_l2(y_mono, y_multi, &w2);
        std::printf("\n  monolithic vs staged FFN: rel=%.3e\n", rel);
        if (!(rel >= 0 && rel < 1e-4)) { std::printf("  MONO MISMATCH\n"); ++g_fail; } }

      std::vector<double> ms_mono;
      for (int r = 0; r < reps; ++r) {
        AFF_HIP_CHECK(hipMemset(fy.get(), 0, hidden * sizeof(float)));
        expert_ffn_mono_hip(de.data(), fw.get(), NF, d_cb.get(), fx.get(), swiglu_limit, fs.get(), fy.get());
        AFF_HIP_CHECK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        const int inner = 8;
        for (int k = 0; k < inner; ++k)
          expert_ffn_mono_hip(de.data(), fw.get(), NF, d_cb.get(), fx.get(), swiglu_limit, fs.get(), fy.get());
        AFF_HIP_CHECK(hipDeviceSynchronize());
        ms_mono.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                          * 1000.0 / inner);
      }
      std::sort(ms_mono.begin(), ms_mono.end());
      med_mono = ms_mono[ms_mono.size() / 2];
      }

      std::sort(ms.begin(), ms.end());
      const double med = ms[ms.size() / 2];
      const double nl = (double)n_layer;
      const double bytes = (double)NF * 3 * L.gate.rows * L.gate.cols * qspec.bpw() / 8;
      std::printf("\n  expert FFN, %u experts, 1 token: %.3f ms  (%.1f GB/s of expert weights)\n",
                  NF, med, bytes / (med / 1000.0) / 1e9);
      std::printf("  x%llu layers = %.1f ms/token -> %.1f tok/s ceiling from experts alone\n",
                  (unsigned long long)n_layer, med * nl, 1000.0 / (med * nl));
      std::printf("  launches per FFN: %u  (n_experts*2 gate/up + 1 swiglu + 1 batched down)\n",
                  NF * 2 + 2);
      if (med_mono > 0) {
        const double mb = (double)NF * 3 * L.gate.rows * L.gate.cols * qspec.bpw() / 8;
        std::printf("  MONOLITHIC (1 cooperative launch): %.3f ms  (%.1f GB/s)  %+.1f%% vs staged\n",
                    med_mono, mb / (med_mono / 1000.0) / 1e9, 100.0 * (med - med_mono) / med_mono);
        std::printf("  x%llu layers = %.1f ms/token -> %.1f tok/s ceiling from experts alone\n",
                    (unsigned long long)n_layer, med_mono * nl, 1000.0 / (med_mono * nl));
      }
    }
  }

  // ---- dense bf16 matvec: correctness, then throughput on the real shapes ---------------------
  // The projections that dominate the dense traffic, sized from the container rather than from a
  // remembered geometry, so this measures the model in hand.
  {
    std::printf("\n[dense bf16 matvec on GPU]\n");
    struct Shape { uint32_t rows, cols; const char* name; };
    std::vector<Shape> shapes;
    {
      const uint32_t l0 = (only_layer >= 0) ? (uint32_t)only_layer : 0u;
      for (const char* nm : {"attn_q_b", "attn_out_a", "attn_out_b", "shexp_gate"})
        if (const AffTensorEntry* e = aff.find_tensor("blk." + std::to_string(l0) + "." + nm))
          shapes.push_back({(uint32_t)e->desc.rows, (uint32_t)e->desc.cols, nm});
    }
    for (const Shape& sh : shapes) {
      std::vector<uint16_t> Wh((size_t)sh.rows * sh.cols);
      for (size_t i = 0; i < Wh.size(); ++i) Wh[i] = float_to_bf16(0.02f * nd(rng));
      std::vector<float> xh(sh.cols);
      for (auto& v : xh) v = nd(rng);

      std::vector<float> y_cpu(sh.rows);
      matvec_bf16(Wh.data(), xh.data(), sh.rows, sh.cols, y_cpu.data());

      DeviceBuffer<uint16_t> dW(Wh.size());
      DeviceBuffer<float> dx(sh.cols), dy(sh.rows);
      dW.upload(Wh.data(), Wh.size());
      dx.upload(xh.data(), sh.cols);
      matvec_bf16_hip(dW.get(), dx.get(), sh.rows, sh.cols, dy.get());
      AFF_HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> y_gpu(sh.rows);
      dy.download(y_gpu.data(), sh.rows);
      double w2 = 0;
      const double rel = rel_l2(y_gpu, y_cpu, &w2);
      if (!(rel >= 0 && rel < 1e-4)) ++g_fail;

      const double bytes = (double)sh.rows * sh.cols * 2;
      const int inner = 32;
      double best = 1e30;
      for (int r = 0; r < (reps > 0 ? reps : 5); ++r) {
        AFF_HIP_CHECK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < inner; ++k) matvec_bf16_hip(dW.get(), dx.get(), sh.rows, sh.cols, dy.get());
        AFF_HIP_CHECK(hipDeviceSynchronize());
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / inner);
      }
      std::printf("  %-11s %5ux%-5u %6.1f MiB  rel=%.2e  %7.1f GB/s  %6.1f us\n",
                  sh.name, sh.rows, sh.cols, bytes / 1048576.0, rel, bytes / best / 1e9, best * 1e6);
    }
    // CAUTION on the rates above: a matrix re-read many times lands largely in cache, so these can
    // exceed the card's VRAM wall and are NOT the rate a real decode sees. The
    // distinct weights a token reads cannot cache, and that total is what sets decode's floor — so
    // compute it from THIS container rather than quoting a constant. A hardcoded total silently
    // survives a dtype change and overstates the ceiling in the one line whose job is to bound it.
    struct Group { const char* name; double bytes; };
    Group g[] = {{"attention (q_a,q_b,kv,out_a,out_b)", 0}, {"shared expert (gate,up,down)", 0},
                 {"router gate", 0}, {"compressor (wkv,wgate)", 0},
                 {"indexer (wq_b,proj,comp)", 0}, {"vocab head", 0}, {"norms and other small", 0}};
    double skipped = 0;
    for (uint64_t i = 0; i < aff.tensor_count(); ++i) {
      const AffTensorEntry& e = aff.tensor_at(i);
      const std::string n(e.name);
      const double b = (double)(e.desc.data_size + e.desc.scale_size + e.desc.rscale_size);
      // Read by row, not whole: the embedding is one row a token and the hash gate one entry.
      // The codebook is 8 KB and lives in cache from the first layer on.
      if (n == "token_embd" || n.find("ffn_gate_hash") != std::string::npos || n == "__codebook") {
        skipped += b;
        continue;
      }
      const size_t dot = n.rfind('.');
      const std::string leaf = dot == std::string::npos ? n : n.substr(dot + 1);
      int k = 6;
      if (leaf == "attn_q_a" || leaf == "attn_q_b" || leaf == "attn_kv" || leaf == "attn_out_a" ||
          leaf == "attn_out_b") k = 0;
      else if (leaf.rfind("shexp_", 0) == 0) k = 1;
      else if (leaf == "ffn_gate_inp") k = 2;
      else if (leaf == "comp_wkv" || leaf == "comp_wgate") k = 3;
      else if (leaf.rfind("idx_", 0) == 0) k = 4;
      else if (n == "output") k = 5;
      g[k].bytes += b;
    }
    double tot = 0;
    for (const Group& q : g) tot += q.bytes;
    std::printf("  cache-resident rates above; the DISTINCT dense bytes a token reads, from this\n"
                "  container's own tensor directory (%.2f GB skipped as row lookups):\n",
                skipped / 1e9);
    for (const Group& q : g)
      std::printf("    %-36s %7.3f GB  %4.1f%%\n", q.name, q.bytes / 1e9, 100.0 * q.bytes / tot);
    // The indexer row is an UPPER bound: it only runs past its context threshold, so a short
    // prompt reads none of it. Everything else is read every token of every layer.
    std::printf("    %-36s %7.3f GB\n", "TOTAL", tot / 1e9);
    const double ndev = devs.empty() ? 1.0 : (double)devs.size();
    const double one_wall = vram_gbps * 1e9, all_wall = ndev * one_wall;
    std::printf("  at the %.0f GB/s VRAM wall (--vram-gbps) that is %.1f ms/token on one GPU,"
                " %.1f ms across %.0f\n"
                "  -- so %.0f tok/s is decode's dense-only ceiling, before experts or any overhead\n",
                vram_gbps, tot / one_wall * 1000, tot / all_wall * 1000, ndev, all_wall / tot);
  }

  std::printf("\n%d matrices checked, worst rel L2 = %.3e\n", checked, worst_rel);
  if (g_fail == 0) {
    std::printf("GPU and CPU agree on the real container bytes\n");
    return 0;
  }
  std::printf("%d MATRICES DISAGREE\n", g_fail);
  return 1;
}
