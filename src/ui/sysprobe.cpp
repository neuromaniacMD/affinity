#include "ui/sysprobe.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

namespace aff::ui {
namespace {

uint64_t now_ms() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

int open_ro(const std::string& p) { return ::open(p.c_str(), O_RDONLY | O_CLOEXEC); }

// sysfs regenerates a node's contents on every read that starts at offset 0, which is what pread
// gives without a seek — so the fd stays open for the life of the process and each sample is one
// syscall. Reopening 200 times a second would work too and would be 200 path walks.
bool read_num(int fd, double* out) {
  if (fd < 0) return false;
  char buf[64];
  const ssize_t n = ::pread(fd, buf, sizeof(buf) - 1, 0);
  if (n <= 0) return false;
  buf[n] = 0;
  char* end = nullptr;
  const double v = std::strtod(buf, &end);
  if (end == buf) return false;
  *out = v;
  return true;
}

// The clock nodes are "<n> Mhz" lines with the active one starred; everything else is a bare
// integer. One parser, because the alternative is a second sampling path that gets forgotten.
bool read_starred_mhz(int fd, double* out) {
  if (fd < 0) return false;
  char buf[512];
  const ssize_t n = ::pread(fd, buf, sizeof(buf) - 1, 0);
  if (n <= 0) return false;
  buf[n] = 0;
  for (char* line = buf; line && *line;) {
    char* nl = std::strchr(line, '\n');
    if (nl) *nl = 0;
    if (std::strchr(line, '*')) {
      if (const char* c = std::strchr(line, ':')) {
        *out = std::strtod(c + 1, nullptr);
        return true;
      }
    }
    line = nl ? nl + 1 : nullptr;
  }
  return false;
}

std::string first_subdir(const std::string& dir, const char* prefix) {
  DIR* d = ::opendir(dir.c_str());
  if (!d) return {};
  std::string found;
  while (dirent* e = ::readdir(d)) {
    if (std::strncmp(e->d_name, prefix, std::strlen(prefix)) == 0) { found = e->d_name; break; }
  }
  ::closedir(d);
  return found;
}

}  // namespace

Probe::~Probe() {
  for (Dev& d : dev_)
    for (Node* n : {&d.busy, &d.mem_busy, &d.vram_used, &d.temp, &d.power, &d.sclk, &d.mclk})
      if (n->fd >= 0) ::close(n->fd);
}

void Probe::add_device(const DeviceInfo& info) {
  Dev d;
  d.info = info;
  // /sys/bus/pci/devices/<bdf> is the device itself; the drm nodes and the amdgpu attributes hang
  // off it, so the PCI address the driver reports is enough and no card index has to be guessed.
  const std::string base = "/sys/bus/pci/devices/" + info.bus_id;
  if (::access(base.c_str(), F_OK) == 0) d.sysfs = base;

  if (!d.sysfs.empty()) {
    d.busy.fd = open_ro(d.sysfs + "/gpu_busy_percent");
    d.mem_busy.fd = open_ro(d.sysfs + "/mem_busy_percent");
    d.vram_used.fd = open_ro(d.sysfs + "/mem_info_vram_used");
    d.sclk.fd = open_ro(d.sysfs + "/pp_dpm_sclk");
    d.mclk.fd = open_ro(d.sysfs + "/pp_dpm_mclk");
    // hwmon's instance number is assigned at probe time and is not stable across boots, so it is
    // discovered rather than assumed.
    const std::string hdir = d.sysfs + "/hwmon";
    if (const std::string h = first_subdir(hdir, "hwmon"); !h.empty()) {
      const std::string hp = hdir + "/" + h;
      d.temp.fd = open_ro(hp + "/temp1_input");     // millidegrees
      d.temp.scale = 1e-3;
      d.power.fd = open_ro(hp + "/power1_average"); // microwatts
      d.power.scale = 1e-6;
      if (d.power.fd < 0) { d.power.fd = open_ro(hp + "/power1_input"); d.power.scale = 1e-6; }
    }
  }
  dev_.push_back(std::move(d));
}

void Probe::tick() {
  double v = 0;
  for (Dev& d : dev_) {
    if (read_num(d.busy.fd, &v)) { d.s_busy += v; ++d.n_busy; }
    if (read_num(d.mem_busy.fd, &v)) { d.s_mem += v; ++d.n_mem; }
    if (read_num(d.vram_used.fd, &v)) { d.s_vram += v / 1073741824.0; ++d.n_vram; }
    if (read_num(d.temp.fd, &v)) { d.s_temp += v * d.temp.scale; ++d.n_temp; }
    if (read_num(d.power.fd, &v)) { d.s_power += v * d.power.scale; ++d.n_power; }
    if (read_starred_mhz(d.sclk.fd, &v)) { d.s_sclk += v; ++d.n_sclk; }
    if (read_starred_mhz(d.mclk.fd, &v)) { d.s_mclk += v; ++d.n_mclk; }
  }
}

void Probe::roll(std::vector<DeviceSample>* out) {
  out->assign(dev_.size(), DeviceSample{});
  for (size_t i = 0; i < dev_.size(); ++i) {
    Dev& d = dev_[i];
    DeviceSample& s = (*out)[i];
    s.present = !d.sysfs.empty();
    auto avg = [](double sum, uint32_t n) { return n ? sum / (double)n : -1.0; };
    s.busy = avg(d.s_busy, d.n_busy);
    s.mem_busy = avg(d.s_mem, d.n_mem);
    s.vram_used_gib = avg(d.s_vram, d.n_vram);
    s.temp_c = avg(d.s_temp, d.n_temp);
    s.power_w = avg(d.s_power, d.n_power);
    s.sclk_mhz = avg(d.s_sclk, d.n_sclk);
    s.mclk_mhz = avg(d.s_mclk, d.n_mclk);
    d.s_busy = d.s_mem = d.s_vram = d.s_temp = d.s_power = d.s_sclk = d.s_mclk = 0;
    d.n_busy = d.n_mem = d.n_vram = d.n_temp = d.n_power = d.n_sclk = d.n_mclk = 0;
  }
}

const HostSample& Probe::host() {
  const uint64_t t = now_ms();
  if (host_at_ms_ && t - host_at_ms_ < 250) return host_;
  host_at_ms_ = t;

  auto scan = [](const char* path, const char* const* keys, double* out, int n) {
    std::FILE* f = std::fopen(path, "re");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
      for (int i = 0; i < n; ++i) {
        const size_t kl = std::strlen(keys[i]);
        if (std::strncmp(line, keys[i], kl) == 0) {
          // Every field of interest in both files is in kB.
          out[i] = std::strtod(line + kl, nullptr) / 1048576.0;
          break;
        }
      }
    }
    std::fclose(f);
  };

  const char* mk[] = {"MemTotal:", "MemAvailable:"};
  double mv[2] = {0, 0};
  scan("/proc/meminfo", mk, mv, 2);
  host_.mem_total_gib = mv[0];
  host_.mem_avail_gib = mv[1];

  const char* sk[] = {"VmRSS:", "VmHWM:"};
  double sv[2] = {0, 0};
  scan("/proc/self/status", sk, sv, 2);
  host_.rss_gib = sv[0];
  host_.rss_peak_gib = sv[1];
  return host_;
}

}  // namespace aff::ui
