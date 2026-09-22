# local/affinity:a48b72e-therock7.14 — amdnas 2026-09-21: StillDeadcode/affinity (a48b72e) for gfx1201, built on the
# TheRock ROCm 7.14 image (kyuz0/vllm-therock-gfx1201 = the gfx1201 TheRock build this workspace already uses for
# FreeToken/vLLM work; user 2026-09-21: "make sure you use the rocm builds from TheRock"). The host has no ROCm and
# nothing is installed on it. hipcc + ROCm clang come from the image venv _rocm_sdk_devel tree; AFF_HIPCC is passed
# explicitly so affinity CMake derives ROCM_ROOT = that tree (its default find_program would pick /opt/venv/bin/hipcc
# and derive /opt/venv, which has no lib/llvm/bin). Build context = a copy of the repo checkout minus .git.
# Source of this file: ~/Claude/infra/amdnas/deployed/build/affinity-docker/Dockerfile (deployed to
# amdnas:/mnt/models2/build/affinity-docker/).
FROM kyuz0/vllm-therock-gfx1201:rocm7.14.0-torch2.11.0-vllm0.27.1
ARG SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
COPY affinity /opt/affinity-src
RUN cd /opt/affinity-src \
 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAFF_HIP=ON -DAFF_GPU_ARCH=gfx1201 -DAFF_NATIVE=ON \
      -DAFF_HIPCC=$SDK/bin/hipcc -DCMAKE_CXX_COMPILER=g++ \
 && cmake --build build -j \
 && ls -l build/affinity build/aff-info build/aff-gpucheck build/aff-quantize \
 && ldd build/affinity | grep -E "amdhip|not found"
ENV PATH=/opt/affinity-src/build:$PATH
ENTRYPOINT ["/opt/affinity-src/build/affinity"]
