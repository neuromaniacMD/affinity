# affinity for gfx1201 (Radeon AI PRO R9700) on AMD's TheRock ROCm 7.14 build — the host needs no ROCm.
#   docker build -f docker/Dockerfile -t affinity:<tag> .      (from the repo root)
# hipcc and ROCm clang come from the image venv's _rocm_sdk_devel tree. AFF_HIPCC is passed explicitly so the
# CMake build derives ROCM_ROOT from that tree; its default find_program would pick /opt/venv/bin/hipcc and derive
# /opt/venv, which has no lib/llvm/bin.
FROM kyuz0/vllm-therock-gfx1201:rocm7.14.0-torch2.11.0-vllm0.27.1
ARG SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
COPY . /opt/affinity-src
RUN cd /opt/affinity-src \
 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAFF_HIP=ON -DAFF_GPU_ARCH=gfx1201 -DAFF_NATIVE=ON \
      -DAFF_HIPCC=$SDK/bin/hipcc -DCMAKE_CXX_COMPILER=g++ \
 && cmake --build build -j \
 && ls -l build/affinity build/aff-info build/aff-gpucheck build/aff-quantize \
 && ldd build/affinity | grep -E "amdhip|not found"
ENV PATH=/opt/affinity-src/build:$PATH
ENTRYPOINT ["/opt/affinity-src/build/affinity"]
