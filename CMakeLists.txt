cmake_minimum_required(VERSION 3.24)
project(affinity-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ON by default because this is a build-from-source project and the CPU kernels are gated on
# __AVX512F__ — without -march=native they silently fall back to scalar. The resulting binary runs
# only on CPUs at least as capable as the build host; turn this OFF to produce a portable one.
option(AFF_NATIVE "Build for the host CPU (-march=native)" ON)

# ---- build stamp ------------------------------------------------------------------------------
#
# Provenance only — it is fixed at CONFIGURE time, so it does not move when a source file is edited
# and rebuilt. What actually catches a stale binary is `__DATE__ __TIME__` in src/build_stamp.h,
# which is the compile time of that binary's own translation unit, and the mtime check in
# bench/measure.sh. Do not treat this string as evidence that a binary is current.
execute_process(COMMAND git describe --always --dirty
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE AFF_GIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT AFF_GIT_ID)
  set(AFF_GIT_ID "nogit")
endif()
add_compile_definitions(AFF_BUILD_ID="${AFF_GIT_ID}")

# HIP is optional so the LIBRARY and the tests still build CPU-only on a machine without ROCm. The
# `affinity` binary is not in that set — see the note on its target. Enabled when hipcc is present.
#
# NOTE: a CACHE STRING, not option() — option() coerces its value to a boolean, so "AUTO" would
# silently become OFF and the GPU would never be built.
set(AFF_HIP "AUTO" CACHE STRING "Build the GPU kernels: ON, OFF, or AUTO")
set(AFF_GPU_ARCH "gfx1201" CACHE STRING "GPU architecture to compile for")
find_program(AFF_HIPCC hipcc HINTS /opt/rocm/bin /opt/rocm/llvm/bin)
if(AFF_HIP STREQUAL "AUTO")
  if(AFF_HIPCC)
    set(AFF_HIP ON)
  else()
    set(AFF_HIP OFF)
  endif()
endif()

if(AFF_HIP)
  # CMakeDetermineHIPCompiler looks for the ROCm root via ROCM_PATH / PATH, so a configure run
  # without /opt/rocm/bin on PATH fails outright rather than falling back. Derive it from hipcc.
  get_filename_component(AFF_ROCM_ROOT "${AFF_HIPCC}" DIRECTORY)
  get_filename_component(AFF_ROCM_ROOT "${AFF_ROCM_ROOT}" DIRECTORY)
  # CMakeDetermineHIPCompiler resolves this by running `hipconfig --rocmpath`, which needs ROCm on
  # PATH. Setting it directly means a plain `cmake -S . -B build` works from any shell.
  set(CMAKE_HIP_COMPILER_ROCM_ROOT "${AFF_ROCM_ROOT}" CACHE PATH "" FORCE)
  set(ENV{ROCM_PATH} "${AFF_ROCM_ROOT}")
  # ROCm's clang++ invokes clang-offload-bundler and llvm-objcopy by name. Without ROCm's llvm/bin
  # ahead of the system one, it finds /usr/bin/clang-offload-bundler, which cannot see ROCm's
  # llvm-objcopy, and the compiler test dies with "unable to find 'llvm-objcopy' in path".
  set(ENV{PATH} "${AFF_ROCM_ROOT}/lib/llvm/bin:${AFF_ROCM_ROOT}/bin:$ENV{PATH}")
  enable_language(HIP)
  # One arch, never a generic target: every cross-lane op in src/gpu/ assumes wave32, and the WMMA
  # builtins are the gfx12 spelling. The runtime device filter compares against this exact string,
  # so a card of a different arch is refused rather than handed code objects it cannot run.
  if(NOT AFF_GPU_ARCH MATCHES "^gfx12")
    message(FATAL_ERROR
      "AFF_GPU_ARCH is '${AFF_GPU_ARCH}'. src/gpu/ requires RDNA4 (gfx12xx): wave32 cross-lane "
      "arithmetic and the __builtin_amdgcn_wmma_*_w32_gfx12 intrinsics have no fallback path.")
  endif()
  set(CMAKE_HIP_ARCHITECTURES ${AFF_GPU_ARCH})
  add_compile_definitions(AFF_GPU_ARCH="${AFF_GPU_ARCH}")
  # CMake resolves the HIP compiler to ROCm's clang++ rather than the hipcc wrapper, and plain
  # clang++ does not add ROCm's include/lib paths on its own.
  find_library(AFF_HIP_RT amdhip64 HINTS "${AFF_ROCM_ROOT}/lib")
  # As a -I flag, NOT target_include_directories: CMake classifies ${ROCM}/include as an implicit
  # HIP include directory and strips it from the command line, after which ROCm's own clang++
  # cannot find hip/hip_runtime.h. A raw compile option is not filtered.
  set(AFF_HIP_INC "-I${AFF_ROCM_ROOT}/include")
  # ROCm's clang++ adds `-lamdhip64` implicitly under --hip-link but does not add its own lib dir,
  # so the link fails unless we point the linker at it.
  set(AFF_HIP_LIBDIR "-L${AFF_ROCM_ROOT}/lib")
  message(STATUS "HIP enabled: ${AFF_GPU_ARCH} (rocm at ${AFF_ROCM_ROOT})")
else()
  message(STATUS "HIP disabled (no hipcc found) — CPU-only build")
endif()

# ---- the terminal UI ---------------------------------------------------------------------------
#
# Its own library, and it depends on nothing but libc and POSIX. `aff_format` links it rather than
# absorbing it because ui/log.h is included from engine/ and gpu/ — the informational output of the
# whole engine goes through one sink — and a UI that could reach back into the engine would make
# that a cycle. Nothing here calls HIP, so it builds in the CPU-only configure too.
add_library(aff_ui STATIC
  src/ui/log.cpp
  src/ui/term.cpp
  src/ui/sysprobe.cpp
  src/ui/dashboard.cpp
)
target_include_directories(aff_ui PUBLIC src)
target_compile_options(aff_ui PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_library(aff_format STATIC
  src/format/aff_format.cpp
  src/format/aff_writer.cpp
  src/format/aff_reader.cpp
  src/model/safetensors.cpp
  src/quant/source_dtypes.cpp
  src/quant/hadamard.cpp
  src/quant/blockquant.cpp
  src/quant/imatrix.cpp
  src/quant/evalharness.cpp
  src/engine/expert_kernel.cpp
  src/engine/attention.cpp
  src/engine/ops.cpp
  src/engine/host_pool.cpp
  src/engine/expert_heat.cpp
  src/engine/instrument.cpp
  src/engine/route_trace.cpp
  src/engine/config.cpp
  src/engine/chat_encode.cpp
  src/engine/prefix_cache.cpp
  src/engine/threadpool.cpp
  src/engine/sampler.cpp
  src/engine/tokenizer.cpp
  src/server/server.cpp
  src/engine/model.cpp
  src/engine/engram.cpp
)
target_include_directories(aff_format PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(aff_format PUBLIC Threads::Threads aff_ui)
target_compile_options(aff_format PRIVATE -Wall -Wextra -Wno-unused-parameter)
if(AFF_NATIVE)
  target_compile_options(aff_format PRIVATE -march=native)
endif()
# Cross-rank/host-device bitwise determinism: never let the compiler fuse multiplies
# differently between build units.
target_compile_options(aff_format PRIVATE -ffp-contract=off)

enable_testing()

# ---- ONE CMakeLists FOR TWO TREES ---------------------------------------------------------------
#
# The release tree ships src/ and the tools a user needs to quantise and run a model. tests/, bench/
# and the development tools are not in it. Rather than maintain a second CMakeLists that drifts from
# this one, every optional target is guarded on its own source existing: a missing directory is a
# smaller build, never a configure error, and a tree that has everything configures identically.
#
# Anything added below must go through `aff_tool` or inside one of the `AFF_HAVE_*` guards, or it
# will break the configure of whichever tree does not carry it.
function(aff_tool name src)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${src}")
    return()
  endif()
  add_executable(${name} ${src})
  if(ARGN)
    target_link_libraries(${name} PRIVATE ${ARGN})
  endif()
endfunction()

set(AFF_HAVE_TESTS FALSE)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_tokenizer.cpp")
  set(AFF_HAVE_TESTS TRUE)
endif()

# ---- tools --------------------------------------------------------------------------------------
# The first three are the user-facing path: quantise a checkpoint, look at what came out, run it.
# The rest are development tools and are absent from the release tree.
aff_tool(aff-info     tools/aff_info.cpp     aff_format)
aff_tool(aff-synth    tools/aff_synth.cpp    aff_format)
aff_tool(aff-kbench   tools/aff_kbench.cpp   aff_format)
aff_tool(aff-verify   tools/aff_verify.cpp   aff_format)
aff_tool(aff-probe    tools/aff_probe.cpp    aff_format)
aff_tool(aff-quantlab tools/aff_quantlab.cpp aff_format)
# No model, no card, no engine: they read a --route-trace file and nothing else.
# The n-gram hash, printed, so it can be diffed against the reference's own NgramHashState. It
# needs no model and no card: the constants come out of tools/v41/engram_prep.py.
aff_tool(aff-engramcheck tools/aff_engramcheck.cpp aff_format)
aff_tool(aff-predict  tools/aff_predict.cpp)
aff_tool(aff-profile  tools/aff_profile.cpp)

# ---- tests --------------------------------------------------------------------------------------
if(AFF_HAVE_TESTS)
  add_executable(test_tokenizer tests/test_tokenizer.cpp)
  target_link_libraries(test_tokenizer PRIVATE aff_format)
  add_test(NAME tokenizer COMMAND test_tokenizer)
  set_tests_properties(tokenizer PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(test_aff_format tests/test_aff_format.cpp)
  target_link_libraries(test_aff_format PRIVATE aff_format)
  add_test(NAME aff_format COMMAND test_aff_format)

  add_executable(test_ldlq tests/test_ldlq.cpp)
  target_link_libraries(test_ldlq PRIVATE aff_format)
  add_test(NAME ldlq COMMAND test_ldlq)

  add_executable(test_expert_kernel tests/test_expert_kernel.cpp)
  target_link_libraries(test_expert_kernel PRIVATE aff_format)
  add_test(NAME expert_kernel COMMAND test_expert_kernel)

  add_executable(test_attention tests/test_attention.cpp)
  target_link_libraries(test_attention PRIVATE aff_format)
  add_test(NAME attention COMMAND test_attention)

  add_executable(test_expert_heat tests/test_expert_heat.cpp)
  target_link_libraries(test_expert_heat PRIVATE aff_format)
  add_test(NAME expert_heat COMMAND test_expert_heat)

  add_executable(test_ops tests/test_ops.cpp)
  target_link_libraries(test_ops PRIVATE aff_format)
  add_test(NAME ops COMMAND test_ops)

  add_executable(test_prefix_cache tests/test_prefix_cache.cpp)
  target_link_libraries(test_prefix_cache PRIVATE aff_format)
  add_test(NAME prefix_cache COMMAND test_prefix_cache)

  add_executable(test_server tests/test_server.cpp)
  target_link_libraries(test_server PRIVATE aff_format)
  add_test(NAME server COMMAND test_server)

  add_executable(test_quant tests/test_quant.cpp)
  target_link_libraries(test_quant PRIVATE aff_format)
  add_test(NAME quant COMMAND test_quant)

  # The fixtures are the container's own, so the test needs their path rather than a copy.
  add_executable(test_chat_encode tests/test_chat_encode.cpp)
  target_link_libraries(test_chat_encode PRIVATE aff_format)
  add_test(NAME chat_encode
           COMMAND test_chat_encode ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/chat)
  set_tests_properties(chat_encode PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

if(AFF_HIP)
  add_library(aff_gpu STATIC src/gpu/expert_kernel.hip src/gpu/tp_allreduce.hip
                             src/gpu/sampler_gpu.hip src/gpu/moe_plan.hip
                             src/gpu/digest.hip src/gpu/vmem.hip
                             src/gpu/static_placement.hip src/gpu/dense_gpu.hip
                             src/gpu/attention_gpu.hip src/gpu/keepalive.hip
                             src/gpu/batch_kernels.hip src/gpu/attn_flash.hip
                             src/gpu/indexer_gpu.hip src/gpu/compressor_gpu.hip)
  set_source_files_properties(src/gpu/expert_kernel.hip src/gpu/tp_allreduce.hip
                              src/gpu/moe_plan.hip src/gpu/digest.hip src/gpu/vmem.hip
                              src/gpu/static_placement.hip src/gpu/dense_gpu.hip
                              src/gpu/attention_gpu.hip src/gpu/keepalive.hip
                              src/gpu/batch_kernels.hip src/gpu/attn_flash.hip
                              src/gpu/indexer_gpu.hip src/gpu/compressor_gpu.hip
                              PROPERTIES LANGUAGE HIP)
  target_include_directories(aff_gpu PUBLIC src)
  target_link_libraries(aff_gpu PUBLIC "${AFF_HIP_RT}" aff_format)
  target_compile_options(aff_gpu PUBLIC "${AFF_HIP_INC}")
  target_link_options(aff_gpu PUBLIC "${AFF_HIP_LIBDIR}")
  target_compile_options(aff_gpu PRIVATE -ffp-contract=off)

  target_compile_definitions(aff_gpu PUBLIC AFF_WITH_HIP)

  # Compiled as HIP because it uses DeviceBuffer; it is a tool, not a unit test, because it needs a
  # real container to point at.
  if(EXISTS "${CMAKE_SOURCE_DIR}/tools/aff_gpucheck.cpp")
    add_executable(aff-gpucheck tools/aff_gpucheck.cpp)
    set_source_files_properties(tools/aff_gpucheck.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(aff-gpucheck PRIVATE aff_gpu aff_format)
  endif()

  if(AFF_HAVE_TESTS)
    # One HIP test: source, LANGUAGE, aff_gpu, register. The block below predates it and is left as
    # it is; new tests use this.
    function(aff_hip_test tgt name)
      if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tests/${tgt}.hip")
        return()
      endif()
      add_executable(${tgt} tests/${tgt}.hip)
      set_source_files_properties(tests/${tgt}.hip PROPERTIES LANGUAGE HIP)
      target_link_libraries(${tgt} PRIVATE aff_gpu aff_format)
      add_test(NAME ${name} COMMAND ${tgt})
    endfunction()

    add_executable(test_tp tests/test_tp.hip)
    set_source_files_properties(tests/test_tp.hip PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_tp PRIVATE aff_gpu)
    add_test(NAME tp_allreduce COMMAND test_tp)

    add_executable(test_hip_expert tests/test_hip_expert.hip)
    set_source_files_properties(tests/test_hip_expert.hip PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_hip_expert PRIVATE aff_gpu aff_format)
    add_test(NAME hip_expert COMMAND test_hip_expert)

    add_executable(test_attn_gpu tests/test_attn_gpu.hip)
    set_source_files_properties(tests/test_attn_gpu.hip PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_attn_gpu PRIVATE aff_gpu aff_format)
    add_test(NAME attn_gpu COMMAND test_attn_gpu)

    add_executable(test_sampler_gpu tests/test_sampler_gpu.hip)
    set_source_files_properties(tests/test_sampler_gpu.hip PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_sampler_gpu PRIVATE aff_gpu)
    add_test(NAME sampler_gpu COMMAND test_sampler_gpu)

    # The N-way split is bytes, not arithmetic — see the file. It runs without a GPU, so it is the
    # first thing to check when a rank count above two produces wrong text.
    aff_hip_test(test_expert_shard expert_shard)

    add_executable(test_moe_plan tests/test_moe_plan.hip)
    set_source_files_properties(tests/test_moe_plan.hip PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_moe_plan PRIVATE aff_gpu)
    add_test(NAME moe_plan COMMAND test_moe_plan)

    # Guards the WMMA fragment layout every matrix kernel in src/gpu depends on. AMD does not
    # document the accumulator mapping, so this test IS the documentation.
    add_executable(test_wmma tests/test_wmma.hip)
    set_source_files_properties(tests/test_wmma.hip PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_wmma PRIVATE aff_gpu)
    add_test(NAME wmma COMMAND test_wmma)
  endif()

  # ---- benches ----------------------------------------------------------------------------------
  #
  # EVERY bench/*.hip is a target, by glob, and the glob is CONFIGURE_DEPENDS so a new file appears
  # without anyone remembering to add it. A bench built by hand once is a bench that keeps reporting
  # success from an aging binary, which is worse than one that does not build at all.
  #
  # EXCLUDE_FROM_ALL so a plain build stays fast; bench/measure.sh names the one it wants and fails
  # if it does not build.
  file(GLOB AFF_BENCH_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bench/*.hip)
  if(AFF_BENCH_SRC)
    foreach(src ${AFF_BENCH_SRC})
      get_filename_component(name ${src} NAME_WE)
      add_executable(bench_${name} EXCLUDE_FROM_ALL ${src})
      set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
      target_link_libraries(bench_${name} PRIVATE aff_gpu aff_format)
      list(APPEND AFF_BENCH_TARGETS bench_${name})
    endforeach()
    add_custom_target(benches DEPENDS ${AFF_BENCH_TARGETS})
  endif()
endif()

add_executable(aff-quantize tools/aff_quantize.cpp)
target_link_libraries(aff-quantize PRIVATE aff_format)

# HIP-only, and that is a statement about what it IS rather than a build convenience. Every tier
# `affinity` drives — the dense weights, the KV cache, the expert slabs, the placement engine —
# lives on the cards; without them there is no model resident anywhere and the first forward aborts.
# A CPU-only configure still builds the library, every test, and aff-verify's host reference path.
if(AFF_HIP)
  add_executable(affinity tools/affinity.cpp)
  target_link_libraries(affinity PRIVATE aff_format)
endif()

# ---------------------------------------------------------------------------------------------
# GPU consumers, wired LAST.
#
# target_link_libraries needs its target to exist already, so anything naming an executable has to
# come after every add_executable. Getting this wrong fails the configure, and a failed configure
# leaves the previous build directory in place to be rebuilt from — so the symptom is a binary with
# the GPU path compiled out that still looks like it was linked against it.
# ---------------------------------------------------------------------------------------------
if(AFF_HIP)
  foreach(t aff-verify affinity)
    if(TARGET ${t})
      target_link_libraries(${t} PRIVATE aff_gpu)
      target_compile_definitions(${t} PRIVATE AFF_WITH_HIP)
    endif()
  endforeach()
endif()
