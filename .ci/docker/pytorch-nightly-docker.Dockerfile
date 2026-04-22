ARG BASE_IMAGE=rocm/pytorch-autobuild:base-latest
FROM ${BASE_IMAGE}
WORKDIR /tmp
USER root

ENV CI=1
ENV PYTORCH_TEST_WITH_ROCM=1
ENV PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
ENV USE_NVSHMEM=0

RUN git clone https://github.com/pytorch/pytorch --recursive \
    && cd pytorch \
    # Bypass sccache on torch_rocshmem: its -fgpu-rdc + mixed xnack± offload-arch flags break sccache's argv parser.
    && sed -i 's|set_target_properties(torch_rocshmem PROPERTIES LINKER_LANGUAGE HIP)|set_target_properties(torch_rocshmem PROPERTIES LINKER_LANGUAGE HIP CXX_COMPILER_LAUNCHER "" HIP_COMPILER_LAUNCHER "")|' caffe2/CMakeLists.txt \
    && pip install -r requirements.txt \
    && git config --local user.name "AMD AMD" \
    && git config --local user.email "amd@amd.com" \
    && git remote add rocm https://github.com/ROCm/pytorch.git \
    && git fetch rocm \
    && git cherry-pick 519160d466782f5a62365be051fcb3ef90fa0b00 \
    && (.ci/pytorch/build.sh > /tmp/build.log 2>&1 || (tail -300 /tmp/build.log; exit 1)) \
    && rm -rf /tmp/pytorch/.git
RUN git clone https://github.com/pytorch/vision \
    && cd vision \
    && FORCE_CUDA=1 python setup.py install \
    && rm -rf /tmp/vision/.git
RUN git clone https://github.com/pytorch/audio \
    && cd audio \
    && python setup.py install \
    && rm -rf /tmp/audio/.git
