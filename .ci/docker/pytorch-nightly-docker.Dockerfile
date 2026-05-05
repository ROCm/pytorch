ARG BASE_IMAGE=rocm/pytorch-autobuild:base-latest

FROM ${BASE_IMAGE} AS base
WORKDIR /tmp
USER root

ENV CI=1
ENV PYTORCH_TEST_WITH_ROCM=1
ENV PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
# FIXME: Disable sccache preprocessor cache mode. Its linemarker parser fails with "Failed to parse included file path" on hipcc preprocessed output
ENV SCCACHE_DIRECT=false

FROM base AS pytorch
RUN git clone https://github.com/pytorch/pytorch --recursive \
    && cd pytorch \
    && pip install -r requirements.txt \
    && (.ci/pytorch/build.sh > /tmp/build.log 2>&1 || (tail -300 /tmp/build.log; exit 1)) \
    && rm -rf /tmp/pytorch/.git

FROM pytorch AS vision
RUN git clone https://github.com/pytorch/vision \
    && cd vision \
    && FORCE_CUDA=1 python setup.py install \
    && rm -rf /tmp/vision/.git

FROM vision AS audio
RUN git clone https://github.com/pytorch/audio \
    && cd audio \
    && python setup.py install \
    && rm -rf /tmp/audio/.git
