ARG BASE_IMAGE=rocm/pytorch-autobuild:base-latest
FROM ${BASE_IMAGE}
WORKDIR /tmp
USER root

ENV CI=1
ENV PYTORCH_TEST_WITH_ROCM=1
ENV PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"

RUN git clone https://github.com/pytorch/pytorch --recursive \
    && cd pytorch \
    && pip install -r requirements.txt \
    && git config --local user.name "AMD AMD" \
    && git config --local user.email "amd@amd.com" \
    && git remote add rocm https://github.com/ROCm/pytorch.git \
    && git fetch rocm \
    && git cherry-pick 519160d466782f5a62365be051fcb3ef90fa0b00 \
    && if ! .ci/pytorch/build.sh; then \
         echo "PyTorch build failed. Re-running likely failing HIP test targets with serial verbose Ninja output."; \
         if [ -d build ]; then \
           ninja -C build -t clean hip_half_test hip_distributions_test || true; \
           ninja -C build -j1 -v hip_half_test || true; \
           ninja -C build -j1 -v hip_distributions_test || true; \
         else \
           echo "Expected build directory 'build' was not found after failure."; \
         fi; \
         exit 1; \
       fi \
    && rm -rf /tmp/pytorch/.git
RUN git clone https://github.com/pytorch/vision \
    && cd vision \
    && FORCE_CUDA=1 python setup.py install \
    && rm -rf /tmp/vision/.git
RUN git clone https://github.com/pytorch/audio \
    && cd audio \
    && python setup.py install \
    && rm -rf /tmp/audio/.git
