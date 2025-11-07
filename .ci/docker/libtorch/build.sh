#!/usr/bin/env bash
# Script used only in CD pipeline

<<<<<<< HEAD
set -eou pipefail
=======
set -eoux pipefail
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

image="$1"
shift

if [ -z "${image}" ]; then
<<<<<<< HEAD
  echo "Usage: $0 IMAGE"
  exit 1
fi

DOCKER_IMAGE="pytorch/${image}"

TOPDIR=$(git rev-parse --show-toplevel)

GPU_ARCH_TYPE=${GPU_ARCH_TYPE:-cpu}
GPU_ARCH_VERSION=${GPU_ARCH_VERSION:-}

WITH_PUSH=${WITH_PUSH:-}

DOCKER=${DOCKER:-docker}

case ${GPU_ARCH_TYPE} in
    cpu)
        BASE_TARGET=cpu
        DOCKER_TAG=cpu
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    cuda)
        BASE_TARGET=cuda${GPU_ARCH_VERSION}
        DOCKER_TAG=cuda${GPU_ARCH_VERSION}
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    rocm)
        BASE_TARGET=rocm
        DOCKER_TAG=rocm${GPU_ARCH_VERSION}
        GPU_IMAGE=rocm/dev-ubuntu-20.04:${GPU_ARCH_VERSION}-complete
        PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
        DOCKER_GPU_BUILD_ARG="--build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
        ;;
    *)
        echo "ERROR: Unrecognized GPU_ARCH_TYPE: ${GPU_ARCH_TYPE}"
=======
  echo "Usage: $0 IMAGENAME:ARCHTAG"
  exit 1
fi

TOPDIR=$(git rev-parse --show-toplevel)

DOCKER=${DOCKER:-docker}

# Go from imagename:tag to tag
DOCKER_TAG_PREFIX=$(echo "${image}" | awk -F':' '{print $2}')

GPU_ARCH_VERSION=""
if [[ "${DOCKER_TAG_PREFIX}" == cuda* ]]; then
    # extract cuda version from image name.  e.g. manylinux2_28-builder:cuda12.8 returns 12.8
    GPU_ARCH_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'cuda' '{print $2}')
elif [[ "${DOCKER_TAG_PREFIX}" == rocm* ]]; then
    # extract rocm version from image name.  e.g. manylinux2_28-builder:rocm6.2.4 returns 6.2.4
    GPU_ARCH_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'rocm' '{print $2}')
fi

case ${DOCKER_TAG_PREFIX} in
    cpu)
        BASE_TARGET=cpu
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    cuda*)
        BASE_TARGET=cuda${GPU_ARCH_VERSION}
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    rocm*)
        BASE_TARGET=rocm
        GPU_IMAGE=rocm/dev-ubuntu-22.04:${GPU_ARCH_VERSION}-complete
        PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
        DOCKER_GPU_BUILD_ARG="--build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} --build-arg ROCM_VERSION=${GPU_ARCH_VERSION}"
        ;;
    *)
        echo "ERROR: Unrecognized DOCKER_TAG_PREFIX: ${DOCKER_TAG_PREFIX}"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        exit 1
        ;;
esac

<<<<<<< HEAD

(
    set -x
    DOCKER_BUILDKIT=1 ${DOCKER} build \
         --target final \
        ${DOCKER_GPU_BUILD_ARG} \
        --build-arg "GPU_IMAGE=${GPU_IMAGE}" \
        --build-arg "BASE_TARGET=${BASE_TARGET}" \
        -t "${DOCKER_IMAGE}" \
        $@ \
        -f "${TOPDIR}/.ci/docker/libtorch/Dockerfile" \
        "${TOPDIR}/.ci/docker/"

)

GITHUB_REF=${GITHUB_REF:-$(git symbolic-ref -q HEAD || git describe --tags --exact-match)}
GIT_BRANCH_NAME=${GITHUB_REF##*/}
GIT_COMMIT_SHA=${GITHUB_SHA:-$(git rev-parse HEAD)}
DOCKER_IMAGE_BRANCH_TAG=${DOCKER_IMAGE}-${GIT_BRANCH_NAME}
DOCKER_IMAGE_SHA_TAG=${DOCKER_IMAGE}-${GIT_COMMIT_SHA}

if [[ "${WITH_PUSH}" == true ]]; then
  (
    set -x
    ${DOCKER} push "${DOCKER_IMAGE}"
    if [[ -n ${GITHUB_REF} ]]; then
        ${DOCKER} tag ${DOCKER_IMAGE} ${DOCKER_IMAGE_BRANCH_TAG}
        ${DOCKER} tag ${DOCKER_IMAGE} ${DOCKER_IMAGE_SHA_TAG}
        ${DOCKER} push "${DOCKER_IMAGE_BRANCH_TAG}"
        ${DOCKER} push "${DOCKER_IMAGE_SHA_TAG}"
    fi
  )
fi
=======
tmp_tag=$(basename "$(mktemp -u)" | tr '[:upper:]' '[:lower:]')

DOCKER_BUILDKIT=1 ${DOCKER} build \
    --target final \
    ${DOCKER_GPU_BUILD_ARG} \
    --build-arg "GPU_IMAGE=${GPU_IMAGE}" \
    --build-arg "BASE_TARGET=${BASE_TARGET}" \
    -t "${tmp_tag}" \
    $@ \
    -f "${TOPDIR}/.ci/docker/libtorch/Dockerfile" \
    "${TOPDIR}/.ci/docker/"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
