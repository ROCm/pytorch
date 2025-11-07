#!/usr/bin/env bash
# Script used only in CD pipeline

<<<<<<< HEAD
set -eou pipefail
=======
set -exou pipefail
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

TOPDIR=$(git rev-parse --show-toplevel)

image="$1"
shift

if [ -z "${image}" ]; then
<<<<<<< HEAD
  echo "Usage: $0 IMAGE"
  exit 1
fi

DOCKER_IMAGE="pytorch/${image}"

DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"

GPU_ARCH_TYPE=${GPU_ARCH_TYPE:-cpu}
GPU_ARCH_VERSION=${GPU_ARCH_VERSION:-}
MANY_LINUX_VERSION=${MANY_LINUX_VERSION:-}
DOCKERFILE_SUFFIX=${DOCKERFILE_SUFFIX:-}
WITH_PUSH=${WITH_PUSH:-}

case ${GPU_ARCH_TYPE} in
    cpu)
        TARGET=cpu_final
        DOCKER_TAG=cpu
        GPU_IMAGE=centos:7
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=9"
        ;;
    cpu-manylinux_2_28)
        TARGET=cpu_final
        DOCKER_TAG=cpu
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=11"
        MANY_LINUX_VERSION="2_28"
        ;;
    cpu-aarch64)
        TARGET=final
        DOCKER_TAG=cpu-aarch64
        GPU_IMAGE=arm64v8/centos:7
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=10"
        MANY_LINUX_VERSION="aarch64"
        ;;
    cpu-aarch64-2_28)
        TARGET=final
        DOCKER_TAG=cpu-aarch64
        GPU_IMAGE=arm64v8/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=11 --build-arg NINJA_VERSION=1.12.1"
        MANY_LINUX_VERSION="2_28_aarch64"
        ;;
    cpu-cxx11-abi)
        TARGET=final
        DOCKER_TAG=cpu-cxx11-abi
=======
  echo "Usage: $0 IMAGE:ARCHTAG"
  exit 1
fi

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

MANY_LINUX_VERSION=${MANY_LINUX_VERSION:-}
DOCKERFILE_SUFFIX=${DOCKERFILE_SUFFIX:-}
OPENBLAS_VERSION=${OPENBLAS_VERSION:-}

case ${image} in
    manylinux2_28-builder:cpu)
        TARGET=cpu_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="2_28"
        ;;
    manylinux2_28_aarch64-builder:cpu-aarch64)
        TARGET=final
        GPU_IMAGE=arm64v8/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=13 --build-arg NINJA_VERSION=1.12.1"
        MANY_LINUX_VERSION="2_28_aarch64"
        OPENBLAS_VERSION="v0.3.30"
        ;;
    manylinuxcxx11-abi-builder:cpu-cxx11-abi)
        TARGET=final
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        GPU_IMAGE=""
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=9"
        MANY_LINUX_VERSION="cxx11-abi"
        ;;
<<<<<<< HEAD
    cpu-s390x)
        TARGET=final
        DOCKER_TAG=cpu-s390x
=======
    manylinuxs390x-builder:cpu-s390x)
        TARGET=final
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        GPU_IMAGE=s390x/almalinux:8
        DOCKER_GPU_BUILD_ARG=""
        MANY_LINUX_VERSION="s390x"
        ;;
<<<<<<< HEAD
    cuda)
        TARGET=cuda_final
        DOCKER_TAG=cuda${GPU_ARCH_VERSION}
        # Keep this up to date with the minimum version of CUDA we currently support
        GPU_IMAGE=centos:7
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=9"
        ;;
    cuda-manylinux_2_28)
        TARGET=cuda_final
        DOCKER_TAG=cuda${GPU_ARCH_VERSION}
=======
    manylinux2_28-builder:cuda11*)
        TARGET=cuda_final
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=11"
        MANY_LINUX_VERSION="2_28"
        ;;
<<<<<<< HEAD
    cuda-aarch64)
        TARGET=cuda_final
        DOCKER_TAG=cuda${GPU_ARCH_VERSION}
        GPU_IMAGE=arm64v8/centos:7
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=11"
        MANY_LINUX_VERSION="aarch64"
        DOCKERFILE_SUFFIX="_cuda_aarch64"
        ;;
    rocm|rocm-manylinux_2_28)
        TARGET=rocm_final
        DOCKER_TAG=rocm${GPU_ARCH_VERSION}
        GPU_IMAGE=rocm/dev-centos-7:${GPU_ARCH_VERSION}-complete
        DEVTOOLSET_VERSION="9"
        if [ ${GPU_ARCH_TYPE} == "rocm-manylinux_2_28" ]; then
            MANY_LINUX_VERSION="2_28"
            DEVTOOLSET_VERSION="11"
            GPU_IMAGE=rocm/dev-almalinux-8:${GPU_ARCH_VERSION}-complete
        fi
        PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
        DOCKER_GPU_BUILD_ARG="--build-arg ROCM_VERSION=${GPU_ARCH_VERSION} --build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} --build-arg DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}"
        ;;
    xpu)
        TARGET=xpu_final
        DOCKER_TAG=xpu
=======
    manylinux2_28-builder:cuda12*)
        TARGET=cuda_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="2_28"
        ;;
    manylinuxaarch64-builder:cuda*)
        TARGET=cuda_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="aarch64"
        DOCKERFILE_SUFFIX="_cuda_aarch64"
        ;;
    manylinux2_28-builder:rocm*)
        TARGET=rocm_final
        MANY_LINUX_VERSION="2_28"
        DEVTOOLSET_VERSION="11"
        GPU_IMAGE=rocm/dev-almalinux-8:${GPU_ARCH_VERSION}-complete
        PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
        DOCKER_GPU_BUILD_ARG="--build-arg ROCM_VERSION=${GPU_ARCH_VERSION} --build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} --build-arg DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}"
        ;;
    manylinux2_28-builder:xpu)
        TARGET=xpu_final
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=11"
        MANY_LINUX_VERSION="2_28"
        ;;
    *)
<<<<<<< HEAD
        echo "ERROR: Unrecognized GPU_ARCH_TYPE: ${GPU_ARCH_TYPE}"
=======
        echo "ERROR: Unrecognized image name: ${image}"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        exit 1
        ;;
esac

<<<<<<< HEAD
IMAGES=''

if [[ -n ${MANY_LINUX_VERSION} && -z ${DOCKERFILE_SUFFIX} ]]; then
    DOCKERFILE_SUFFIX=_${MANY_LINUX_VERSION}
fi
(
    set -x

    # Only activate this if in CI
    if [ "$(uname -m)" != "s390x" ] && [ -v CI ]; then
        # TODO: Remove LimitNOFILE=1048576 patch once https://github.com/pytorch/test-infra/issues/5712
        # is resolved. This patch is required in order to fix timing out of Docker build on Amazon Linux 2023.
        sudo sed -i s/LimitNOFILE=infinity/LimitNOFILE=1048576/ /usr/lib/systemd/system/docker.service
        sudo systemctl daemon-reload
        sudo systemctl restart docker
    fi

    DOCKER_BUILDKIT=1 docker build  \
        ${DOCKER_GPU_BUILD_ARG} \
        --build-arg "GPU_IMAGE=${GPU_IMAGE}" \
        --target "${TARGET}" \
        -t "${DOCKER_IMAGE}" \
        $@ \
        -f "${TOPDIR}/.ci/docker/manywheel/Dockerfile${DOCKERFILE_SUFFIX}" \
        "${TOPDIR}/.ci/docker/"
)

GITHUB_REF=${GITHUB_REF:-"dev")}
GIT_BRANCH_NAME=${GITHUB_REF##*/}
GIT_COMMIT_SHA=${GITHUB_SHA:-$(git rev-parse HEAD)}
DOCKER_IMAGE_BRANCH_TAG=${DOCKER_IMAGE}-${GIT_BRANCH_NAME}
DOCKER_IMAGE_SHA_TAG=${DOCKER_IMAGE}-${GIT_COMMIT_SHA}

if [[ "${WITH_PUSH}" == true ]]; then
    (
        set -x
        docker push "${DOCKER_IMAGE}"
        if [[ -n ${GITHUB_REF} ]]; then
            docker tag ${DOCKER_IMAGE} ${DOCKER_IMAGE_BRANCH_TAG}
            docker tag ${DOCKER_IMAGE} ${DOCKER_IMAGE_SHA_TAG}
            docker push "${DOCKER_IMAGE_BRANCH_TAG}"
            docker push "${DOCKER_IMAGE_SHA_TAG}"
        fi
    )
fi
=======
if [[ -n ${MANY_LINUX_VERSION} && -z ${DOCKERFILE_SUFFIX} ]]; then
    DOCKERFILE_SUFFIX=_${MANY_LINUX_VERSION}
fi
# Only activate this if in CI
if [ "$(uname -m)" != "s390x" ] && [ -v CI ]; then
    # TODO: Remove LimitNOFILE=1048576 patch once https://github.com/pytorch/test-infra/issues/5712
    # is resolved. This patch is required in order to fix timing out of Docker build on Amazon Linux 2023.
    sudo sed -i s/LimitNOFILE=infinity/LimitNOFILE=1048576/ /usr/lib/systemd/system/docker.service
    sudo systemctl daemon-reload
    sudo systemctl restart docker
fi

tmp_tag=$(basename "$(mktemp -u)" | tr '[:upper:]' '[:lower:]')

DOCKER_BUILDKIT=1 docker build  \
    ${DOCKER_GPU_BUILD_ARG} \
    --build-arg "GPU_IMAGE=${GPU_IMAGE}" \
    --build-arg "OPENBLAS_VERSION=${OPENBLAS_VERSION}" \
    --target "${TARGET}" \
    -t "${tmp_tag}" \
    $@ \
    -f "${TOPDIR}/.ci/docker/manywheel/Dockerfile${DOCKERFILE_SUFFIX}" \
    "${TOPDIR}/.ci/docker/"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
