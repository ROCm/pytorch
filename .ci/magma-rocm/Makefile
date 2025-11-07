SHELL=/usr/bin/env bash

DOCKER_CMD ?= docker
<<<<<<< HEAD
DESIRED_ROCM ?= 7.0
DESIRED_ROCM_SHORT = $(subst .,,$(DESIRED_ROCM))
PACKAGE_NAME = magma-rocm
# inherit this from underlying docker image, do not pass this env var to docker
#PYTORCH_ROCM_ARCH ?= gfx900;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201
=======
DESIRED_ROCM ?= 6.4
DESIRED_ROCM_SHORT = $(subst .,,$(DESIRED_ROCM))
PACKAGE_NAME = magma-rocm
# inherit this from underlying docker image, do not pass this env var to docker
#PYTORCH_ROCM_ARCH ?= gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

DOCKER_RUN = set -eou pipefail; ${DOCKER_CMD} run --rm -i \
	-v $(shell git rev-parse --show-toplevel)/.ci:/builder \
	-w /builder \
	-e PACKAGE_NAME=${PACKAGE_NAME}${DESIRED_ROCM_SHORT} \
	-e DESIRED_ROCM=${DESIRED_ROCM} \
	"pytorch/almalinux-builder:rocm${DESIRED_ROCM}" \
	magma-rocm/build_magma.sh

.PHONY: all
<<<<<<< HEAD
all: magma-rocm70
all: magma-rocm64
=======
all: magma-rocm64
all: magma-rocm63
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

.PHONY:
clean:
	$(RM) -r magma-*
	$(RM) -r output

<<<<<<< HEAD
.PHONY: magma-rocm70
magma-rocm70: DESIRED_ROCM := 7.0
magma-rocm70:
	$(DOCKER_RUN)

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
.PHONY: magma-rocm64
magma-rocm64: DESIRED_ROCM := 6.4
magma-rocm64:
	$(DOCKER_RUN)
<<<<<<< HEAD
=======

.PHONY: magma-rocm63
magma-rocm63: DESIRED_ROCM := 6.3
magma-rocm63:
	$(DOCKER_RUN)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
