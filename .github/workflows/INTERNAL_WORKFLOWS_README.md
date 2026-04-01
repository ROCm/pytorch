# Internal ROCm Workflows

This document explains how to adapt PyTorch's upstream workflows for internal ROCm testing.

## Overview

The upstream PyTorch repository uses AWS ECR for Docker images and PyTorch-specific infrastructure (runners, S3 buckets, etc.). For internal testing, we've created simplified workflows that:

1. Use public GHCR images (no ECR login required)
2. Work with internal self-hosted runners
3. Remove PyTorch-specific infrastructure dependencies
4. Use GitHub Actions artifacts instead of S3

## Key Changes from Upstream

### 1. Repository Owner Checks
**Upstream:**
```yaml
if: github.repository_owner == 'pytorch'
```

**Internal:**
```yaml
if: github.repository_owner == 'ROCm'  # or your org name
```

### 2. Docker Images
**Upstream:**
```yaml
- name: Login to ECR
  uses: ./.github/actions/ecr-login

- name: Calculate docker image
  uses: pytorch/test-infra/.github/actions/calculate-docker-image@main
```

**Internal:**
```yaml
- name: Pull Docker Image
  run: |
    # Use public GHCR image directly
    docker pull ghcr.io/pytorch/ci-image:pytorch-linux-jammy-rocm-n-py3-<TAG>
```

### 3. Artifact Storage
**Upstream:**
```yaml
- name: Download from S3
  uses: seemethere/download-artifact-s3@v4
```

**Internal:**
```yaml
- name: Download Artifacts
  uses: actions/download-artifact@v4
  with:
    name: build-artifacts
```

### 4. Runner Labels
**Upstream:**
```yaml
runs-on: linux.rocm.gpu.2
```

**Internal:**
```yaml
runs-on: [self-hosted, rocm, mi200]  # adjust to your labels
```

### 5. External Dependencies Removed
- `pytorch/test-infra` actions
- AWS ECR login
- S3 artifact storage
- Runner determinator
- Scribe logging
- Target determination (optional)

## Workflow Files

### rocm-mi200-simple.yml (Recommended for starting)
A simplified workflow that:
- Builds PyTorch in Docker container using public image
- Tests on self-hosted ROCm MI200 runners
- Uses GitHub Actions artifacts
- 6-way test sharding

**Usage:**
```bash
# Update runner labels in the workflow to match your infrastructure
# Ensure your self-hosted runners have:
# - ROCm 6.0+ installed
# - Labels: [self-hosted, rocm, mi200]
# - Sufficient disk space (100GB+)
```

### rocm-mi200-internal.yml (Full-featured)
A more comprehensive workflow with:
- Separate build and test jobs
- Distributed testing support
- Test result summarization
- Multi-GPU support

**Requirements:**
- Runners with labels `[self-hosted, rocm, mi200]` for single-GPU tests
- Runners with labels `[self-hosted, rocm, mi200, multi-gpu]` for distributed tests

## Setting Up Self-Hosted Runners

### Prerequisites
```bash
# Install ROCm 6.0+
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install GitHub Actions Runner
# Follow: https://docs.github.com/en/actions/hosting-your-own-runners
```

### Configure Runner Labels
When setting up your self-hosted runner, add labels:
- `rocm` - indicates ROCm support
- `mi200` - GPU architecture (or mi210, mi250, mi300, etc.)
- `multi-gpu` - (optional) for runners with 2+ GPUs

### Verify Setup
```bash
# Check ROCm
rocminfo
rocm-smi

# Check Docker
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi

# Check GPU count
rocminfo | grep -c -E 'Name:.*\sgfx'
```

## Customization Guide

### Using Internal Container Registry

Replace public GHCR images with your internal registry:

```yaml
# In build step
- name: Pull Docker Image
  run: |
    # Login to your registry
    echo "${{ secrets.REGISTRY_PASSWORD }}" | docker login \
      your-registry.com -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin

    # Pull image
    docker pull your-registry.com/pytorch-rocm:jammy-py3.10
```

### Adjust Test Sharding

For faster testing, increase shards:

```yaml
strategy:
  matrix:
    include:
      - { shard: 1, num_shards: 12, runner: [self-hosted, rocm, mi200] }
      - { shard: 2, num_shards: 12, runner: [self-hosted, rocm, mi200] }
      # ... up to shard 12
```

### Add Specific Tests

To run only specific test suites:

```yaml
- name: Test
  run: |
    # Run only distributed tests
    python test/run_test.py --verbose --distributed

    # Or specific test files
    python test/test_cuda.py
    python test/test_torch.py
```

### Skip Slow Tests

```yaml
- name: Test
  env:
    PYTORCH_TEST_SKIP_SLOW: 1
  run: |
    python test/run_test.py --verbose
```

## Troubleshooting

### Docker Permission Denied
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### ROCm Not Found
```bash
# Add to ~/.bashrc or runner environment
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Build Timeouts
Increase timeout or reduce jobs:
```yaml
timeout-minutes: 600  # 10 hours
# In build step
-e MAX_JOBS=8  # instead of $(nproc)
```

### Test Failures
Check specific test output:
```bash
# Download artifacts from failed run
# Look in test-results-shard-*/test/**/*.xml
```

### Out of Memory
Reduce concurrent jobs or increase swap:
```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Migration Checklist

- [ ] Update repository owner checks to your organization
- [ ] Configure self-hosted runners with appropriate labels
- [ ] Test Docker image pulling (GHCR or internal registry)
- [ ] Verify ROCm installation on runners
- [ ] Update runner labels in workflow files
- [ ] Test build job locally first
- [ ] Run single test shard to verify test setup
- [ ] Scale to full test matrix
- [ ] Set up test result monitoring
- [ ] Configure artifact retention policies

## Reference

### Original Upstream Files
- `.github/workflows/rocm-mi200.yml` - Main workflow
- `.github/workflows/_linux-build.yml` - Reusable build workflow
- `.github/workflows/_rocm-test.yml` - Reusable test workflow

### Modified Internal Files
- `rocm-mi200-simple.yml` - Simplified self-contained workflow
- `rocm-mi200-internal.yml` - Full-featured internal workflow

### Useful Commands

```bash
# Build locally
docker run -v $PWD:/workspace -w /workspace \
  ghcr.io/pytorch/ci-image:pytorch-linux-jammy-rocm-n-py3-<TAG> \
  bash -c "python tools/amd_build/build_amd.py && pip install -e ."

# Test locally
python test/run_test.py --verbose --shard 1 --num-shards 6

# Check test coverage
python test/run_test.py --list-tests
```
