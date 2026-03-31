# Parity Report Workflow

Generates ROCm vs CUDA test parity CSVs for MI200, MI300, and MI355 architectures.

## Quick Start

Trigger from the Actions tab or CLI:

```bash
gh workflow run parity.yml --ref add-parity-scripts-dashboard
```

The workflow:
1. Downloads CI test artifacts (XML reports) for each architecture
2. Generates a per-architecture CSV comparing ROCm and CUDA test results
3. Produces a combined summary with per-workflow stats and overall parity metrics
4. Displays the summary as a markdown table on the workflow run page

## Weekly Process

### 1. Run the parity workflow

Trigger with default inputs or specify a commit SHA:

```bash
gh workflow run parity.yml -f sha=<commit_sha> -f csv_name=march_w3
```

The workflow automatically looks for a previous week's CSV to carry forward
`skip_reason`, `assignee`, `comments`, and `existed_last_week` columns:
- **First**: checks the `parity-input` GitHub Release for a user-uploaded CSV
- **Fallback**: downloads the CSV artifact from the last successful parity run

### 2. Review and edit skip reasons

Download the generated CSV artifact from the workflow run page. Open it and
fill in `skip_reason`, `assignee`, and `comments` for any new skipped/missed
tests that don't already have labels.

### 3. Upload the edited CSV

Upload your edited CSV to the `parity-input` release so the next run picks it up:

```bash
gh release upload parity-input my_edited_all_tests_status.csv --clobber -R ROCm/pytorch
```

If the release doesn't exist yet (first-time setup):

```bash
gh release create parity-input --title "Parity Input CSV" \
  --notes "Upload edited parity CSVs here. The parity workflow downloads from this release automatically." \
  -R ROCm/pytorch
gh release upload parity-input my_edited_all_tests_status.csv -R ROCm/pytorch
```

### 4. Next week

The next parity run will automatically download your edited CSV and carry forward
all skip reasons. Only newly skipped/missed tests will have empty labels.

## Workflow Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `sha` | *(latest green)* | PyTorch commit SHA |
| `pr_id` | | Pull request ID (alternative to SHA) |
| `arch` | `mi355, mi300, mi200` | Architectures (comma or space separated) |
| `csv_name` | | Custom prefix for output filenames and artifacts |
| `exclude_distributed` | `false` | Skip distributed tests |
| `exclude_inductor` | `false` | Skip inductor tests |
| `exclude_default` | `false` | Skip default tests |
| `include_logs` | `true` | Download and include CI log files in artifact |
| `skip_rocm` | `false` | Skip ROCm artifact download |
| `skip_cuda` | `false` | Skip CUDA artifact download |
| `set1_name` | `rocm` | Column header for set1 |
| `set2_name` | `cuda` | Column header for set2 |
| `filter` | | Status filter (e.g. `SKIPPED-NOT_SKIPPED-MISSED-NOT_MISSED`) |

## Outputs

Each run produces:
- **Per-architecture artifacts** (`parity-csv-{arch}` or `{csv_name}_{arch}`): CSV, running time CSV, processing logs, and CI log files
- **Summary artifact** (`parity-summary` or `{csv_name}_summary`): Combined summary CSV and markdown
- **Job summary**: Markdown table rendered on the workflow run page with per-workflow stats, failed tests, and artifact download links

## Required Secrets

| Secret | Description |
|--------|-------------|
| `IFU_GITHUB_TOKEN` | GitHub token for downloading CI artifacts |
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 artifact access |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 artifact access |
