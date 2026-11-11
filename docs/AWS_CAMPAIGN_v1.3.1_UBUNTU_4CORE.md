# AWS campaign profile — MC DSL v1.3.1 (Ubuntu 24.04, 4 physical cores)

This profile is a complementary campaign, not a replacement for the Windows campaign. It writes to a separate results directory and checkpoint directory. It adds a true in-process serial baseline to Exp. 2 and four-worker memory measurements to Exp. 8.

## Target instance

Recommended target: `c7i.2xlarge` (8 vCPU, 16 GiB; AWS documents 4 cores with 2 threads/core). Confirm that the exact size is offered in the selected Availability Zone before launch. The `C7i` family is listed for `mx-central-1`, but exact-size capacity is subject to AZ capacity and account quotas.

Use Ubuntu Server 24.04 LTS x86_64, On-Demand (not Spot, to avoid planned interruption), 40 GiB gp3 EBS, and a security group with TCP/22 restricted to your current public IP. Do not open SSH to `0.0.0.0/0`.

## Local PowerShell: verify the instance and Ubuntu AMI (AWS CLI)

```powershell
aws ec2 describe-instance-types `
  --region mx-central-1 `
  --instance-types c7i.2xlarge `
  --query 'InstanceTypes[0].{vCPUs:VCpuInfo.DefaultVCpus,Cores:VCpuInfo.DefaultCores,ThreadsPerCore:VCpuInfo.DefaultThreadsPerCore,MemoryMiB:MemoryInfo.SizeInMiB}' `
  --output table

aws ec2 describe-instance-type-offerings `
  --region mx-central-1 `
  --location-type availability-zone `
  --filters Name=instance-type,Values=c7i.2xlarge `
  --query 'InstanceTypeOfferings[].Location' `
  --output table

aws ssm get-parameter `
  --region mx-central-1 `
  --name /aws/service/canonical/ubuntu/server/noble/stable/current/amd64/hvm/ebs-gp3/ami-id `
  --query Parameter.Value `
  --output text
```

The first command should report 8 vCPU, 4 cores, 2 threads/core and 16 GiB. If no Availability Zone is returned, choose another region that lists C7i and check the exact type there. Also verify the account's On-Demand vCPU quota and current capacity before launching. The SSM parameter resolves Canonical's current Ubuntu 24.04 LTS AMD64 AMI in that region.

## Launch in the AWS Console

1. EC2 → Launch instance.
2. Name: `MC-DSL-v1.3.1-AWS-4core`.
3. AMI: Ubuntu Server 24.04 LTS, 64-bit x86.
4. Instance type: `c7i.2xlarge`.
5. Create/select an EC2 key pair and download the `.pem` file.
6. Security group: inbound SSH TCP 22, source `My IP` only.
7. Storage: 40 GiB gp3 (results/checkpoints need space; monitor disk use).
8. Purchase option: On-Demand. Do not use Spot for the primary confirmatory campaign.
9. Launch and note public IPv4/DNS and instance ID.

## Connect from Windows PowerShell

```powershell
ssh -i "$env:USERPROFILE\Downloads\YOUR_KEY.pem" ubuntu@YOUR_EC2_PUBLIC_DNS
```

If Windows rejects key permissions, run:

```powershell
icacls "$env:USERPROFILE\Downloads\YOUR_KEY.pem" /inheritance:r
icacls "$env:USERPROFILE\Downloads\YOUR_KEY.pem" /grant:r "$($env:USERNAME):(R)"
```

## Ubuntu setup

```bash
sudo apt update
sudo apt install -y build-essential git unzip tmux util-linux curl

# Install uv, then Python 3.11.9 (same Python patch release as the Windows campaign).
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.11.9

mkdir -p ~/work/mc-dsl-v1.3.1-aws
cd ~/work/mc-dsl-v1.3.1-aws
```

Create a clean source ZIP on Windows from the patched project root. The packager excludes `.venv`, `.git`, prior `benchmarks/results`, `benchmarks/checkpoints`, generated `docs/tables`/`docs/figures`, logs and caches:

```powershell
python scripts/package_aws_source.py --output .\dist\Montecarlo_DSL_Research_v1.3.1_aws_source.zip
```

Transfer that source ZIP from local PowerShell:

```powershell
scp -i "$env:USERPROFILE\Downloads\YOUR_KEY.pem" `
  ".\dist\Montecarlo_DSL_Research_v1.3.1_aws_source.zip" `
  ubuntu@YOUR_EC2_PUBLIC_DNS:~/work/mc-dsl-v1.3.1-aws/
```

On Ubuntu:

```bash
cd ~/work/mc-dsl-v1.3.1-aws
unzip Montecarlo_DSL_Research_v1.3.1_aws_source.zip
cd Montecarlo_DSL_Research

uv venv --python 3.11.9 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-aws-experiment.txt
python -m pip check
```

The pinned versions are taken from the completed Windows campaign's `dependencies-publication.txt` / `environment.json`. Keep them fixed for the cross-platform comparison. If a pinned wheel is unavailable for Linux/Python 3.11, stop and report the pip error; do not silently upgrade dependencies.

## Capture hardware and environment before running

```bash
mkdir -p logs
{
  date -u --iso-8601=seconds
  uname -a
  lscpu
  nproc
  free -h
  df -h .
  python --version
  python -m pip freeze
} | tee logs/aws-hardware-environment.txt
```

Expected for `c7i.2xlarge`: 8 logical CPUs, 4 physical cores, 16 GiB RAM. Treat `lscpu` as the authoritative recorded topology; do not infer physical cores from `nproc` alone.

## Validate before the long campaign

```bash
python -m pytest -q
python benchmarks/rng_reproducibility.py \
  --models normal_mean --iterations 100000 --repetitions 2 \
  --output-dir .pytest_tmp/rng_smoke_aws
```

Both commands must pass. Inspect `lscpu` and `logs/aws-hardware-environment.txt` before proceeding.

## Start AWS-specific campaign with checkpoints

Use separate paths so AWS results cannot overwrite Windows results:

```bash
mkdir -p logs

tmux new -s mcdsl-aws
```

Inside tmux:

```bash
source .venv/bin/activate
set -o pipefail
python -u benchmarks/run_v1_3_experiments.py \
  --profile aws-4core \
  --instance-type c7i.2xlarge \
  --cloud-region mx-central-1 \
  --campaign-label aws-c7i-2xlarge-ubuntu-24.04 \
  --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10 \
  --checkpoint-dir benchmarks/checkpoints/v1.3.1_aws_c7i_4core \
  --output-dir benchmarks/results/final/v1.3.1_aws_c7i_4core \
  2>&1 | tee logs/experiments-v1.3.1-aws.log
```

Detach from tmux with `Ctrl+B`, then `D`. Reattach with `tmux attach -t mcdsl-aws`. To resume after interruption, run the same command with `--resume` appended. Do not change code, dependency versions, repetition counts, profile, instance type or output/checkpoint paths when resuming.

## What this profile changes

- Exp. 2 adds `v1.3_serial_in_process` (`ast_only`, one worker, no `ProcessPoolExecutor`) to distinguish compute speed from process-pool overhead. Existing fixed pool w=1, w=2, w=4 and adaptive hybrid variants remain.
- The 10-million-iteration Exp. 2 comparison also includes the serial in-process baseline.
- Exp. 8 adds `multiprocess_w4` at 100k, 1M and 2.5M iterations for both `normal_mean` and `quadratic_pm`, under `retain_all` and `bounded`.
- The campaign writes to the specified AWS output path and records profile, instance type, region, logical CPU count and detected physical core topology in `environment.json`.

## Generate separate AWS tables and figures

After the campaign finishes successfully:

```bash
python scripts/generate_v1_3_figures_tables.py \
  --results-dir benchmarks/results/final/v1.3.1_aws_c7i_4core \
  --tables-dir docs/tables/v1.3.1_aws_c7i_4core \
  --figures-dir docs/figures/v1.3.1_aws_c7i_4core
```

Do not merge AWS and Windows raw rows into one table of pooled timings. Compare platform-specific medians/intervals side by side and state that CPU model, OS, kernel, virtualization and memory subsystem differ.

## Retrieve results to Windows

From local PowerShell:

```powershell
scp -i "$env:USERPROFILE\Downloads\YOUR_KEY.pem" -r `
  ubuntu@YOUR_EC2_PUBLIC_DNS:~/work/mc-dsl-v1.3.1-aws/Montecarlo_DSL_Research/benchmarks/results/final/v1.3.1_aws_c7i_4core `
  "$env:USERPROFILE\Downloads\MC_DSL_AWS_Results"

scp -i "$env:USERPROFILE\Downloads\YOUR_KEY.pem" -r `
  ubuntu@YOUR_EC2_PUBLIC_DNS:~/work/mc-dsl-v1.3.1-aws/Montecarlo_DSL_Research/docs/tables/v1.3.1_aws_c7i_4core `
  "$env:USERPROFILE\Downloads\MC_DSL_AWS_Tables"
```

Also retrieve `logs/aws-hardware-environment.txt`, `logs/experiments-v1.3.1-aws.log`, the checkpoint `campaign_status.json`, and `environment.json` for the paper's artifact package.

## Stop billing

After copying all results, stop or terminate the EC2 instance in the AWS Console. Stopping an EBS-backed instance stops compute charges but EBS storage continues to incur charges; terminate it if you no longer need the instance and have safely copied the results/checkpoints.
