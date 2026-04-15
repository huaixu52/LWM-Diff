# LWM-Diffusion Policy for Autonomous Robotic Ultrasound

Code for the paper: *Mitigating Covariate Shift in Imitation Learning for Autonomous Robotic Ultrasound Using Latent World Model*

## Method

LWM-Diffusion Policy is a hierarchical visuomotor policy that:
1. Predicts a coarse action chunk from the observation history
2. Rolls these actions through a latent world model to forecast future states
3. Refines the output with a conditional diffusion decoder

The closed-loop latent rollout exposes the policy to its own compounding errors during training, mitigating covariate shift without online environment interaction.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

This code uses the [RACINES](https://arxiv.org/abs/2511.00114) cardiac ultrasound dataset.We thank the authors for making their data publicly available. Set `root_dir` in the config YAML to your local path.

## Mutual information-based Encoder
The encoder used is from [MI-GPSR](https://github.com/yuan-12138/MI-GPSR). We thank the authors for their excellent work. We retrained the encoder which can be found at https://drive.google.com/file/d/1XSNIVocFEbkFHfTCxShrdPAZGk7wTg71/view?usp=drive_link.

## Training

### LWM-Diffusion Policy (proposed method)

```bash
python scripts/train_lmw_diffusion_policy.py \
    --config scripts/configs/racines/lmw_diffusion_policy.yaml \
    --root_dir /path/to/racines/
```

### Ablation: w/o diffusion head (LWM Policy)

```bash
python scripts/train_lmw_policy.py \
    --config scripts/configs/racines/lmw_policy.yaml \
    --root_dir /path/to/racines/
```

### Other ablations (all via CLI flags)

```bash
# w/o two-stage training
python scripts/train_lmw_diffusion_policy.py \
    --config scripts/configs/racines/lmw_diffusion_policy.yaml \
    --training_stage1_epochs 0

# w/o latent regularization
python scripts/train_lmw_diffusion_policy.py \
    --config scripts/configs/racines/lmw_diffusion_policy.yaml \
    --lambda_latent 0

# Observation horizon experiment
python scripts/train_lmw_diffusion_policy.py \
    --config scripts/configs/racines/lmw_diffusion_policy.yaml \
    --obs_horizon 8
```

### OOD robustness evaluation

```bash
python scripts/train_lmw_diffusion_policy.py \
    --config scripts/configs/racines/lmw_diffusion_policy.yaml \
    --ood_corruption_type all
```
