# LWM-Diffusion Policy for Autonomous Robotic Ultrasound

TODO:The dataset preprocessing and experimental setup sections have been uploaded. The remaining components — model architecture, training scripts, and evaluation code — will be released in full upon paper acceptance.


## Setup

```bash
pip install -r requirements.txt
```

## Mutual information-based Encoder
The encoder used can be found at https://drive.google.com/file/d/1XSNIVocFEbkFHfTCxShrdPAZGk7wTg71/view?usp=drive_link.

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
