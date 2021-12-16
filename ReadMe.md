# Environment
We provide I2V_attack-env.yml to recover the used environment.
```
conda env create -f I2V_attack-env.yml
```
## GPU infos
```
NVIDIA GeForce RTX 2080TI
NVIDIA-SMI 430.14       Driver Version: 430.14       CUDA Version: 10.2 
```

# Attacked Dataset
The used datasets are sampled from UCF101 and Kinetics-400. Download attacked datasets from [here](https://drive.google.com/drive/folders/1O4XyLw37WqGKqFvWFaE2ps5IAD_shSpG?usp=sharing). 
Change the **UCF_DATA_ROOT** and **Kinetic_DATA_ROOT** of utils.py into your dataset path.

# Models
Non-local, SlowFast, TPN with ResNet-50 and ResNet-101 as backbones are used here.
## UCF101
We fine-tune video models on UCF101.
Download checkpoint files from [here](https://drive.google.com/drive/folders/10KOlWdi5bsV9001uL4Bn1T48m9hkgsZ2?usp=sharing).
Change the **UCF_MODEL_ROOT** of utils.py into your checkpoint path.

## Kinetics-400
We use pretrained models on Kinetics-400 from [gluoncv](https://cv.gluon.ai/model_zoo/action_recognition.html) to conduct experiments.

# Attack
Assign your output path to **OPT_PATH** of utils.py.
## Generate adversarial examples.
```
python attack_kinetics.py/attack_ucf101.py --gpu 0 --batch_size 1 --model slowfast_resnet101 --attack_method TemporalTranslation --step 10 --file_prefix yours --momentum --kernlen 15 --move_type adj --kernel_mode gaussian
```
* model: the white-box model
* attack_method: TemporalTranslation(TT/TT-MI) or TemporalTranslation_TI(TT-TI)
* step: the attack step
* file_prefix: additional names for the output file
* momentum: TT-MI
* kernlen: 2 * (Shift Length) + 1
* move_type: shifting strategies
* kernel_mode: weight matrix generation strategies

## Attack Success rate
```
python reference_kinetics.py/reference_ucf101.py --gpu 0 --adv_path your_adv_path
```
* adv_path: name of the output file 
