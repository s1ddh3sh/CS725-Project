
## Setup

- `cd coco_pipeline`

- `pip install torch torchvision transformers pandas scikit-learn pillow tqdm`



- `COCO=/path/to/coco` # contains train_2017/ and val_2017/

- `LDM=/path/to/ldm-text2im-large-256`

- `SD15=/path/to/stable-diffusion-v1-5`

- `SDXLT=/path/to/sdxl-turbo`

<!-- 
## 1. Smoke test

`python smoke_test_coco.py` -->

## 2. Build the image list. 

This uses 10,000 COCO IDs from train_2017: 9,000 for training and 1,000 for validation. It picks them at random with a fixed seed, which represents the dataset better than the first 10,000 by filename.


 `python build_coco_manifest.py --coco_root $COCO \

 --fake ldm=$LDM --fake sd15=$SD15 --fake sdxl_turbo=$SDXLT \

 --out manifests/coco10k.csv --n_train_ids 9000 --n_val_ids 1000 --n_test_ids 2000`

 make changes here to change set size and name of manifest/coco15k.csv

 And accordingly raise the --samples_per_epoch to 14kx4 = 56000

 And then python 



## 3. Train

```bash

python train_coco.py --manifest manifests/coco10k.csv --out_dir runs/coco10k --samples_per_epoch 36000 --batch_size 64 --workers 32
 

 ### only for check

 - `python train_coco.py --manifest manifests/coco10k.csv --out_dir runs/memtest --epochs1 1 --epochs2 1 --samples_per_epoch 2048 --batch_size 64 --workers 8`

## 4. Evaluate

- `python evaluate_coco.py --run_dir runs/coco10k`

 // later, optional: add --synthbuster_root /path/to/synthbuster --sb_jpeg 90`

- `python evaluate_coco.py --run_dir runs/coco15k --synthbuster_root data/synthbuster_final --skip_robustness`
```


# Notes

### 1. There are two stages, s1 and s2, with 8 epochs each by default (--epochs1 8, --epochs2 8). That's 16 epochs in total.



- Estimate 

Stage What trains Time per epoch Stage total

s1 CNN branches, fusion, head (CLIP frozen) 23 min (measured) 8 × 23 ≈ 3 h

s2 Same, plus LoRA in CLIP's last 8 layers ~28–35 min (estimate) ≈ 4–4.5 h

Total ≈ 7–8 h



- Stage 2 is slower because gradients now also flow back through 8 of CLIP ViT-L's 24 layers, which s1 never does. The s2 numbers are my estimate; the first s2 epoch will give you the real figure.



- That estimate is for the 10k run (--samples_per_epoch 36000). The 15k run with --samples_per_epoch 56000 has about 1.55× as many steps per epoch, so roughly 11–12 hours in total.

```bash
`python train_coco.py --manifest manifests/coco10k.csv --out_dir runs/coco10k_s2 \

 --resume runs/coco10k/best.pt --epochs1 0 --samples_per_epoch 36000 --batch_size 64 --w````