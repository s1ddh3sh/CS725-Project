# OLD runner

- python build_coco_manifest.py --coco_root /data/coco \
    --fake ldm=/data/DRCT-2M/ldm-text2im-large-256 \
    --fake sd15=/data/DRCT-2M/stable-diffusion-v1-5 \
    --fake sdxl_turbo=/data/DRCT-2M/sdxl-turbo \
    --out /data/coco_manifest.csv --n_train_ids 20000 --n_val_ids 1000

- python train_coco.py --manifest /data/coco_manifest.csv --out_dir runs/coco_v1

- python evaluate_coco.py --run_dir runs/coco_v1 --synthbuster_root /data/synthbuster --sb_jpeg 90

# NEW Runner 

## Prerun installations
- `cd coco_pipeline`
- `pip install torch torchvision transformers pandas scikit-learn pillow tqdm`

- `COCO=/path/to/coco`              # contains train_2017/ and val_2017/
- `LDM=/path/to/ldm-text2im-large-256`
- `SD15=/path/to/stable-diffusion-v1-5`
- `SDXLT=/path/to/sdxl-turbo`

## 1. Smoke test
`python smoke_test_coco.py`

## 2. Build the image list. 
This uses 10,000 COCO IDs from train_2017: 9,000 for training and 1,000 for validation. It picks them at random with a fixed seed, which represents the dataset better than the first 10,000 by filename.

 `python build_coco_manifest.py --coco_root $COCO \
    --fake ldm=$LDM --fake sd15=$SD15 --fake sdxl_turbo=$SDXLT \
    --out manifests/coco10k.csv --n_train_ids 9000 --n_val_ids 1000 --n_test_ids 2000`
    make changes here to change set size and name of manifest/coco15k.csv
    And accordingly raise the --samples_per_epoch to 14kx4 = 56000
    And then python 

## 3. Train
`python train_coco.py --manifest manifests/coco10k.csv --out_dir runs/coco10k \
    --samples_per_epoch 36000 --batch_size 64 --workers 8`

    `python train_coco.py --manifest manifests/coco10k.csv --out_dir runs/coco10k \
    --samples_per_epoch 36000 --batch_size 64 --workers 12` // use this
    
    ### only for check
    ` python train_coco.py --manifest manifests/coco10k.csv --out_dir runs/memtest \
    --epochs1 1 --epochs2 1 --samples_per_epoch 2048 --batch_size 64 --workers 8`
## 4. Evaluate
`python evaluate_coco.py --run_dir runs/coco10k`
    // later, optional: add  --synthbuster_root /path/to/synthbuster --sb_jpeg 90`