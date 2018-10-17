# DeblurGAN

## How to run

### Data
Download dataset for Object Detection benchmark from [Google Drive](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view) and unpack it to the GOPRO_Large folder.

### Train

```bash
python3 train.py --dataroot GOPRO_Large --learn_residual --resize_or_crop crop --fineSize 256 --niter 25 --niter_decay 25 --name exp1
```
#### Main changes:
1) in data/unaligned_dataset.py
2) image_folder.py




