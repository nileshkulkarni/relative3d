### Input
1. Image
2. Image RoI
3. Use common mask
4. Use spatial encoder
5. Upsample mask

### Train  Time
1. Train for 8 epochs, batch_size=24, LR=1E-4
2. Optimize binary and unary predictions together after 4 epochs

### Output
1. Unaries (Translation, Scale, Rotation)
2. Binaries (Relative Translation, Relative Scale, Relative Direction)



# Train SunCG

## Box3D base model
```
nice -n 20 python -m relative3d.experiments.suncg.box3d --plot_scalars --save_epoch_freq=4 --batch_size=24 --name=box3d_base_spatial_mask_common_upsample --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=8 --num_epochs=8 --suncg_dir /nvme-scratch/nileshk/suncg/ --pred_relative=True --visdom --display_port=8094 --display_id=1 --rel_opt=True --display_freq=100 --display_visuals  --auto_rel_opt=5  --use_spatial_map=True --use_mask_in_common=True  --upsample_mask=True
```
## DWR model
Finetune predictions on detections
```
python -m relative3d.experiments.suncg.dwr --plot_scalars --save_epoch_freq=1 --batch_size=24 --name=dwr_base_spatial_mask_common_upsample --use_context --pred_voxels=False --classify_rot --box3d_ft --box3d_pretrain_name=box3d_base_spatial_mask_common_upsample --shape_loss_wt=10 --n_data_workers=8 --num_epochs=1 --suncg_dir /nvme-scratch/nileshk/suncg/ --pred_relative=True --visdom --display_port=8094 --display_id=1 --rel_opt=True --display_freq=100 --display_visuals  --use_spatial_map=True --use_mask_in_common=True  --upsample_mask=True
```

## DWR FT 
Fineture shape decoder
```
python -m relative3d.experiments.suncg.dwr --name=dwr_base_spatial_mask_common_upsample_ft --classify_rot --shape_dec_ft --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=1000 --display_id=202 --shape_loss_wt=2 --label_loss_wt=10 --batch_size=24 --num_epochs=1 --ft_pretrain_epoch=1 --ft_pretrain_name=dwr_base_spatial_mask_common_upsample --split_size=1.0 --display_port=8094 --suncg_dir=/nvme-scratch/nileshk/suncg/ --n_data_workers=4 --visdom=True --use_spatial_map=True --use_mask_in_common=True --upsample_mask=True --pred_relative=True --rel_opt=True
```


# Train on NYU
## Box3D base model
We are going to fine tune the model trained on SunCG
```
python -m relative3d.experiments.nyu.box3d --plot_scalars --save_epoch_freq=4 --batch_size=24 --name=nyu_box3d_base_spatial_mask_common_upsample --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=8 --num_epochs=16 --nyu_dir /nfs.yoda/imisra/nileshk/nyud2/ --pred_relative=True --visdom --display_port=8094 --display_id=1 --rel_opt=True --display_freq=100 --display_visuals   --use_spatial_map=True --use_mask_in_common=True  --upsample_mask=True --ft_pretrain_name=box3d_base_spatial_mask_common_upsample --ft_pretrain_epoch=8
```

## DWR base model
We are going to fine tune the model trained on SunCG for detection. We do not fine tune the shape decoder on NYUv2 as the dataset has very few CAD models.

```
python -m relative3d.experiments.nyu.dwr --plot_scalars --save_epoch_freq=1 --batch_size=8 --name=nyu_dwr_base_spatial_mask_common_upsample --use_context --pred_voxels=False --classify_rot --ft_pretrain_name=dwr_base_spatial_mask_common_upsample --ft_pretrain_epoch=1 --shape_loss_wt=10 --n_data_workers=0 --num_epochs=16 --nyu_dir /nfs.yoda/imisra/nileshk/nyud2/ --pred_relative=True --visdom --display_port=8094 --display_id=1 --rel_opt=True --display_freq=100 --display_visuals  --use_spatial_map=True --use_mask_in_common=True  --upsample_mask=True
```