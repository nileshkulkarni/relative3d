# Train Baselines

## Factored3d Baseline SunCG

### Box3d Base
Train on gt bboxes
```
nice -n 20 python -m relative3d.experiments.suncg.box3d --plot_scalars --save_epoch_freq=4 --batch_size=24 --name=box3d_base_factored3d_baseline --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=8 --num_epochs=8 --suncg_dir /nvme-scratch/nileshk/suncg/ --pred_relative=False --visdom --display_port=8094 --display_id=1  --display_freq=100 --display_visuals
```

### DWR  model
Fine tune on proposals

```
python -m relative3d.experiments.suncg.dwr --plot_scalars --save_epoch_freq=1 --batch_size=24 --name=dwr_base_factored3d_baseline --use_context --pred_voxels=False --classify_rot --box3d_ft --box3d_pretrain_name=box3d_base_factored3d_baseline --shape_loss_wt=10 --n_data_workers=8 --num_epochs=1 --suncg_dir /nvme-scratch/nileshk/suncg/ --pred_relative=False --visdom --display_port=8094 --display_id=1  --display_freq=100 --display_visuals  
```



### DWR FT
Fine tune shape decoder

```
python -m factored3d.experiments.suncg.dwr --name=dwr_base_factored3d_baseline_ft --classify_rot --shape_dec_ft --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=1000 --display_id=202 --shape_loss_wt=2 --label_loss_wt=10 --batch_size=24 --num_epochs=1 --ft_pretrain_epoch=1 --ft_pretrain_name=dwr_base_factored3d_baseline --split_size=1.0 --display_port=8100 --suncg_dir=/nvme-scratch/nileshk/suncg/ --n_data_workers=4 --visdom=True --use_spatial_map=True --use_mask_in_common=True --upsample_mask=True
```

## Factored3D Baseline on NYU.
We are going to fine tune models trained on the SunCG dataset using the NYUv2 dataset.
### Box3d base model
```
python -m relative3d.experiments.nyu.box3d --plot_scalars --save_epoch_freq=4 --batch_size=24 --name=box3d_base_factored3d_baseline_nyu --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=8 --num_epochs=16 --nyu_dir /nvme-scratch/nileshk/nyuv2/ --pred_relative=False --visdom --display_port=8094 --display_id=1  --display_freq=100 --display_visuals --ft_pretrain_name=box3d_base_factored3d_baseline --ft_pretrain_epoch=8
```

### DWR model. 
NYUv2 in detection setting. We going to fine tune the SunCG model in detection setting with NYUv2 dataset.
```
python -m relative3d.experiments.nyu.dwr --plot_scalars --save_epoch_freq=1 --batch_size=24 --name=dwr_base_factored3d_baseline_nyu --use_context --pred_voxels=False --classify_rot --box3d_ft --box3d_pretrain_name=box3d_base_factored3d_baseline --shape_loss_wt=10 --n_data_workers=8 --num_epochs=16 --nyu_dir /nvme-scratch/nileshk/nyu/ --pred_relative=False --visdom --display_port=8094 --display_id=1  --display_freq=100  --display_visuals  --ft_pretrain_name=dwr_base_factored3d_baseline --ft_pretrain_epoch=1
```

### Evaluate on SunCG
Evaluate of GT bboxes to get errors in translation, rotation, and scale.
```
python -m relative3d.benchmark.suncg.box3d --num_train_epoch=8 --name=box3d_base_factored3d_baseline --classify_rot --pred_voxels=False --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=False --suncg_dir=/nvme-scratch/nileshk/suncg/ --preload_stats=False  --results_name=box3d_base_factored3d_baseline --do_updates=False --save_predictions_to_disk=True
```

Evaluate with detections from a pretrained model, to compute mAP scores.
```
python -m relative3d.benchmark.suncg.dwr --num_train_epoch=1 --name=dwr_base_factored3d_baseline_ft  --classify_rot --pred_voxels=True --use_context   --eval_set=val --suncg_dir=/nvme-scratch/nileshk/suncg/ --use_spatial_map=True
```
### Evaluate on NYU 
Evaluate of GT bboxes to get errors in translation, rotation, and scale.
```
python -m relative3d.benchmark.nyu.box3d --num_train_epoch=16 --name=box3d_base_factored3d_baseline_nyu --classify_rot --pred_voxels=False --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=False --nyu_dir=/nvme-scratch/nileshk/nyu/ --preload_stats=False  --results_name=box3d_base_factored3d_baseline_nyu --do_updates=False --save_predictions_to_disk=True
```

Evaluate with detections from a pretrained model, to compute mAP scores.

```
python -m relative3d.benchmark.nyu.dwr_save_predictions --num_train_epoch=16 --name=dwr_base_factored3d_baseline_ft_nyu  --classify_rot --pred_voxels=True --use_context   --eval_set=val --nyu_dir=/nvme-scratch/nileshk/nyu/ --use_spatial_map=True
```
## GCN Net Baseline
### Training GCN on SunCG
Box3d Base: Train on gt bboxes
```
python -m relative3d.experiments.suncg.gcn --plot_scalars --save_epoch_freq=4 --batch_size=24 --name=box3d_base_gcn_baseline --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=8 --num_epochs=8 --suncg_dir /nvme-scratch/nileshk/suncg/  --visdom --display_port=8094 --display_id=1  --display_freq=100 --display_visuals
```

DWR  model :Fine tune on proposals
```
python -m relative3d.experiments.suncg.dwr_gcn --plot_scalars --save_epoch_freq=1 --batch_size=24 --name=dwr_base_gcn_baseline --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=8 --num_epochs=1 --suncg_dir /nvme-scratch/nileshk/suncg/ --visdom --display_port=8094 --display_id=1 --display_freq=100 --display_visuals
```

DWR FT : Fine tune shape decoder
    
```
python -m relative3d.experiments.suncg.dwr_gcn --name=dwr_base_gcn_baseline_ft --classify_rot --shape_dec_ft --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=100 --display_id=1 --shape_loss_wt=2 --label_loss_wt=10 --batch_size=24 --num_epochs=1 --ft_pretrain_epoch=1 --ft_pretrain_name=dwr_base_gcn_baseline --split_size=1.0 --display_port=8094 --suncg_dir=/nvme-scratch/nileshk/suncg/ --n_data_workers=4 --visdom=True --use_spatial_map=True --use_mask_in_common=True --upsample_mask=True
```

### Training on NYU
Follow instructions to train it on SunCG dataset and then fine-tune the model using `relative3d.experiments.nyu.gcn` and `relative3d.experiments.suncg.dwr_gcn`.


### Evaluate on SunCG
Evaluate of GT bboxes to get errors in translation, rotation, and scale
```
python -m relative3d.benchmark.suncg.gcnNet --num_train_epoch=8 --name=box3d_base_gcn_baseline --classify_rot --pred_voxels=False --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=False --suncg_dir=/nvme-scratch/nileshk/suncg/ --preload_stats=False  --results_name=box3d_base_gcn_baseline --save_predictions_to_disk=True
```

Evaluate with detections from a pretrained model, to compute mAP scores.
```
python -m relative3d.benchmark.suncg.gcn_dwr --num_train_epoch=1 --name=dwr_base_gcn_baseline  --classify_rot --pred_voxels=True --use_context   --eval_set=val --suncg_dir=/nvme-scratch/nileshk/suncg/   --use_spatial_map=True --max_eval_iter=20
```

## Internet Baseline

### Training 

Box3d Base: Train on gt bboxes
```
nice -n 20 python -m relative3d.experiments.suncg.internet --plot_scalars --save_epoch_freq=4 --batch_size=24 --name=box3d_base_internet_baseline --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=8 --num_epochs=8 --suncg_dir /nvme-scratch/nileshk/suncg/ --pred_relative=False --visdom --display_port=8094 --display_id=1  --display_freq=100 --display_visuals
```

DWR model: Fine tune on proposals
```
python -m relative3d.experiments.suncg.dwr_internet --plot_scalars --save_epoch_freq=1 --batch_size=24 --name=dwr_base_internet_baseline --use_context --pred_voxels=False --classify_rot --box3d_ft --box3d_pretrain_name=box3d_base_internet_baseline --shape_loss_wt=10 --n_data_workers=8 --num_epochs=1 --suncg_dir /nvme-scratch/nileshk/suncg/ --pred_relative=False --visdom --display_port=8094 --display_id=1  --display_freq=100 --display_visuals  
```

DWR FT: Fine tune shape decoder
```
python -m factored3d.experiments.suncg.dwr_internet --name=dwr_base_internet_baseline_ft --classify_rot --shape_dec_ft --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=1000 --display_id=202 --shape_loss_wt=2 --label_loss_wt=10 --batch_size=24 --num_epochs=1 --ft_pretrain_epoch=1 --ft_pretrain_name=dwr_base_internet_baseline --split_size=1.0 --display_port=8100 --suncg_dir=/nvme-scratch/nileshk/suncg/ --n_data_workers=4 --visdom=True --use_spatial_map=True --use_mask_in_common=True --upsample_mask=True
```




### Evaluate on SunCG
Evaluate of GT bboxes to get errors in translation, rotation, and scale
```
python -m relative3d.benchmark.suncg.internet --num_train_epoch=8 --name=box3d_base_internet_baseline --classify_rot --pred_voxels=False --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=False --suncg_dir=/nvme-scratch/nileshk/suncg/ --preload_stats=False  --results_name=box3d_base_internet_baseline --save_predictions_to_disk=True
```

Evaluate with detections from a pretrained model, to compute mAP scores.
```
python -m relative3d.benchmark.suncg.internet_dwr --num_train_epoch=1 --name=dwr_base_internet_baseline  --classify_rot --pred_voxels=True --use_context   --eval_set=val --suncg_dir=/nvme-scratch/nileshk/suncg/   --use_spatial_map=True --save_predictions_to_disk=True
```


## CRF Baseline
Evaluate on SUNCG with GT bboxes. We used the Factored3d model trained on SunCG dataset for the unary predictions. The cachedir contains the GMM clusters for all the relative predictions which are used to create the CRF potential functions. The code creating the clusters is [here](https://github.com/nileshkulkarni/relative3d/tree/master/experiments/suncg/crf)

```
python -m relative3d.benchmark.suncg.box3d_crf --num_train_epoch=4 --name=box3d_base_factored3d_baseline --classify_rot --pred_voxels=False --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=False --suncg_dir=/nvme-scratch/nileshk/suncg/ --preload_stats=False --crf_optimize=False --results_name=box3d_base_factored3d_crf_baseline --do_updates=False --save_predictions_to_disk=True
```

