## Evaluation Scipts
Evaluate results on SunCG with GT bboxes

```
python -m relative3d.benchmark.suncg.box3d --num_train_epoch=8 --name=box3d_base_spatial_mask_common_upsample --classify_rot --pred_voxels=False --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=True --suncg_dir=/nvme-scratch/nileshk/suncg/ --preload_stats=False --results_name=box3d_base_spatial_mask_common_upsample --do_updates=True --save_predictions_to_disk=True
```

Evaluate results on SunCG in detection setting to report mAP scores

```
python -m relative3d.benchmark.suncg.dwr --num_train_epoch=1 --name=dwr_base_spatial_mask_common_upsample --classify_rot --pred_voxels=True --use_context --save_visuals --visuals_freq=50 --eval_set=val --pred_relative=True --suncg_dir=/nvme-scratch/nileshk/suncg/ --preload_stats=False  --results_name=dwr_base_spatial_mask_common_upsample --do_updates=True  --save_predictions_to_disk=True --use_spatial_map=True --use_mask_in_common=True --upsample_mask=True
```


To evaluate the baselines please look at [Baselines](baselines.md)