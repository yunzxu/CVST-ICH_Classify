# CVST_Classify
Model:https://drive.google.com/drive/folders/1rmaXS6EZLbFT1hT7QAxgs0lOG1HxNwDZ?usp=drive_link

# usage

要求：python >=3.8

```bash
pip3 install -r requirements.txt
```

下载后修改模型路径：infer_main.py

```python
    seg_model_path = ["train_class_CVST_tseg_p128_dice_augdrop_diceceloss_trainfile_fold_total_n32_pretrain_jit.pth"]
    class_model_path = ['train_class_CVST_maskpatch_foldtotal_segweight_addmask_nofreeze_035PICH_bestTruenum_bestauc_jit.pth',
                        'train_class_CVST_maskpatch_UNetCBAM_foldtotal_segweight_nofreeze_035PICH_bestauc_jit.pth']

```

运行

```bash
python3 infer_main.py --file nii_file_path
```

输出结果：

```bash
UNet      = 0.9412037 0.9155715584754944 pred sICH
UNet+CBAM = 0.9942616 0.9964494824409484 pred sICH
```

2个模型的预测，分别为单模型预测值，test augmentation预测值,预测类别。

# 2025.4.25更新：修正代码错误：
infer_main.py 第147行缺失了将图像填充到256x256x256的代码，导致ROI区域与分割不一致。重新修正代码。