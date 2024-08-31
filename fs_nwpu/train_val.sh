# base stage 的训练
python fs_nwpu/train_fasterrcnn.py

python fs_nwpu/prototype_fasterrcnn.py --config-file configs/NWPU/prototype/split1.yml

# 将收集的 feature 转为 prototype
python fs_nwpu/scripts/convert_feature_as_prototype.py

python fs_nwpu/prototype_fasterrcnn.py --config-file configs/NWPU/prototype/select_shot1.yml


## 生成 train_part novel 训练集
python fs_nwpu/scripts/select_shot.py
python fs_nwpu/scripts/generate_novel_dataset.py

python fs_nwpu/scripts/ckpt_redetect.py

## 进行 incremental 训练
python fs_nwpu/inc_train_fasterrcnn.py

