import torch
import torch.nn as nn

# 指定你的 pt 文件路径
pt_file_path = "/raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_scibert_mRNAdesigner/checkpoint-44000/pytorch_model.bin"
pt_file_path_2 = "ckpt_2000.pt"
# 加载模型权重
checkpoint = torch.load(pt_file_path_2, map_location=torch.device('cpu'))

print("Checkpoint Keys and Shapes:")
for key, value in checkpoint.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")

print(checkpoint['model'].keys())
print(checkpoint['model_args'])
print(checkpoint['config'])

