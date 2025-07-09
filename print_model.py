import torch

# 加载模型权重
checkpoint_2000 = torch.load('/home/yzhang/research/pro2rna/pro2rna/mRNAGPT/ckpt_2000.pt', map_location='cpu')
checkpoint_563000 = torch.load('/home/yzhang/research/pro2rna/pro2rna/mRNAGPT/ckpt_563000.pt', map_location='cpu')
checkpoint_generrna = torch.load('/home/yzhang/research/pro2rna/pro2rna/GenerRNA/model_updated.pt', map_location='cpu')
# 获取模型的state_dict

print(checkpoint_563000['model_args'])
model_2000 = checkpoint_2000['model']
model_563000 = checkpoint_563000['model']
model_generrna = checkpoint_generrna['model']
# 打印模型的命名参数
print("Named parameters for ckpt_2000.pt:")
for name, param in model_2000.items():
    print(f"{name}: {param.size()}")

print("\nNamed parameters for ckpt_563000.pt:")
for name, param in model_563000.items():
    print(f"{name}: {param.size()}")


# print("\nNamed parameters for ckpt_generrna.pt:")
# for name, param in model_generrna.items():
#     print(f"{name}: {param.size()}")