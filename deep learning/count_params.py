# Keras
model.count_params()

# PyTorch
pytorch_total_params = sum(p.numel() for p in model.parameters())

## PyTorch only trainable parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
