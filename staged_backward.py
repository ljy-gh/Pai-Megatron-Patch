import torch

layer1 = torch.nn.Linear(10, 10)
layer2 = torch.nn.Linear(10, 10)

x = torch.randn(10)
x.requires_grad = True

out1 = layer1(x)
out1_detached = out1.detach().clone()
out1_detached.requires_grad = True
out2 = layer2(out1_detached)
print("After forward pass:")
print(f"layer2.weight.grad: {layer2.weight.grad}")
print(f"out1_detached.grad: {out1_detached.grad}")
print(f"layer1.weight.grad: {layer1.weight.grad}")
print(f"x.grad: {x.grad}")

out2_grad = torch.randn_like(out2)
torch.autograd.backward(out2, out2_grad)
print()
print("After layer2 backward pass:")
print(f"layer2.weight.grad: {layer2.weight.grad}")
print(f"out1_detached.grad: {out1_detached.grad}")
print(f"layer1.weight.grad: {layer1.weight.grad}")
print(f"x.grad: {x.grad}")

torch.autograd.backward(out1, out1_detached.grad)
print()
print("After layer1 backward pass:")
print(f"layer2.weight.grad: {layer2.weight.grad}")
print(f"out1_detached.grad: {out1_detached.grad}")
print(f"layer1.weight.grad: {layer1.weight.grad}")
print(f"x.grad: {x.grad}")
