import torch
import time

is_staged = True
class DetachFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_detached = x.detach()
        x_detached.requires_grad = True
        return x_detached

    @staticmethod
    def backward(ctx, grad_output):
        if is_staged:
            return None
        else:
            return grad_output

if __name__ == "__main__":
    layer1 = torch.nn.Linear(8192, 8192)
    layer2 = torch.nn.Linear(8192, 8192)
    layer1.to("cuda")
    layer2.to("cuda")
    torch.cuda.manual_seed(0)

    backward_times = []
    for i in range(10):
        x = torch.randn(8192).to("cuda")
        x.requires_grad = True

        out1 = layer1(x)
        out1_detached = DetachFunction.apply(out1)
        out1_detached.retain_grad()
        out2 = layer2(out1_detached)

        out2_grad = torch.randn_like(out2)
        start = time.perf_counter()
        torch.autograd.backward(out2, out2_grad, retain_graph=True)

        torch.autograd.backward(out1, out1_detached.grad)
        end = time.perf_counter()
        backward_times.append(end - start)

    print(f"backward_times: {backward_times}")
    print(f"Backward time: {backward_times[-1]} seconds")
