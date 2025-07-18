import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import torch
import torch.distributed as dist

from torch import cuda as accel  # noqa

DEVICE_TYPE = "cuda"
BACKEND = "nccl"

import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available()
from torch import cuda as accel  # noqa

DEVICE_TYPE = "cuda"
BACKEND = "nccl"

dist.init_process_group(backend=BACKEND)

# Matrix sizes, iterations, and warmups. Dimensions chosen to make the compute and comms times
# similar.
COMPUTE_DIM = 2**14
COMMS_DIM = 4 * COMPUTE_DIM
ITERS = 20
WARMUPS = 3


RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"{DEVICE_TYPE}:{LOCAL_RANK}")
DTYPE = torch.bfloat16
accel.set_device(DEVICE)

compute_stream = accel.Stream(device=DEVICE)
comms_stream = accel.Stream(device=DEVICE)

compute_matrix = torch.randn(COMPUTE_DIM, COMPUTE_DIM, device=DEVICE, dtype=DTYPE)
comms_matrix = torch.randn(COMMS_DIM, COMMS_DIM, device=DEVICE, dtype=DTYPE)

def compute(stream: Optional[accel.Stream] = None) -> torch.Tensor:
    with accel.stream(stream):
        for _ in range(ITERS):
            out = compute_matrix @ compute_matrix
    return out


def comms(stream: Optional[accel.Stream] = None) -> None:
    with accel.stream(stream):
        for _ in range(ITERS):
            dist.all_reduce(comms_matrix)

# Warmup
for _ in range(WARMUPS):
    compute()
    comms()

torch.cuda.synchronize()
start = perf_counter()

# 真正同时启动计算和通信
# 使用异步方式，避免顺序执行

# 启动计算任务
def run_compute():
    with accel.stream(compute_stream):
        for _ in range(ITERS):
            out = compute_matrix @ compute_matrix
    return out

# 启动通信任务  
def run_comms():
    with accel.stream(comms_stream):
        for _ in range(ITERS):
            dist.all_reduce(comms_matrix)

# 同时启动两个任务
import threading
compute_thread = threading.Thread(target=run_compute)
comms_thread = threading.Thread(target=run_comms)

compute_thread.start()
comms_thread.start()

# 等待两个任务完成
compute_thread.join()
comms_thread.join()

# 确保GPU操作完成
torch.cuda.synchronize()


end = perf_counter()
print(f"Time: {end - start}")
# print(f"Time: {t.s}")

dist.destroy_process_group()
