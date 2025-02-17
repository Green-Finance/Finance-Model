#!/usr/bin/env python3
try:
    import torch
except ImportError:
    raise ImportError('Install torch via `pip install torch`')

from packaging.version import Version as V

# 현재 torch 버전 및 CUDA 버전 가져오기
v = V(torch.__version__)
cuda = str(torch.version.cuda)
is_ampere = torch.cuda.get_device_capability()[0] >= 8

# 지원되는 CUDA 버전 확인 (지원: 11.8, 12.1, 12.4)
if cuda not in ["11.8", "12.1", "12.4"]:
    raise RuntimeError(f"CUDA = {cuda} not supported!")

# torch 버전 별 unsloth 설치 옵션 문자열 결정
if   v <= V('2.1.0'):
    raise RuntimeError(f"Torch = {v} too old!")
elif v <= V('2.1.1'):
    x = 'cu{}{}-torch211'
elif v <= V('2.1.2'):
    x = 'cu{}{}-torch212'
elif v < V('2.3.0'):
    x = 'cu{}{}-torch220'
elif v < V('2.4.0'):
    x = 'cu{}{}-torch230'
elif v < V('2.5.0'):
    x = 'cu{}{}-torch240'
elif v < V('2.6.0'):
    x = 'cu{}{}-torch250'
else:
    raise RuntimeError(f"Torch = {v} too new!")

# GPU가 Ampere 이상이면 "-ampere" 옵션 추가
x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")

# 최종 pip 명령어 출력
print(f'pip install --upgrade pip && pip install "unsloth[{x}] @ git+https://github.com/unslothai/unsloth.git"')