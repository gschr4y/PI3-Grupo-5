"""Mede o tempo por lote da CNN3D em CPU e GPU e testa o modo determinístico na GPU."""

import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import config
from experimentos.cnn import CNN3D


def medir(device, aquecimento, voltas, threads=4):
    torch.set_num_threads(threads)
    config.fixar_semente()
    modelo = CNN3D().to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), 1e-4)
    x = torch.rand(32, 1, 32, 32, 32, device=device)
    y = torch.randint(0, 2, (32,), device=device)

    def passo():
        otimizador.zero_grad()
        torch.nn.functional.cross_entropy(modelo(x), y).backward()
        otimizador.step()

    for _ in range(aquecimento):
        passo()
    sincronizar = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    sincronizar()
    inicio = time.time()
    for _ in range(voltas):
        passo()
    sincronizar()
    treino = (time.time() - inicio) / voltas
    modelo.eval()
    inicio = time.time()
    with torch.inference_mode():
        for _ in range(voltas):
            modelo(x)
    sincronizar()
    return treino, (time.time() - inicio) / voltas


def pesos_apos_treino_curto():
    config.fixar_semente()
    modelo = CNN3D().cuda()
    otimizador = torch.optim.Adam(modelo.parameters(), 1e-4)
    gerador = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(50):
        x = torch.rand(32, 1, 32, 32, 32, device="cuda", generator=gerador)
        y = torch.randint(0, 2, (32,), device="cuda", generator=gerador)
        otimizador.zero_grad()
        torch.nn.functional.cross_entropy(modelo(x), y).backward()
        otimizador.step()
    return torch.cat([p.detach().flatten() for p in modelo.parameters()])


treino, inferencia = medir("cpu", 3, 20)
print(f"cpu: treino {treino * 1000:.1f} ms/lote, inferencia {inferencia * 1000:.1f} ms/lote")
if torch.cuda.is_available():
    treino, inferencia = medir("cuda", 5, 50)
    print(f"gpu: treino {treino * 1000:.1f} ms/lote, inferencia {inferencia * 1000:.1f} ms/lote")
    torch.use_deterministic_algorithms(True)
    try:
        medir("cuda", 1, 1)
        print("modo estrito na gpu: passou")
    except RuntimeError as erro:
        print("modo estrito na gpu:", str(erro).split(",")[0])
    torch.use_deterministic_algorithms(True, warn_only=True)
    a, b = pesos_apos_treino_curto(), pesos_apos_treino_curto()
    print(f"dois treinos de 50 lotes: diferenca maxima {(a - b).abs().max().item()}")
