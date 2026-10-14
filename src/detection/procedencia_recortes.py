"""Identifica o volume e o código que produziram um lote de recortes."""

import hashlib
from pathlib import Path


def sha256(caminho):
    h = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        while bloco := arquivo.read(1024 * 1024):
            h.update(bloco)
    return h.hexdigest()


def assinatura(volume, tamanho):
    raiz = Path(__file__).resolve().parents[2]
    return {
        "volume_sha256": sha256(volume),
        "tamanho": list(tamanho),
        "codigo": {nome: sha256(raiz / nome) for nome in (
            "scripts/09_extrair_patches.py", "src/detection/patches.py",
            "src/detection/procedencia_recortes.py",
        )},
    }
