"""Ícones simples (desenho de linhas) dos objetos que o visor identifica.

Cada ícone é uma lista de formas em coordenadas de 0 a 1 dentro de um quadrado, desenhadas
com OpenCV. Não precisa baixar nada. icon_for("caneta") ou icon_for("CELULAR") devolve a
chave do ícone; nomes desconhecidos usam um cubo genérico.

Para adicionar um ícone: crie uma entrada em ICONS e os nomes que o chamam em ALIASES.
"""

import math
import re
import unicodedata

import cv2
import numpy as np


def _rect(x1, y1, x2, y2):
    return ("poly", [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], True)


def _arc(cx, cy, rx, ry, start, end, steps=24):
    """Arco de elipse como polilinha (ângulos em graus, 0 = direita, 90 = baixo)."""
    pts = [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
           for a in np.linspace(start, end, steps)]
    return ("poly", pts, False)


def _circle(cx, cy, r):
    return ("circle", cx, cy, r)


def _line(x1, y1, x2, y2):
    return ("poly", [(x1, y1), (x2, y2)], False)


# Formas de cada ícone. "rotate" (graus) gira o desenho todo em torno do centro.
ICONS = {
    "caneta": {"rotate": 40, "shapes": [
        _rect(0.44, 0.06, 0.56, 0.32),                         # tampa
        ("poly", [(0.56, 0.10), (0.61, 0.10), (0.61, 0.30)], False),   # clipe
        _rect(0.44, 0.32, 0.56, 0.74),                         # corpo
        _line(0.44, 0.40, 0.56, 0.40),
        ("poly", [(0.44, 0.74), (0.50, 0.92), (0.56, 0.74)], False),   # ponta
        _line(0.50, 0.86, 0.50, 0.94),
    ]},
    "celular": {"shapes": [
        _rect(0.31, 0.08, 0.69, 0.92), _rect(0.35, 0.19, 0.65, 0.78),
        _line(0.45, 0.135, 0.55, 0.135), _circle(0.50, 0.85, 0.03),
    ]},
    "copo": {"shapes": [
        _arc(0.47, 0.24, 0.22, 0.05, 0, 360),
        ("poly", [(0.25, 0.24), (0.30, 0.86), (0.64, 0.86), (0.69, 0.24)], False),
        _arc(0.47, 0.86, 0.17, 0.035, 0, 180),
        _arc(0.70, 0.52, 0.13, 0.16, -80, 80),                  # alça
    ]},
    "garrafa": {"shapes": [
        _rect(0.43, 0.05, 0.57, 0.12),
        ("poly", [(0.44, 0.12), (0.44, 0.25), (0.33, 0.38), (0.33, 0.92), (0.67, 0.92),
                  (0.67, 0.38), (0.56, 0.25), (0.56, 0.12)], False),
        _rect(0.33, 0.50, 0.67, 0.72),
    ]},
    "livro": {"shapes": [
        ("poly", [(0.50, 0.26), (0.32, 0.18), (0.10, 0.22), (0.10, 0.82), (0.32, 0.78),
                  (0.50, 0.86)], False),
        ("poly", [(0.50, 0.26), (0.68, 0.18), (0.90, 0.22), (0.90, 0.82), (0.68, 0.78),
                  (0.50, 0.86)], False),
        _line(0.50, 0.26, 0.50, 0.86),
        _line(0.17, 0.36, 0.42, 0.38), _line(0.17, 0.48, 0.42, 0.50), _line(0.17, 0.60, 0.42, 0.62),
        _line(0.58, 0.38, 0.83, 0.36), _line(0.58, 0.50, 0.83, 0.48), _line(0.58, 0.62, 0.83, 0.60),
    ]},
    "notebook": {"shapes": [
        _rect(0.20, 0.18, 0.80, 0.62), _rect(0.25, 0.23, 0.75, 0.57),
        ("poly", [(0.20, 0.62), (0.80, 0.62), (0.92, 0.78), (0.08, 0.78)], True),
        _line(0.42, 0.72, 0.58, 0.72),
    ]},
    "teclado": {"shapes": [
        _rect(0.06, 0.32, 0.94, 0.70),
        *[_rect(0.11 + i * 0.08, 0.38, 0.17 + i * 0.08, 0.45) for i in range(10)],
        *[_rect(0.14 + i * 0.08, 0.49, 0.20 + i * 0.08, 0.56) for i in range(9)],
        _rect(0.28, 0.60, 0.72, 0.65),
    ]},
    "mouse": {"shapes": [
        _arc(0.50, 0.52, 0.21, 0.34, 0, 360), _line(0.50, 0.18, 0.50, 0.44),
        _arc(0.50, 0.44, 0.21, 0.06, 0, 180), _rect(0.48, 0.27, 0.52, 0.36),
        ("poly", [(0.50, 0.18), (0.50, 0.10), (0.58, 0.04)], False),   # fio
    ]},
    "tesoura": {"shapes": [
        _circle(0.32, 0.77, 0.11), _circle(0.68, 0.77, 0.11),
        _line(0.39, 0.68, 0.64, 0.08), _line(0.61, 0.68, 0.36, 0.08), _circle(0.50, 0.40, 0.025),
    ]},
    "relogio": {"shapes": [
        _circle(0.50, 0.50, 0.38), _circle(0.50, 0.50, 0.33),
        *[_line(0.50 + 0.28 * math.cos(a), 0.50 + 0.28 * math.sin(a),
                0.50 + 0.32 * math.cos(a), 0.50 + 0.32 * math.sin(a))
          for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)],
        _line(0.50, 0.50, 0.50, 0.28), _line(0.50, 0.50, 0.66, 0.58), _circle(0.50, 0.50, 0.02),
    ]},
    "pessoa": {"shapes": [
        _circle(0.50, 0.30, 0.14), _arc(0.50, 0.92, 0.32, 0.40, 180, 360), _line(0.18, 0.92, 0.82, 0.92),
    ]},
    "cadeira": {"shapes": [
        _rect(0.30, 0.10, 0.70, 0.46), _line(0.38, 0.20, 0.62, 0.20),
        ("poly", [(0.26, 0.58), (0.74, 0.58), (0.70, 0.50), (0.30, 0.50)], True),
        _line(0.30, 0.46, 0.30, 0.50), _line(0.70, 0.46, 0.70, 0.50),
        _line(0.30, 0.58, 0.28, 0.92), _line(0.70, 0.58, 0.72, 0.92),
        _line(0.40, 0.58, 0.40, 0.84), _line(0.60, 0.58, 0.60, 0.84),
    ]},
    "tela": {"shapes": [
        _rect(0.10, 0.16, 0.90, 0.68), _rect(0.15, 0.21, 0.85, 0.63),
        ("poly", [(0.44, 0.68), (0.40, 0.82), (0.60, 0.82), (0.56, 0.68)], False),
        _line(0.30, 0.84, 0.70, 0.84),
    ]},
    "controle": {"shapes": [
        _rect(0.38, 0.06, 0.62, 0.94), _circle(0.50, 0.18, 0.04),
        _circle(0.50, 0.36, 0.07), _line(0.50, 0.29, 0.50, 0.43), _line(0.43, 0.36, 0.57, 0.36),
        *[_circle(0.45 + (i % 2) * 0.10, 0.55 + (i // 2) * 0.10, 0.025) for i in range(6)],
    ]},
    "chave": {"rotate": -35, "shapes": [
        _circle(0.26, 0.50, 0.15), _circle(0.26, 0.50, 0.05),
        _line(0.41, 0.50, 0.90, 0.50),
        ("poly", [(0.72, 0.50), (0.72, 0.62), (0.78, 0.62), (0.78, 0.50)], False),
        ("poly", [(0.84, 0.50), (0.84, 0.60), (0.90, 0.60), (0.90, 0.50)], False),
    ]},
    "oculos": {"shapes": [
        _arc(0.29, 0.52, 0.17, 0.13, 0, 360), _arc(0.71, 0.52, 0.17, 0.13, 0, 360),
        _arc(0.50, 0.50, 0.05, 0.04, 180, 360),
        _line(0.12, 0.48, 0.04, 0.38), _line(0.88, 0.48, 0.96, 0.38),
    ]},
    "mochila": {"shapes": [
        _arc(0.50, 0.22, 0.10, 0.08, 180, 360),
        ("poly", [(0.24, 0.92), (0.24, 0.36), (0.32, 0.24), (0.68, 0.24), (0.76, 0.36),
                  (0.76, 0.92)], True),
        _rect(0.34, 0.58, 0.66, 0.82), _line(0.34, 0.66, 0.66, 0.66),
    ]},
    "gato": {"shapes": [
        ("poly", [(0.22, 0.42), (0.24, 0.12), (0.40, 0.28), (0.60, 0.28), (0.76, 0.12),
                  (0.78, 0.42)], False),
        _arc(0.50, 0.52, 0.28, 0.32, -20, 200),
        _circle(0.39, 0.50, 0.04), _circle(0.61, 0.50, 0.04),
        ("poly", [(0.46, 0.62), (0.54, 0.62), (0.50, 0.67)], True),
        _line(0.44, 0.68, 0.14, 0.64), _line(0.44, 0.71, 0.14, 0.76),
        _line(0.56, 0.68, 0.86, 0.64), _line(0.56, 0.71, 0.86, 0.76),
    ]},
    "cao": {"shapes": [
        _arc(0.50, 0.46, 0.24, 0.30, 180, 360),
        ("poly", [(0.26, 0.46), (0.30, 0.74), (0.40, 0.86), (0.60, 0.86), (0.70, 0.74),
                  (0.74, 0.46)], False),
        ("poly", [(0.30, 0.26), (0.12, 0.34), (0.12, 0.62), (0.24, 0.60), (0.27, 0.42)], False),
        ("poly", [(0.70, 0.26), (0.88, 0.34), (0.88, 0.62), (0.76, 0.60), (0.73, 0.42)], False),
        _circle(0.40, 0.46, 0.035), _circle(0.60, 0.46, 0.035),
        _arc(0.50, 0.66, 0.08, 0.05, 0, 360), _line(0.50, 0.71, 0.50, 0.78),
        _arc(0.50, 0.74, 0.08, 0.05, 20, 160),
    ]},
    "generico": {"shapes": [   # cubo em perspectiva com um "?"
        ("poly", [(0.50, 0.12), (0.86, 0.30), (0.50, 0.48), (0.14, 0.30)], True),
        ("poly", [(0.14, 0.30), (0.14, 0.72), (0.50, 0.90), (0.50, 0.48)], False),
        ("poly", [(0.86, 0.30), (0.86, 0.72), (0.50, 0.90)], False),
        _arc(0.68, 0.53, 0.06, 0.06, 180, 405, 12), _line(0.70, 0.59, 0.68, 0.65),
        _circle(0.68, 0.72, 0.01),
    ]},
}

# Nomes (português e inglês, sem acentos) que levam a cada ícone. A ordem importa: o primeiro
# ícone com um nome que aparece no texto vence.
ALIASES = {
    "caneta": ["caneta", "canetas", "lapis", "lapiseira", "marcador", "pincel", "pen", "pencil",
               "marker", "ballpoint"],
    "celular": ["celular", "telefone", "smartphone", "iphone", "cell phone", "phone"],
    "notebook": ["notebook", "laptop", "computador portatil"],
    "teclado": ["teclado", "keyboard"],
    "mouse": ["mouse"],
    "controle": ["controle", "remote", "controller"],
    "tela": ["tv", "televisao", "televisor", "monitor", "tela", "screen"],
    "copo": ["copo", "caneca", "xicara", "taca", "cup", "mug", "glass"],
    "garrafa": ["garrafa", "garrafinha", "bottle"],
    "livro": ["livro", "caderno", "agenda", "book"],
    "tesoura": ["tesoura", "scissors"],
    "relogio": ["relogio", "clock", "watch"],
    "oculos": ["oculos", "glasses", "sunglasses"],
    "chave": ["chave", "chaves", "key", "keys"],
    "mochila": ["mochila", "bolsa", "backpack", "bag"],
    "cadeira": ["cadeira", "poltrona", "chair"],
    "gato": ["gato", "gata", "felino", "cat", "kitten"],
    "cao": ["cao", "cachorro", "cadela", "filhote", "dog", "puppy"],
    "pessoa": ["pessoa", "homem", "mulher", "rosto", "person", "man", "woman", "face"],
}


def _plain(text):
    normalized = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return normalized.lower()


def icon_for(name):
    """Chave do ícone para um nome de objeto (qualquer idioma); "generico" se não conhecer."""
    text = _plain(name)
    for key, words in ALIASES.items():
        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", text):
                return key
    return "generico"


def _transform(points, size, rotate):
    pts = np.array(points, dtype=np.float32) - 0.5
    if rotate:
        a = math.radians(rotate)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
        pts = pts @ rot.T
    return ((pts + 0.5) * size).astype(np.int32)


_tile_cache = {}


def icon_tile(key, size, color=(240, 240, 240), thickness=2):
    """Imagem quadrada (fundo preto) com o ícone desenhado. Fica em cache."""
    cache_key = (key, size, color, thickness)
    if cache_key not in _tile_cache:
        icon = ICONS.get(key, ICONS["generico"])
        rotate = icon.get("rotate", 0)
        tile = np.zeros((size, size, 3), dtype=np.uint8)
        for shape in icon["shapes"]:
            if shape[0] == "circle":
                _, cx, cy, r = shape
                center = tuple(_transform([(cx, cy)], size, rotate)[0])
                cv2.circle(tile, center, max(1, int(r * size)), color, thickness, cv2.LINE_AA)
            else:
                _, points, closed = shape
                cv2.polylines(tile, [_transform(points, size, rotate)], closed, color, thickness,
                              cv2.LINE_AA)
        _tile_cache[cache_key] = tile
    return _tile_cache[cache_key]
