"""Desenho do HUD do visor: tingimento da lente, moldura, miras nos alvos, leituras e painel do Wolf.

Tudo é desenhado com OpenCV puro. As fontes Hershey do OpenCV não têm acentos, então os textos
passam por ascii_text() antes de ir para a tela (o terminal continua mostrando o texto completo).
"""

import math
import unicodedata

import cv2
import numpy as np

# Cores em BGR
RED = (50, 40, 235)
RED_DIM = (35, 25, 150)
WHITE = (240, 240, 240)
GREY = (160, 160, 160)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA

# Intensidade dos efeitos da lente (0 desliga)
TINT_STRENGTH = 0.18
VIGNETTE_STRENGTH = 0.55
SCANLINE_DARKEN = 0.12

_vignette_cache = {}


# ---------------------------------------------------------------------------
# Texto e primitivas
# ---------------------------------------------------------------------------

def ascii_text(text):
    """Remove acentos (ã -> a, ç -> c) porque a fonte do OpenCV só desenha ASCII."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return normalized.encode("ascii", "ignore").decode("ascii")


def put(img, text, org, scale=0.5, color=WHITE, thickness=1):
    cv2.putText(img, ascii_text(text), (int(org[0]), int(org[1])), FONT, scale, color, thickness, LINE)


def text_width(text, scale=0.5, thickness=1):
    return cv2.getTextSize(ascii_text(text), FONT, scale, thickness)[0][0]


def blend_rect(img, x1, y1, x2, y2, color, alpha):
    """Retângulo preenchido semitransparente (alpha = opacidade da cor)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def corner_box(img, x1, y1, x2, y2, color, length=None, thickness=2):
    """Só as quatro cantoneiras de um retângulo, como uma mira."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    length = int(length or max(8, min(x2 - x1, y2 - y1) * 0.22))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, LINE)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, LINE)


def wrap_text(text, max_width, scale=0.5):
    """Quebra o texto em linhas que cabem em max_width pixels."""
    lines = []
    for paragraph in ascii_text(text).split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if text_width(candidate, scale) <= max_width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Efeito da lente
# ---------------------------------------------------------------------------

def _vignette(h, w):
    key = (h, w)
    if key not in _vignette_cache:
        ys, xs = np.ogrid[:h, :w]
        dist = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
        mask = 1 - VIGNETTE_STRENGTH * np.clip(dist - 0.55, 0, 1)
        scan = np.ones((h, 1), dtype=np.float32)
        scan[::3] = 1 - SCANLINE_DARKEN
        _vignette_cache[key] = (mask[..., None] * scan[..., None]).astype(np.float32)
    return _vignette_cache[key]


def apply_lens(frame):
    """Tinge a imagem de vermelho, escurece as bordas e adiciona linhas de varredura."""
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    red_tint = gray * np.array([0.35, 0.35, 1.0], dtype=np.float32)
    out = out * (1 - TINT_STRENGTH) + red_tint * TINT_STRENGTH
    out *= _vignette(h, w)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Elementos fixos do visor
# ---------------------------------------------------------------------------

def draw_frame(img, t):
    """Moldura, régua de rumo no topo, retícula central e a faixa de varredura."""
    h, w = img.shape[:2]
    m = 18
    corner_box(img, m, m, w - m, h - m, RED, length=60, thickness=2)

    # Régua no topo: marcações que deslizam devagar
    cx, top = w // 2, 34
    half = int(w * 0.22)
    cv2.line(img, (cx - half, top), (cx + half, top), RED_DIM, 1, LINE)
    offset = (t * 40) % 40
    x = cx - half + offset
    while x < cx + half:
        size = 8 if int((x - offset) / 40) % 3 == 0 else 4
        cv2.line(img, (int(x), top), (int(x), top - size), RED, 1, LINE)
        x += 40
    cv2.drawMarker(img, (cx, top + 7), WHITE, cv2.MARKER_TRIANGLE_UP, 9, 1, LINE)

    # Retícula central com segmentos girando
    c = (w // 2, h // 2)
    for i in range(4):
        start = math.degrees(t * 0.8) + i * 90
        cv2.ellipse(img, c, (34, 34), 0, start, start + 50, RED, 2, LINE)
    cv2.line(img, (c[0] - 12, c[1]), (c[0] - 4, c[1]), WHITE, 1, LINE)
    cv2.line(img, (c[0] + 4, c[1]), (c[0] + 12, c[1]), WHITE, 1, LINE)
    cv2.line(img, (c[0], c[1] - 12), (c[0], c[1] - 4), WHITE, 1, LINE)
    cv2.line(img, (c[0], c[1] + 4), (c[0], c[1] + 12), WHITE, 1, LINE)

    # Faixa de varredura descendo
    y = int((t * 0.35 % 1.0) * h)
    blend_rect(img, m, y - 2, w - m, y + 2, RED, 0.25)


def draw_target(img, target, index, t):
    """Mira em um alvo: cantoneiras, etiqueta com fundo e linha guia."""
    x1, y1, x2, y2 = target["box"]
    color = WHITE if target["kind"] == "FACE" else RED
    pulse = 3 * math.sin(t * 6 + index)
    corner_box(img, x1 - pulse, y1 - pulse, x2 + pulse, y2 + pulse, color, thickness=2)

    tag = f"TGT-{index + 1:02d} {target['label']}"
    sub = target.get("sub", "")
    tw = max(text_width(tag, 0.45), text_width(sub, 0.4)) + 12
    h, w = img.shape[:2]
    tx = int(min(max(x1, 4), w - tw - 4))
    ty = int(y1) - 44
    if ty < 50:  # sem espaço acima: etiqueta embaixo do alvo
        ty = int(min(y2 + 8, h - 40))
    blend_rect(img, tx, ty, tx + tw, ty + 36, BLACK, 0.6)
    cv2.line(img, (tx, ty), (tx, ty + 36), color, 2, LINE)
    put(img, tag, (tx + 6, ty + 15), 0.45, color, 1)
    if sub:
        put(img, sub, (tx + 6, ty + 31), 0.4, GREY, 1)


def draw_readouts(img, info, t):
    """Leituras de status nos cantos e a barra de energia."""
    h, w = img.shape[:2]

    put(img, "VISOR // ONLINE", (36, 62), 0.55, RED, 2)
    put(img, f"FPS {info['fps']:5.1f}", (36, 86), 0.45)
    put(img, f"ALVOS {info['targets']:02d}", (36, 106), 0.45)
    put(img, f"BIO {'DETECTADO' if info['face'] else '--'}", (36, 126), 0.45)

    right = [
        f"MAOS {info['hands']}",
        f"OBJ {info['objects']}",
        f"T+ {info['uptime']}",
    ]
    for i, line in enumerate(right):
        put(img, line, (w - 36 - text_width(line, 0.45), 62 + i * 20), 0.45)

    # Barra de energia vertical à esquerda, com um leve "respiro"
    bx, by, bh = 40, 160, int(h * 0.35)
    level = 0.72 + 0.08 * math.sin(t * 1.5)
    cv2.rectangle(img, (bx, by), (bx + 10, by + bh), RED_DIM, 1, LINE)
    fill = int(bh * level)
    blend_rect(img, bx + 2, by + bh - fill, bx + 9, by + bh - 1, RED, 0.85)
    put(img, "ENERGIA", (bx - 4, by + bh + 18), 0.4, GREY)
    put(img, f"{int(level * 100)}%", (bx - 2, by + bh + 36), 0.45)

    # Gesto reconhecido (vindo do MemeCV)
    gesture = info.get("gesture")
    if gesture:
        label = f"GESTO: {gesture.upper()}"
        put(img, label, (36, h - 40), 0.6, WHITE, 2)
    if info.get("hint"):
        put(img, info["hint"], (36, h - 18), 0.4, GREY)


# ---------------------------------------------------------------------------
# Painel do Wolf
# ---------------------------------------------------------------------------

def draw_wolf_panel(img, wolf_state, t):
    """Painel de comunicação do Wolf no canto inferior direito.

    wolf_state: dict com log [(quem, texto)], online, status, busy, typing, input
    """
    h, w = img.shape[:2]
    pw = int(min(460, w * 0.4))
    ph = int(min(300, h * 0.46))
    x1, y1 = w - pw - 30, h - ph - 30
    x2, y2 = x1 + pw, y1 + ph
    blend_rect(img, x1, y1, x2, y2, BLACK, 0.62)
    corner_box(img, x1, y1, x2, y2, RED, length=18, thickness=2)

    # Cabeçalho
    online = wolf_state["online"]
    blink = int(t * 2) % 2 == 0
    put(img, "WOLF // LINK DE SUPORTE", (x1 + 12, y1 + 22), 0.5, RED, 2)
    status = wolf_state["status"]
    dot_color = (RED if blink else RED_DIM) if wolf_state["busy"] else (WHITE if online else GREY)
    cv2.circle(img, (x2 - 16, y1 + 16), 5, dot_color, -1, LINE)
    put(img, status, (x2 - 26 - text_width(status, 0.4), y1 + 20), 0.4, GREY)
    cv2.line(img, (x1 + 10, y1 + 32), (x2 - 10, y1 + 32), RED_DIM, 1, LINE)

    # Mensagens: as mais recentes ficam embaixo
    scale, line_h = 0.45, 18
    text_w = pw - 24
    input_h = 30
    area_top, area_bottom = y1 + 40, y2 - input_h - 6
    rows = []
    for speaker, text in wolf_state["log"]:
        prefix = "WOLF> " if speaker == "wolf" else "VOCE> "
        color = WHITE if speaker == "wolf" else GREY
        for i, line in enumerate(wrap_text(prefix + text, text_w, scale)):
            rows.append((line, color))
    max_rows = max(1, (area_bottom - area_top) // line_h)
    rows = rows[-max_rows:]
    for i, (line, color) in enumerate(rows):
        put(img, line, (x1 + 12, area_top + 14 + i * line_h), scale, color)

    # Linha de digitação
    cv2.line(img, (x1 + 10, y2 - input_h), (x2 - 10, y2 - input_h), RED_DIM, 1, LINE)
    if wolf_state["typing"]:
        cursor = "_" if blink else " "
        typed = "> " + wolf_state["input"] + cursor
        # mostra só o final se o texto passar da largura do painel
        while text_width(typed, scale) > text_w and len(typed) > 3:
            typed = typed[:2] + typed[3:]
        put(img, typed, (x1 + 12, y2 - 10), scale, WHITE)
    else:
        put(img, "[T] falar  [V] analisar cena  [H] ajuda", (x1 + 12, y2 - 10), 0.4, GREY)


def draw_help(img):
    h, w = img.shape[:2]
    keys = [
        ("T / ENTER", "abrir o canal com o Wolf (digite e ENTER envia)"),
        ("V", "enviar a imagem atual para o Wolf analisar"),
        ("H", "mostrar/esconder esta ajuda"),
        ("ESC", "sair (ou cancelar a digitacao)"),
    ]
    footer = "Voce tambem pode digitar para o Wolf no terminal (aceita acentos)."
    bw, bh = 640, 60 + (len(keys) + 1) * 24
    x1, y1 = (w - bw) // 2, (h - bh) // 2
    blend_rect(img, x1, y1, x1 + bw, y1 + bh, BLACK, 0.75)
    corner_box(img, x1, y1, x1 + bw, y1 + bh, RED, length=20)
    put(img, "CONTROLES", (x1 + 20, y1 + 30), 0.6, RED, 2)
    for i, (key, desc) in enumerate(keys):
        y = y1 + 58 + i * 24
        put(img, key, (x1 + 20, y), 0.5, RED)
        put(img, desc, (x1 + 140, y), 0.5)
    put(img, footer, (x1 + 20, y1 + bh - 16), 0.45, GREY)
