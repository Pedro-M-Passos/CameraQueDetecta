"""Visor: protótipo da interface de um visor de ciborgue com o Wolf como IA de suporte.

Uso (PowerShell, na pasta do projeto):
    pip install -r requirements.txt
    ollama pull gemma3:4b                   # cérebro gratuito do Wolf (uma vez só)
    python visor.py

O Wolf usa o Ollama (gratuito, roda no seu PC). Para usar o Claude (pago), defina
$env:ANTHROPIC_API_KEY. Veja wolf.py para as outras opções.

Opções:
    --camera N            índice da webcam (padrão 0)
    --image FOTO          usa uma foto no lugar da webcam
    --snapshot SAIDA.png  com --image: desenha o HUD, salva a imagem e sai (sem janela)
    --width PIXELS        largura da janela do visor (padrão 1280)

Teclas: T ou ENTER fala com o Wolf, V manda a imagem atual para ele analisar,
H mostra a ajuda, ESC sai. Também dá para digitar para o Wolf no próprio terminal.

Reaproveita do MemeCV (main.py) os modelos de rosto e mãos e a classificação de gestos.
"""

import argparse
import queue
import sys
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

import main as memecv
import visor_hud as hud
from wolf import Wolf

OBJECT_MODEL = "efficientdet_lite0.tflite"
OBJECT_SCORE_THRESHOLD = 0.45
OBJECT_MAX_RESULTS = 5
SCAN_IMAGE_WIDTH = 1024   # largura máxima da imagem enviada ao Wolf
SNAPSHOT_FRAMES = 12      # quadros processados antes de salvar no modo --snapshot

ENTER_KEYS = (10, 13)
BACKSPACE_KEYS = (8, 127)

WINDOW = "Visor"

GESTURE_LABELS = {
    memecv.SMILE: "sorriso",
    memecv.PEACE: "paz",
    memecv.THINKING: "pensando",
    memecv.THUMBS_UP: "joinha",
    memecv.TIMEOUT: "timeout",
}
HANDEDNESS = {"Left": "MAO ESQ", "Right": "MAO DIR"}
# Nomes em português para as classes mais comuns do detector (as outras ficam em inglês)
OBJECT_NAMES = {
    "person": "PESSOA", "cell phone": "CELULAR", "cup": "COPO", "bottle": "GARRAFA",
    "chair": "CADEIRA", "laptop": "NOTEBOOK", "keyboard": "TECLADO", "mouse": "MOUSE",
    "remote": "CONTROLE", "tv": "TV", "book": "LIVRO", "backpack": "MOCHILA", "tie": "GRAVATA",
    "dog": "CÃO", "cat": "GATO", "scissors": "TESOURA", "clock": "RELÓGIO", "couch": "SOFÁ",
    "bed": "CAMA", "dining table": "MESA", "car": "CARRO", "bicycle": "BICICLETA",
}


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------

def create_object_detector():
    options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=memecv.model_path(OBJECT_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        max_results=OBJECT_MAX_RESULTS,
        score_threshold=OBJECT_SCORE_THRESHOLD,
    )
    return vision.ObjectDetector.create_from_options(options)


def landmarks_box(landmarks, w, h, pad=0.08):
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return ((x1 - px) * w, (y1 - py) * h, (x2 + px) * w, (y2 + py) * h)


class Sensors:
    """Roda rosto, mãos e objetos em cada quadro e monta a lista de alvos do HUD."""

    def __init__(self):
        self.face = memecv.create_face_landmarker()
        self.hands = memecv.create_hand_landmarker()
        self.objects = create_object_detector()
        self.stabilizer = memecv.GestureStabilizer()
        self.timeout_grace = memecv.TimeoutGrace()
        self.start = time.monotonic()
        self.last_ts = -1

    def close(self):
        self.face.close()
        self.hands.close()
        self.objects.close()

    def detect(self, frame):
        """Retorna (alvos em coordenadas normalizadas 0..1, landmarks das mãos, leituras)."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # O modo VIDEO exige timestamps estritamente crescentes.
        ts = max(int((time.monotonic() - self.start) * 1000), self.last_ts + 1)
        self.last_ts = ts

        face_result = self.face.detect_for_video(image, ts)
        hand_result = self.hands.detect_for_video(image, ts)
        object_result = self.objects.detect_for_video(image, ts)

        targets = []
        face = face_result.face_landmarks[0] if face_result.face_landmarks else None
        if face is not None:
            targets.append({"kind": "FACE", "box": landmarks_box(face, 1, 1),
                            "label": "BIO-SINAL", "sub": "ROSTO HUMANO"})

        hand_list = hand_result.hand_landmarks or []
        for i, hand in enumerate(hand_list):
            side = "MAO"
            if hand_result.handedness and i < len(hand_result.handedness):
                side = HANDEDNESS.get(hand_result.handedness[i][0].category_name, "MAO")
            targets.append({"kind": "HAND", "box": landmarks_box(hand, 1, 1),
                            "label": side, "sub": "RASTREANDO"})

        objects = []
        for det in object_result.detections:
            cat = det.categories[0]
            name = cat.category_name or "objeto"
            name = OBJECT_NAMES.get(name, name.upper())
            b = det.bounding_box
            objects.append((name, cat.score))
            targets.append({"kind": "OBJ",
                            "box": (b.origin_x / w, b.origin_y / h,
                                    (b.origin_x + b.width) / w, (b.origin_y + b.height) / h),
                            "label": name, "sub": f"CONFIANCA {cat.score * 100:.0f}%"})

        detected = self.timeout_grace.apply(memecv.classify(face, hand_list), len(hand_list))
        gesture = GESTURE_LABELS.get(self.stabilizer.update(detected))

        readings = {"face": face is not None, "hands": len(hand_list),
                    "objects": objects, "gesture": gesture}
        return targets, hand_list, readings


def sensors_summary(readings):
    """Texto curto com o que o visor vê agora, enviado junto com cada pergunta ao Wolf."""
    objects = ", ".join(f"{name.lower()} ({score:.0%})" for name, score in readings["objects"])
    return (f"rosto: {'sim' if readings['face'] else 'não'} | "
            f"mãos: {readings['hands']} | "
            f"objetos: {objects or 'nenhum'} | "
            f"gesto: {readings['gesture'] or 'nenhum'}")


# ---------------------------------------------------------------------------
# Desenho
# ---------------------------------------------------------------------------

def draw_hand_skeleton(img, hand):
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand]
    for a, b in memecv.HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], hud.RED_DIM, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 2, hud.WHITE, -1, cv2.LINE_AA)


def render(frame, targets, hand_list, readings, wolf_state, info, show_help, t):
    """Monta a imagem final do visor no tamanho da janela."""
    view = hud.apply_lens(frame)
    h, w = view.shape[:2]

    for hand in hand_list:
        draw_hand_skeleton(view, hand)
    for i, target in enumerate(targets):
        x1, y1, x2, y2 = target["box"]
        hud.draw_target(view, dict(target, box=(x1 * w, y1 * h, x2 * w, y2 * h)), i, t)

    hud.draw_frame(view, t)
    hud.draw_readouts(view, dict(info, targets=len(targets), face=readings["face"],
                                 hands=readings["hands"], objects=len(readings["objects"]),
                                 gesture=readings["gesture"]), t)
    hud.draw_wolf_panel(view, wolf_state, t)
    if show_help:
        hud.draw_help(view)
    return view


def encode_scan(frame):
    """JPEG da imagem limpa (sem HUD) para o Wolf analisar."""
    h, w = frame.shape[:2]
    if w > SCAN_IMAGE_WIDTH:
        frame = cv2.resize(frame, (SCAN_IMAGE_WIDTH, int(h * SCAN_IMAGE_WIDTH / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Entrada pelo terminal
# ---------------------------------------------------------------------------

def start_terminal_input(lines):
    """Lê linhas digitadas no terminal (aceita acentos) e coloca na fila."""
    if not sys.stdin or not sys.stdin.isatty():
        return

    def reader():
        for line in sys.stdin:
            if line.strip():
                lines.put(line.strip())

    threading.Thread(target=reader, daemon=True).start()


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Protótipo do visor com o Wolf")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image")
    parser.add_argument("--snapshot")
    parser.add_argument("--width", type=int, default=1280)
    return parser.parse_args()


def main():
    args = parse_args()
    still = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            sys.exit(f"[ERRO] Não foi possível abrir a imagem {args.image}")
        cap = None
    else:
        cap = memecv.open_webcam(args.camera)

    wolf = Wolf()
    wolf.greet()
    terminal_lines = queue.Queue()
    if not args.snapshot:
        start_terminal_input(terminal_lines)
        print("Visor ativo. T/ENTER fala com o Wolf, V analisa a cena, H ajuda, ESC sai.")

    sensors = Sensors()
    typing, typed, show_help = False, "", False
    started = last = time.monotonic()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("[ERRO] A webcam parou de enviar imagens.", file=sys.stderr)
                    break
                frame = cv2.flip(frame, 1)
            else:
                frame = still.copy()
            frame = resize_to_width(frame, args.width)

            targets, hand_list, readings = sensors.detect(frame)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            elapsed = int(now - started)
            info = {"fps": fps, "uptime": f"{elapsed // 60:02d}:{elapsed % 60:02d}",
                    "hint": "H: ajuda"}

            while not terminal_lines.empty():
                wolf.ask(terminal_lines.get(), sensors_summary(readings))

            wolf_state = dict(wolf.snapshot(), typing=typing, input=typed)
            view = render(frame, targets, hand_list, readings, wolf_state, info, show_help, now)

            frame_count += 1
            if args.snapshot:
                if frame_count >= SNAPSHOT_FRAMES:
                    cv2.imwrite(args.snapshot, view)
                    print(f"HUD salvo em {args.snapshot} | sensores: {sensors_summary(readings)}")
                    break
                continue

            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if typing:
                if key == memecv.ESC_KEY:
                    typing, typed = False, ""
                elif key in ENTER_KEYS:
                    if typed.strip():
                        wolf.ask(typed, sensors_summary(readings))
                    typing, typed = False, ""
                elif key in BACKSPACE_KEYS:
                    typed = typed[:-1]
                elif 32 <= key <= 126:
                    typed += chr(key)
            elif key == memecv.ESC_KEY:
                break
            elif key in ENTER_KEYS or key in (ord("t"), ord("T")):
                typing, typed = True, ""
            elif key in (ord("v"), ord("V")):
                wolf.ask("Analise a cena que o visor está vendo agora.",
                         sensors_summary(readings), encode_scan(frame))
            elif key in (ord("h"), ord("H")):
                show_help = not show_help
    finally:
        sensors.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
