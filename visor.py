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
    --detect-width PIXELS largura da imagem usada pelos detectores (padrão 480; menor = mais FPS)
    --object-every N      detector de objetos a cada N quadros (padrão 3; maior = mais FPS)
    --cam-width/--cam-height  resolução pedida à webcam (padrão 640x480)
    --no-objects, --no-lens   começa com objetos ou efeito da lente desligados

Teclas: T ou ENTER fala com o Wolf, V manda a imagem atual para ele analisar,
1/2/3 ligam e desligam rosto, mãos e objetos, L liga e desliga o efeito da lente,
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
DETECT_WIDTH = 480        # os detectores rodam numa cópia pequena do quadro (bem mais rápido)
OBJECT_EVERY = 3          # o detector de objetos roda 1 vez a cada N quadros analisados
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
DETECTOR_KEYS = {"1": "face", "2": "hands", "3": "objects"}
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
    """Roda rosto, mãos e objetos e monta a lista de alvos do HUD.

    Cada detector pode ser ligado/desligado (teclas 1, 2 e 3). O de objetos é o mais pesado,
    então roda só a cada object_every quadros e repete o último resultado nos intervalos.
    """

    def __init__(self, object_every=OBJECT_EVERY):
        self.face = memecv.create_face_landmarker()
        self.hands = memecv.create_hand_landmarker()
        self.objects = create_object_detector()
        self.enabled = {"face": True, "hands": True, "objects": True}
        self.object_every = max(1, object_every)
        self.stabilizer = memecv.GestureStabilizer()
        self.timeout_grace = memecv.TimeoutGrace()
        self.start = time.monotonic()
        self.last_ts = -1
        self.count = 0
        self.last_objects = []

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
        self.count += 1

        targets = []
        face = None
        if self.enabled["face"]:
            face_result = self.face.detect_for_video(image, ts)
            face = face_result.face_landmarks[0] if face_result.face_landmarks else None
        if face is not None:
            targets.append({"kind": "FACE", "box": landmarks_box(face, 1, 1),
                            "label": "BIO-SINAL", "sub": "ROSTO HUMANO"})

        hand_list = []
        if self.enabled["hands"]:
            hand_result = self.hands.detect_for_video(image, ts)
            hand_list = hand_result.hand_landmarks or []
            for i, hand in enumerate(hand_list):
                side = "MAO"
                if hand_result.handedness and i < len(hand_result.handedness):
                    side = HANDEDNESS.get(hand_result.handedness[i][0].category_name, "MAO")
                targets.append({"kind": "HAND", "box": landmarks_box(hand, 1, 1),
                                "label": side, "sub": "RASTREANDO"})

        if not self.enabled["objects"]:
            self.last_objects = []
        elif self.count % self.object_every == 1 or self.object_every == 1:
            self.last_objects = []
            for det in self.objects.detect_for_video(image, ts).detections:
                cat = det.categories[0]
                name = cat.category_name or "objeto"
                b = det.bounding_box
                self.last_objects.append({
                    "kind": "OBJ", "label": OBJECT_NAMES.get(name, name.upper()), "score": cat.score,
                    "box": (b.origin_x / w, b.origin_y / h,
                            (b.origin_x + b.width) / w, (b.origin_y + b.height) / h),
                    "sub": f"CONFIANCA {cat.score * 100:.0f}%"})
        targets += self.last_objects

        detected = self.timeout_grace.apply(memecv.classify(face, hand_list), len(hand_list))
        gesture = GESTURE_LABELS.get(self.stabilizer.update(detected))

        readings = {"face": face is not None, "hands": len(hand_list),
                    "objects": [(o["label"], o["score"]) for o in self.last_objects],
                    "gesture": gesture}
        return targets, hand_list, readings


EMPTY_READINGS = {"face": False, "hands": 0, "objects": [], "gesture": None}


class DetectionWorker:
    """Roda os detectores numa thread própria, sempre no quadro mais recente.

    Assim a janela continua fluida mesmo quando a detecção é mais lenta que a câmera:
    o vídeo é desenhado em todo quadro e as miras usam o último resultado pronto.
    """

    def __init__(self, sensors):
        self.sensors = sensors
        self.result = ([], [], EMPTY_READINGS)
        self.rate = 0.0
        self._frame = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._frame = frame
        self._new.set()

    def latest(self):
        with self._lock:
            return self.result

    def stop(self):
        self._stop = True
        self._new.set()
        self._thread.join(timeout=2)

    def _loop(self):
        last = time.monotonic()
        while not self._stop:
            self._new.wait()
            self._new.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None or self._stop:
                continue
            result = self.sensors.detect(frame)
            now = time.monotonic()
            self.rate = 0.9 * self.rate + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            with self._lock:
                self.result = result


class CameraStream:
    """Lê a webcam numa thread própria; read() devolve na hora o quadro mais novo."""

    def __init__(self, index, width, height):
        self.cap = memecv.open_webcam(index)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, self._frame = self.cap.read()
        self.ok = ok
        self._seq = 0
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            with self._lock:
                self.ok = ok
                if ok:
                    self._frame = frame
                    self._seq += 1
                self._new.notify_all()
            if not ok:
                break

    def read(self, last_seq=-1, timeout=0.05):
        """Retorna (ok, quadro, número do quadro), esperando um pouco por um quadro mais novo
        que last_seq para não redesenhar a mesma imagem à toa."""
        with self._new:
            self._new.wait_for(lambda: self._seq != last_seq or not self.ok, timeout)
            return self.ok, self._frame, self._seq

    def release(self):
        self._stop = True
        self._thread.join(timeout=2)
        self.cap.release()


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


def render(frame, targets, hand_list, readings, wolf_state, info, show_help, t, lens=True):
    """Monta a imagem final do visor no tamanho da janela."""
    view = hud.apply_lens(frame) if lens else frame.copy()
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
    parser.add_argument("--camera", type=int, default=0, help="índice da webcam")
    parser.add_argument("--image", help="usa uma foto no lugar da webcam")
    parser.add_argument("--snapshot", help="com --image: salva o HUD neste arquivo e sai")
    parser.add_argument("--width", type=int, default=1280, help="largura da janela do visor")
    parser.add_argument("--cam-width", type=int, default=640, help="resolução pedida à webcam")
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--detect-width", type=int, default=DETECT_WIDTH,
                        help="largura da imagem usada pelos detectores (menor = mais rápido)")
    parser.add_argument("--object-every", type=int, default=OBJECT_EVERY,
                        help="roda o detector de objetos a cada N quadros (maior = mais rápido)")
    parser.add_argument("--no-objects", action="store_true", help="começa com objetos desligados")
    parser.add_argument("--no-lens", action="store_true", help="começa sem o efeito da lente")
    return parser.parse_args()


def run_snapshot(args, still, wolf):
    """Modo de teste sem câmera: detecta na foto, desenha o HUD e salva."""
    sensors = Sensors(args.object_every)
    sensors.enabled["objects"] = not args.no_objects
    frame = resize_to_width(still, args.width)
    small = resize_to_width(frame, args.detect_width)
    try:
        for _ in range(SNAPSHOT_FRAMES):
            targets, hand_list, readings = sensors.detect(small)
    finally:
        sensors.close()
    info = {"fps": 0.0, "det_rate": 0.0, "uptime": "00:00", "hint": "H: ajuda",
            "sensors": sensors.enabled}
    wolf_state = dict(wolf.snapshot(), typing=False, input="")
    view = render(frame, targets, hand_list, readings, wolf_state, info, False, time.monotonic(),
                  lens=not args.no_lens)
    cv2.imwrite(args.snapshot, view)
    print(f"HUD salvo em {args.snapshot} | sensores: {sensors_summary(readings)}")


def main():
    args = parse_args()
    still = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            sys.exit(f"[ERRO] Não foi possível abrir a imagem {args.image}")

    wolf = Wolf()
    wolf.greet()
    if args.snapshot and still is not None:
        run_snapshot(args, still, wolf)
        return

    camera = None if still is not None else CameraStream(args.camera, args.cam_width, args.cam_height)
    terminal_lines = queue.Queue()
    start_terminal_input(terminal_lines)
    print("Visor ativo. T/ENTER fala com o Wolf, V analisa a cena, 1/2/3/L ajustam o FPS, "
          "H ajuda, ESC sai.")

    sensors = Sensors(args.object_every)
    sensors.enabled["objects"] = not args.no_objects
    worker = DetectionWorker(sensors)
    typing, typed, show_help, lens = False, "", False, not args.no_lens
    started = last = time.monotonic()
    fps = 0.0
    last_seq = -1

    try:
        while True:
            if camera is not None:
                ok, frame, seq = camera.read(last_seq)
                if not ok:
                    print("[ERRO] A webcam parou de enviar imagens.", file=sys.stderr)
                    break
                frame = cv2.flip(frame, 1)
            else:
                frame, seq = still, last_seq + 1
            frame = resize_to_width(frame, args.width)

            # Só manda para os detectores quando a câmera trouxe um quadro novo.
            if seq != last_seq:
                worker.submit(resize_to_width(frame, args.detect_width))
                last_seq = seq
            targets, hand_list, readings = worker.latest()

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            elapsed = int(now - started)
            info = {"fps": fps, "det_rate": worker.rate,
                    "uptime": f"{elapsed // 60:02d}:{elapsed % 60:02d}",
                    "hint": "H: ajuda", "sensors": sensors.enabled}

            while not terminal_lines.empty():
                wolf.ask(terminal_lines.get(), sensors_summary(readings))

            wolf_state = dict(wolf.snapshot(), typing=typing, input=typed)
            view = render(frame, targets, hand_list, readings, wolf_state, info, show_help, now,
                          lens=lens)
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
            elif key in (ord("l"), ord("L")):
                lens = not lens
            elif chr(key) in DETECTOR_KEYS:
                name = DETECTOR_KEYS[chr(key)]
                sensors.enabled[name] = not sensors.enabled[name]
    finally:
        worker.stop()
        sensors.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
