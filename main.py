"""MemeCV: detecta expressões faciais e gestos das mãos pela webcam e mostra o meme correspondente.

Uso:
    pip install -r requirements.txt
    python main.py

Pressione ESC para sair.

Na primeira execução os modelos do MediaPipe (face_landmarker.task e hand_landmarker.task)
são baixados automaticamente para a pasta models/.
"""

import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets", "new")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/1/hand_landmarker.task",
    # usado pelo visor.py (detecção de objetos)
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/"
                                 "efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
}

# Estados possíveis (gesto/expressão estabilizado) -> arquivo do meme
SMILE = "sorriso"
PEACE = "paz"
THINKING = "pensando"
THUMBS_UP = "joinha"
TIMEOUT = "timeout"
NEUTRAL = "neutro"

# Nome do arquivo sem extensão; aceita .jpg, .jpeg, .png ou .webp
MEME_FILES = {
    SMILE: "sorriso",
    PEACE: "sorriso",
    THINKING: "pensando",
    NEUTRAL: "pensando",
    THUMBS_UP: "joinha",
    TIMEOUT: "timeout",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

HISTORY_SIZE = 7          # frames consecutivos necessários para trocar o meme
ESC_KEY = 27
MEME_WINDOW_HEIGHT = 480

WEBCAM_WINDOW = "MemeCV - Webcam"
MEME_WINDOW = "MemeCV - Meme"

# Cores em BGR
WHITE = (255, 255, 255)
PINK = (180, 105, 255)

# Limiares das detecções (ajuste se necessário para sua câmera/iluminação)
SMILE_WIDTH_RATIO = 0.42      # largura da boca / largura do rosto
SMILE_OPEN_RATIO = 0.06       # abertura da boca / altura do rosto
LOOK_UP_IRIS_RATIO = 0.38     # posição vertical da íris no olho (0 = topo, 1 = base)
HEAD_UP_NOSE_RATIO = 0.42     # posição vertical do nariz entre testa e queixo
TIMEOUT_TOUCH_RATIO = 0.6     # distância dedos da mão vertical -> mão horizontal / tamanho da mão
TIMEOUT_AXIS_TOLERANCE = 0.55 # quão inclinada a mão pode estar e ainda contar como vertical/horizontal
TIMEOUT_GRACE_FRAMES = 4      # frames em que o timeout "segura" se o MediaPipe perder uma das mãos

# Índices de landmarks do FaceMesh (o FaceLandmarker usa a mesma malha de 478 pontos)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14
FACE_LEFT, FACE_RIGHT = 234, 454
FOREHEAD, CHIN, NOSE_TIP = 10, 152, 1
# (íris, pálpebra superior, pálpebra inferior) de cada olho
EYES = ((468, 159, 145), (473, 386, 374))

# Índices de landmarks das mãos
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


# ---------------------------------------------------------------------------
# Utilitários geométricos
# ---------------------------------------------------------------------------

def _xy(landmark):
    return np.array([landmark.x, landmark.y])


def _dist(a, b):
    return float(np.linalg.norm(_xy(a) - _xy(b)))


def _hand_size(lm):
    """Tamanho de referência da mão: pulso até a base do dedo médio."""
    return _dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6


def finger_extended(lm, tip, pip):
    """Um dedo está esticado quando a ponta fica mais longe do pulso do que a articulação PIP.

    Usar distâncias (em vez de comparar só o eixo y) funciona com a mão em qualquer orientação.
    """
    return _dist(lm[tip], lm[WRIST]) > _dist(lm[pip], lm[WRIST]) * 1.1


def thumb_extended(lm):
    """Polegar esticado: ponta longe da base do indicador e alinhado com a articulação IP."""
    return (_dist(lm[THUMB_TIP], lm[INDEX_MCP]) > _hand_size(lm) * 0.6
            and _dist(lm[THUMB_TIP], lm[WRIST]) > _dist(lm[THUMB_IP], lm[WRIST]))


def fingers_state(lm):
    """Retorna (indicador, médio, anelar, mindinho) como booleanos de 'esticado'."""
    return (
        finger_extended(lm, INDEX_TIP, INDEX_PIP),
        finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP),
        finger_extended(lm, RING_TIP, RING_PIP),
        finger_extended(lm, PINKY_TIP, PINKY_PIP),
    )


# ---------------------------------------------------------------------------
# Detecção de expressões faciais (FaceLandmarker / malha do FaceMesh)
# ---------------------------------------------------------------------------

def is_big_smile(face):
    """Sorriso grande: boca larga em relação ao rosto e bem aberta."""
    face_width = _dist(face[FACE_LEFT], face[FACE_RIGHT]) or 1e-6
    face_height = _dist(face[FOREHEAD], face[CHIN]) or 1e-6
    mouth_width = _dist(face[MOUTH_LEFT], face[MOUTH_RIGHT]) / face_width
    mouth_open = _dist(face[LIP_TOP], face[LIP_BOTTOM]) / face_height
    return mouth_width > SMILE_WIDTH_RATIO and mouth_open > SMILE_OPEN_RATIO


def is_looking_up(face):
    """Olhar para cima / pose de pensamento: íris no alto dos olhos ou cabeça inclinada para trás."""
    iris_ratios = []
    for iris, top, bottom in EYES:
        eye_height = face[bottom].y - face[top].y
        if eye_height > 1e-6:
            iris_ratios.append((face[iris].y - face[top].y) / eye_height)
    eyes_up = bool(iris_ratios) and float(np.mean(iris_ratios)) < LOOK_UP_IRIS_RATIO

    face_height = face[CHIN].y - face[FOREHEAD].y
    head_up = face_height > 1e-6 and (face[NOSE_TIP].y - face[FOREHEAD].y) / face_height < HEAD_UP_NOSE_RATIO
    return eyes_up or head_up


# ---------------------------------------------------------------------------
# Detecção de gestos das mãos (HandLandmarker)
# ---------------------------------------------------------------------------

def is_peace_sign(lm):
    """Sinal de paz: indicador e médio levantados, anelar e mindinho abaixados."""
    index, middle, ring, pinky = fingers_state(lm)
    return index and middle and not ring and not pinky


def is_thumbs_up(lm):
    """Joinha: só o polegar levantado (apontando para cima), demais dedos dobrados."""
    if any(fingers_state(lm)) or not thumb_extended(lm):
        return False
    # O polegar precisa apontar para cima e ser o ponto mais alto da mão (y cresce para baixo).
    thumb_up = lm[THUMB_TIP].y < lm[THUMB_MCP].y - _hand_size(lm) * 0.3
    highest = all(lm[THUMB_TIP].y <= lm[i].y for i in range(21) if i != THUMB_TIP)
    return thumb_up and highest


def _hand_axis(lm):
    """Direção normalizada do eixo pulso -> base do dedo médio."""
    axis = _xy(lm[MIDDLE_MCP]) - _xy(lm[WRIST])
    return axis / (np.linalg.norm(axis) or 1e-6)


def hand_orientation(lm):
    """'vertical' se o eixo pulso->base do dedo médio for mais vertical que horizontal."""
    dx, dy = _hand_axis(lm)
    return "vertical" if abs(dy) > abs(dx) else "horizontal"


def timeout_distance(hands):
    """Quão perto as mãos estão de formar o T (menor = melhor; None se não há par válido).

    Para cada par (tronco, barra): o tronco precisa estar mais em pé e a barra mais deitada,
    com folga de TIMEOUT_AXIS_TOLERANCE. A distância é a menor entre a ponta de qualquer dedo
    (exceto o polegar) do tronco e qualquer ponto da barra, dividida pelo tamanho médio das mãos. Comparar com
    todos os pontos da barra (e não só o centro da palma) tolera mãos sobrepostas ou o toque
    perto dos dedos.
    """
    best = None
    for stem in hands:
        for bar in hands:
            if stem is bar:
                continue
            # o tronco aponta para cima (y cresce para baixo na imagem)
            if -_hand_axis(stem)[1] < TIMEOUT_AXIS_TOLERANCE:
                continue
            # o tronco é a mão aberta/esticada, não um punho fechado
            if not any(fingers_state(stem)[:2]):
                continue
            if abs(_hand_axis(bar)[0]) < TIMEOUT_AXIS_TOLERANCE:
                continue
            size = (_hand_size(stem) + _hand_size(bar)) / 2
            bar_points = np.array([_xy(p) for p in bar])
            for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
                dist = float(np.min(np.linalg.norm(bar_points - _xy(stem[tip]), axis=1))) / size
                best = dist if best is None else min(best, dist)
    return best


def is_timeout(hands):
    """Timeout (T): uma mão em pé com a ponta dos dedos encostando na mão deitada."""
    dist = timeout_distance(hands) if len(hands) >= 2 else None
    return dist is not None and dist < TIMEOUT_TOUCH_RATIO


# ---------------------------------------------------------------------------
# Classificação e estabilização
# ---------------------------------------------------------------------------

def classify(face_landmarks, hand_landmarks):
    """Escolhe o estado do frame atual. Gestos das mãos têm prioridade sobre o rosto.

    face_landmarks: lista de landmarks de um rosto (ou None)
    hand_landmarks: lista com a lista de landmarks de cada mão detectada
    """
    if is_timeout(hand_landmarks):
        return TIMEOUT
    for hand in hand_landmarks:
        if is_thumbs_up(hand):
            return THUMBS_UP
    for hand in hand_landmarks:
        if is_peace_sign(hand):
            return PEACE
    if face_landmarks is not None:
        if is_big_smile(face_landmarks):
            return SMILE
        if is_looking_up(face_landmarks):
            return THINKING
    return NEUTRAL


class TimeoutGrace:
    """Quando as mãos se sobrepõem no T, o MediaPipe às vezes perde uma delas por 1 ou 2 frames.

    Se o timeout foi visto há pouco e agora só aparece uma mão (ou nenhuma), mantém o timeout
    por até TIMEOUT_GRACE_FRAMES frames para o histórico de 7 frames não ser zerado.
    """

    def __init__(self, frames=TIMEOUT_GRACE_FRAMES):
        self.frames = frames
        self.remaining = 0

    def apply(self, gesture, num_hands):
        if gesture == TIMEOUT:
            self.remaining = self.frames
        elif self.remaining > 0 and num_hands < 2:
            self.remaining -= 1
            return TIMEOUT
        else:
            self.remaining = 0
        return gesture


class GestureStabilizer:
    """Só troca o estado quando o mesmo gesto aparece em HISTORY_SIZE frames consecutivos."""

    def __init__(self, size=HISTORY_SIZE, initial=NEUTRAL):
        self.history = deque(maxlen=size)
        self.current = initial

    def update(self, gesture):
        self.history.append(gesture)
        if len(self.history) == self.history.maxlen and len(set(self.history)) == 1:
            self.current = gesture
        return self.current


# ---------------------------------------------------------------------------
# Imagens e desenho
# ---------------------------------------------------------------------------

def _placeholder(filename):
    img = np.zeros((MEME_WINDOW_HEIGHT, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Imagem nao encontrada:", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(img, f"assets/new/{filename}.jpg", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    return img


def load_memes():
    """Carrega cada imagem uma vez; se faltar, usa um aviso no lugar."""
    cache, memes = {}, {}
    for state, filename in MEME_FILES.items():
        if filename not in cache:
            img = None
            for ext in IMAGE_EXTENSIONS:
                path = os.path.join(ASSETS_DIR, filename + ext)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    break
            if img is None:
                print(f"[aviso] Meme não encontrado: {os.path.join(ASSETS_DIR, filename)}.jpg")
                img = _placeholder(filename)
            else:
                scale = MEME_WINDOW_HEIGHT / img.shape[0]
                img = cv2.resize(img, (int(img.shape[1] * scale), MEME_WINDOW_HEIGHT))
            cache[filename] = img
        memes[state] = cache[filename]
    return memes


def draw_face(frame, face_landmarks):
    h, w = frame.shape[:2]
    for lm in face_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, WHITE, -1)


def draw_timeout_debug(frame, hands):
    """Mostra quantas mãos foram vistas e quão perto o T está do limiar (para ajustar o gesto)."""
    dist = timeout_distance(hands) if len(hands) >= 2 else None
    if len(hands) < 2:
        text = f"Maos: {len(hands)} (timeout precisa das 2 maos visiveis)"
    elif dist is None:
        text = "Maos: 2 | T: uma mao em pe e outra deitada"
    else:
        text = f"Maos: 2 | T: distancia {dist:.2f} (precisa < {TIMEOUT_TOUCH_RATIO:.2f})"
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINK, 2)


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], PINK, 2)
    for point in points:
        cv2.circle(frame, point, 4, PINK, -1)


# ---------------------------------------------------------------------------
# Modelos do MediaPipe
# ---------------------------------------------------------------------------

def model_path(filename):
    """Retorna o caminho do modelo, baixando-o na primeira execução."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"Baixando o modelo {filename} (só na primeira execução)...")
        try:
            urllib.request.urlretrieve(MODEL_URLS[filename], path + ".part")
        except OSError as exc:
            print(f"[ERRO] Não foi possível baixar {filename}: {exc}\n"
                  f"  Baixe manualmente de {MODEL_URLS[filename]}\n"
                  f"  e salve em {path}", file=sys.stderr)
            sys.exit(1)
        os.replace(path + ".part", path)
    return path


def create_face_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def open_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        print(
            "\n[ERRO] Não foi possível abrir a webcam.\n"
            "  - Se você está usando o WSL2, a webcam não fica disponível dentro do Linux.\n"
            "    Execute o programa pelo terminal do Windows (PowerShell ou Prompt de Comando):\n"
            "        pip install -r requirements.txt\n"
            "        python main.py\n"
            "  - Verifique também se a câmera está conectada e não está em uso por outro programa\n"
            "    (Teams, Zoom, navegador...) e se o Windows permite o acesso à câmera em\n"
            "    Configurações > Privacidade e segurança > Câmera.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return cap


def main():
    cap = open_webcam()
    memes = load_memes()
    stabilizer = GestureStabilizer()
    timeout_grace = TimeoutGrace()

    start = time.monotonic()
    last_ts = -1

    with create_face_landmarker() as face_landmarker, create_hand_landmarker() as hand_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERRO] A webcam parou de enviar imagens.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # O modo VIDEO exige timestamps estritamente crescentes.
            timestamp_ms = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = timestamp_ms
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            face = None
            if face_result.face_landmarks:
                face = face_result.face_landmarks[0]
                draw_face(frame, face)

            hand_list = hand_result.hand_landmarks or []
            for hand in hand_list:
                draw_hand(frame, hand)

            detected = timeout_grace.apply(classify(face, hand_list), len(hand_list))
            state = stabilizer.update(detected)

            cv2.putText(frame, f"Detectado: {detected} | Meme: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            draw_timeout_debug(frame, hand_list)
            cv2.imshow(WEBCAM_WINDOW, frame)
            cv2.imshow(MEME_WINDOW, memes[state])

            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
