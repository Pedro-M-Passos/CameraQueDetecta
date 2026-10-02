"""Voz do Wolf: fala as respostas com um efeito robótico, sem internet e sem instalar nada.

Como funciona:
  1. Um sintetizador de voz que já vem no sistema gera um .wav com o texto:
     - Windows: as vozes do próprio Windows, chamadas pelo PowerShell. Aparecem as vozes
       "modernas" (as de Configurações > Fala, como Maria e Daniel) e as clássicas (SAPI).
     - Linux: espeak-ng (se estiver instalado).
  2. O áudio passa por um efeito (numpy). O tom fica mais grave sem deixar a fala mais lenta.
  3. O som toca em segundo plano (winsound no Windows, aplay no Linux).

Tudo roda numa thread própria, então o vídeo não trava enquanto o Wolf fala.
No visor: M liga/desliga a voz, N troca de voz, F troca o efeito. A escolha fica salva.

Autoteste e lista de vozes (no terminal):
    python voice.py --vozes            lista as vozes instaladas
    python voice.py "teste de voz"     testa a voz atual
    python voice.py --ouvir            cada voz fala o próprio nome, para você escolher

Variáveis de ambiente (opcionais; a escolha salva pelo visor vale quando elas não existem):
    WOLF_VOICE=0            começa com a voz desligada
    WOLF_VOICE_NAME         parte do nome da voz (ex.: "Daniel")
    WOLF_VOICE_SPEED        velocidade da fala (1.0 = normal, 1.2 = 20% mais rápida)
    WOLF_VOICE_FX           efeito: robo, leve ou nenhum
"""

import atexit
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".wolf_voz.json")
DEFAULT_SPEED = 1.1

# Efeitos (ajuste a gosto). pitch < 1 deixa a voz mais grave, sem mudar a velocidade.
EFFECTS = {
    "robo": {"pitch": 0.85, "ring_hz": 55, "ring_mix": 0.3, "echo_ms": 14, "echo_gain": 0.3,
             "band": (250, 3800), "bits": 10},
    "leve": {"pitch": 0.93, "ring_hz": 40, "ring_mix": 0.1, "echo_ms": 10, "echo_gain": 0.18,
             "band": (150, 5000), "bits": 0},
    "nenhum": None,
}
EFFECT_ORDER = ["robo", "leve", "nenhum"]
VOLUME = 0.9

# ---------------------------------------------------------------------------
# Scripts do PowerShell (Windows)
# ---------------------------------------------------------------------------

# Funções para esperar operações assíncronas do Windows (WinRT) dentro do PowerShell.
_PS_WINRT = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
    $task.Wait(-1) | Out-Null
    $task.Result
}
[Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null
"""

# Lista as vozes modernas (se o Windows tiver) e as clássicas, uma por linha: tipo|nome|idioma
_PS_LIST = "try {\n" + _PS_WINRT + r"""
[Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | ForEach-Object {
    "moderna|$($_.DisplayName)|$($_.Language)" }
} catch { }
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object {
    "classica|$($_.VoiceInfo.Name)|$($_.VoiceInfo.Culture.Name)" }
$s.Dispose()
"""

_PS_SPEAK_MODERN = _PS_WINRT + r"""
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
    Where-Object { $_.DisplayName -eq $env:WOLF_TTS_VOICE } | Select-Object -First 1
if ($voice) { $synth.Voice = $voice }
try { $synth.Options.SpeakingRate = [double]::Parse($env:WOLF_TTS_SPEED, [Globalization.CultureInfo]::InvariantCulture) } catch { }
$stream = Await ($synth.SynthesizeTextToStreamAsync($env:WOLF_TTS_TEXT)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$size = [uint32]$stream.Size
$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
Await ($reader.LoadAsync($size)) ([uint32]) | Out-Null
$bytes = New-Object byte[] $size
$reader.ReadBytes($bytes)
[IO.File]::WriteAllBytes($env:WOLF_TTS_OUT, $bytes)
"""

_PS_SPEAK_CLASSIC = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:WOLF_TTS_VOICE) { $s.SelectVoice($env:WOLF_TTS_VOICE) }
$s.Rate = [int]$env:WOLF_TTS_RATE
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($env:WOLF_TTS_OUT, $fmt)
$s.Speak($env:WOLF_TTS_TEXT)
$s.Dispose()
"""


def run_powershell(script, env=None, timeout=60):
    """Roda um script no PowerShell do Windows.

    O script vai em -EncodedCommand (base64 UTF-16) para aspas e quebras de linha não serem
    estragadas pela linha de comando do Windows.
    """
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-EncodedCommand", encoded],
        env=env, capture_output=True, text=True, timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def clean_for_speech(text):
    """Tira marcações (asteriscos, colchetes) que o sintetizador leria em voz alta."""
    text = re.sub(r"[*_#`~>\[\]]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Vozes
# ---------------------------------------------------------------------------

def list_voices():
    """Lista as vozes instaladas: [{"engine", "name", "lang"}], as em português primeiro."""
    voices = []
    if sys.platform == "win32":
        if not shutil.which("powershell"):
            raise RuntimeError("PowerShell não encontrado no PATH")
        result = run_powershell(_PS_LIST)
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[1]:
                voices.append({"engine": parts[0], "name": parts[1], "lang": parts[2]})
    else:
        engine = shutil.which("espeak-ng") or shutil.which("espeak")
        if engine:
            voices.append({"engine": "espeak", "name": "pt-br", "lang": "pt-BR"})
    # Português primeiro; entre elas, as modernas antes das clássicas.
    voices.sort(key=lambda v: (not v["lang"].lower().startswith("pt"), v["engine"] != "moderna"))
    return voices


def find_voice(voices, name):
    if not name:
        return None
    name = name.lower()
    for v in voices:
        if v["name"].lower() == name:
            return v
    for v in voices:
        if name in v["name"].lower():
            return v
    return None


def synthesize(text, path, voice, speed=DEFAULT_SPEED):
    """Gera um .wav com o texto na voz escolhida (um item de list_voices())."""
    if voice is None:
        raise RuntimeError("nenhuma voz instalada")
    if voice["engine"] == "espeak":
        engine = shutil.which("espeak-ng") or shutil.which("espeak")
        if not engine:
            raise RuntimeError("espeak-ng não encontrado")
        subprocess.run([engine, "-v", voice["name"], "-s", str(int(165 * speed)), "-w", path, text],
                       check=True, capture_output=True, timeout=60)
        return
    env = dict(os.environ, WOLF_TTS_TEXT=text, WOLF_TTS_OUT=path, WOLF_TTS_VOICE=voice["name"],
               WOLF_TTS_SPEED=f"{speed:.2f}",
               WOLF_TTS_RATE=str(max(-10, min(10, round((speed - 1) * 10)))))
    script = _PS_SPEAK_MODERN if voice["engine"] == "moderna" else _PS_SPEAK_CLASSIC
    if os.path.exists(path):
        os.remove(path)
    result = run_powershell(script, env)
    if result.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) < 100:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(f"a voz {voice['name']} falhou: " + (detail[0] if detail else "sem detalhes"))


# ---------------------------------------------------------------------------
# Áudio
# ---------------------------------------------------------------------------

def read_wav(path):
    with wave.open(path, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise RuntimeError(f"formato de áudio inesperado ({width * 8} bits)")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def write_wav(path, audio, rate):
    data = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def time_stretch(audio, rate, ratio, frame_ms=40, tolerance_ms=10):
    """Muda a duração sem mudar o tom (WSOLA: sobrepõe trechos alinhados pela forma da onda).

    ratio > 1 deixa mais longo, < 1 mais curto.
    """
    n = max(64, int(rate * frame_ms / 1000)) // 2 * 2
    hs = n // 2                                  # passo na saída (50% de sobreposição)
    ha = hs / ratio                              # passo na entrada
    tol = int(rate * tolerance_ms / 1000)
    out_len = int(len(audio) * ratio)
    window = np.hanning(n).astype(np.float32)
    padded = np.concatenate([np.zeros(tol, np.float32), audio, np.zeros(n + 2 * tol, np.float32)])
    out = np.zeros(out_len + n, dtype=np.float32)
    norm = np.zeros_like(out)
    prev = 0                                     # início (na entrada) do último trecho usado
    for k, out_pos in enumerate(range(0, out_len, hs)):
        nominal = int(k * ha) + tol
        if k == 0:
            best = nominal
        else:
            # O trecho que "continuaria" naturalmente o anterior; procura perto da posição
            # nominal o trecho mais parecido com ele, para não haver cancelamento de fase.
            target = padded[prev + hs:prev + hs + n]
            lo, hi = max(0, nominal - tol), min(len(padded) - n, nominal + tol)
            if hi <= lo:
                break
            region = padded[lo:hi + n]
            corr = np.correlate(region, target, mode="valid")
            best = lo + int(np.argmax(corr))
        out[out_pos:out_pos + n] += padded[best:best + n] * window
        norm[out_pos:out_pos + n] += window
        prev = best
    return out[:out_len] / np.maximum(norm[:out_len], 1e-3)


def pitch_shift(audio, rate, factor):
    """Muda o tom sem mudar a duração. factor < 1 deixa a voz mais grave."""
    if abs(factor - 1) < 1e-3 or audio.size < 256:
        return audio
    # 1) Reamostra: tom mais grave, mas fala mais lenta. 2) Comprime o tempo de volta.
    longer = int(len(audio) / factor)
    resampled = np.interp(np.linspace(0, len(audio) - 1, longer), np.arange(len(audio)),
                          audio).astype(np.float32)
    shifted = time_stretch(resampled, rate, len(audio) / longer)
    return shifted[:len(audio)]


def apply_effect(audio, rate, name="robo"):
    """Aplica o efeito escolhido e devolve o áudio novo (mesma duração)."""
    fx = EFFECTS.get(name)
    if audio.size == 0:
        return audio
    if fx:
        audio = pitch_shift(audio, rate, fx["pitch"])
        # Modulação em anel: mistura a voz com ela mesma multiplicada por um seno grave.
        t = np.arange(len(audio)) / rate
        audio = audio * (1 - fx["ring_mix"]) + audio * np.sin(2 * np.pi * fx["ring_hz"] * t) * fx["ring_mix"]
        # Eco curto, como se a voz saísse de dentro de um capacete.
        delay = int(rate * fx["echo_ms"] / 1000)
        if 0 < delay < len(audio):
            echo = np.zeros_like(audio)
            echo[delay:] = audio[:-delay] * fx["echo_gain"]
            audio = audio + echo
        # Filtro de rádio: atenua graves e agudos extremos.
        spectrum = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / rate)
        low, high = fx["band"]
        spectrum[(freqs < low) | (freqs > high)] *= 0.08
        audio = np.fft.irfft(spectrum, n=len(audio))
        if fx["bits"]:
            steps = 2 ** (fx["bits"] - 1)
            audio = np.round(audio * steps) / steps
    peak = float(np.max(np.abs(audio))) or 1.0
    return audio / peak * VOLUME


def play(path, wait=False):
    """Toca o .wav (um som novo interrompe o anterior)."""
    if sys.platform == "win32":
        import winsound
        flags = winsound.SND_FILENAME | (0 if wait else winsound.SND_ASYNC)
        winsound.PlaySound(path, flags)
        return True
    player = shutil.which("aplay") or shutil.which("paplay")
    if player:
        proc = subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if wait:
            proc.wait()
        return True
    return False


# ---------------------------------------------------------------------------
# Configuração salva
# ---------------------------------------------------------------------------

def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def current_settings():
    """(nome da voz, efeito, velocidade): variáveis de ambiente > escolha salva > padrão."""
    saved = load_settings()
    name = os.environ.get("WOLF_VOICE_NAME") or saved.get("voz")
    effect = os.environ.get("WOLF_VOICE_FX") or saved.get("efeito") or "robo"
    if effect not in EFFECTS:
        effect = "robo"
    try:
        speed = float(os.environ.get("WOLF_VOICE_SPEED") or saved.get("velocidade") or DEFAULT_SPEED)
    except ValueError:
        speed = DEFAULT_SPEED
    return name, effect, speed


# ---------------------------------------------------------------------------
# Voz do Wolf
# ---------------------------------------------------------------------------

class Voice:
    """Fila de falas do Wolf. speak() volta na hora; a síntese e o som rodam numa thread."""

    def __init__(self, enabled=None, printer=print, on_error=None):
        if enabled is None:
            enabled = os.environ.get("WOLF_VOICE", "1") != "0"
        self.enabled = enabled
        self.voice_name, self.effect, self.speed = current_settings()
        self.voice = None             # item de list_voices() em uso
        self.voices = None            # carregada na thread (o PowerShell demora ~1 s)
        self.speaking_until = 0.0     # instante (time.monotonic) em que a fala atual termina
        self._print = printer
        self.on_error = on_error      # chamado uma vez com a mensagem se a voz falhar
        self._queue = queue.Queue()
        self._dir = tempfile.mkdtemp(prefix="wolf_voz_")
        atexit.register(shutil.rmtree, self._dir, True)
        self._count = 0
        self._warned = False
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def speaking(self):
        return time.monotonic() < self.speaking_until

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.stop()
        return self.enabled

    def stop(self):
        self.speaking_until = 0.0
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)

    def speak(self, text):
        text = clean_for_speech(text)
        if self.enabled and text:
            self._put(("say", text))

    def next_voice(self):
        """Passa para a próxima voz instalada e fala uma amostra."""
        self._put(("next_voice", None))

    def next_effect(self):
        self.effect = EFFECT_ORDER[(EFFECT_ORDER.index(self.effect) + 1) % len(EFFECT_ORDER)]
        self._save()
        self._print(f"[Wolf] efeito da voz: {self.effect}")
        self._put(("say", f"Efeito {self.effect}."))

    # ------------------------------------------------------------------

    def _put(self, item):
        # Fala nova descarta a que ainda não começou (para não acumular).
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(item)

    def _save(self):
        save_settings({"voz": self.voice["name"] if self.voice else self.voice_name,
                       "efeito": self.effect, "velocidade": self.speed})

    def _ensure_voices(self):
        if self.voices is None:
            self.voices = list_voices()
            self.voice = find_voice(self.voices, self.voice_name) or (self.voices[0] if self.voices else None)
            if self.voice:
                self._print(f"[Wolf] voz: {self.voice['name']} ({self.voice['lang']}) | "
                            f"{len(self.voices)} vozes instaladas, tecla N troca")

    def render(self, text):
        """Sintetiza + aplica efeitos. Retorna (caminho do .wav, duração em segundos)."""
        self._ensure_voices()
        self._count += 1
        raw = os.path.join(self._dir, f"bruto_{self._count % 4}.wav")
        out = os.path.join(self._dir, f"wolf_{self._count % 4}.wav")
        synthesize(text, raw, self.voice, self.speed)
        audio, rate = read_wav(raw)
        audio = apply_effect(audio, rate, self.effect)
        write_wav(out, audio, rate)
        return out, len(audio) / rate

    def _loop(self):
        while True:
            kind, text = self._queue.get()
            try:
                if kind == "next_voice":
                    self._ensure_voices()
                    if not self.voices:
                        raise RuntimeError("nenhuma voz instalada")
                    i = self.voices.index(self.voice) if self.voice in self.voices else -1
                    self.voice = self.voices[(i + 1) % len(self.voices)]
                    self._save()
                    self._print(f"[Wolf] voz {self.voices.index(self.voice) + 1}/{len(self.voices)}: "
                                f"{self.voice['name']} ({self.voice['lang']})")
                    text = f"Voz {self.voice['name'].replace('Microsoft', '')}."
                if not self.enabled:
                    continue
                path, seconds = self.render(text)
                if self.enabled and play(path):
                    self.speaking_until = time.monotonic() + seconds
            except Exception as exc:  # noqa: BLE001 - a voz nunca derruba o visor
                if not self._warned:
                    self._warned = True
                    self._print(f"[Wolf] voz indisponível: {exc}")
                    self._print("[Wolf] Para diagnosticar, rode no terminal: python voice.py")
                    if self.on_error:
                        self.on_error(f"Sem voz: {exc}. Rode python voice.py para diagnosticar.")


# ---------------------------------------------------------------------------
# Autoteste: python voice.py [--vozes | --ouvir | "texto"]
# ---------------------------------------------------------------------------

def print_voices(voices, current=None):
    print("Vozes instaladas:")
    for i, v in enumerate(voices, 1):
        mark = "  <- atual" if current and v["name"] == current["name"] else ""
        print(f"  {i:2d}. {v['name']}  ({v['lang']}, {v['engine']}){mark}")
    if not voices:
        print("  (nenhuma)")


def self_test(args):
    """Testa cada etapa da voz e mostra onde parou."""
    import traceback

    print(f"Sistema: {sys.platform} | Python {sys.version.split()[0]}")
    name, effect, speed = current_settings()
    step = "listar vozes"
    try:
        voices = list_voices()
        voice = find_voice(voices, name) or (voices[0] if voices else None)
        print_voices(voices, voice)
        if args and args[0] == "--vozes":
            print('\nPara escolher: aperte N no visor, ou $env:WOLF_VOICE_NAME = "Daniel"')
            return
        tmp = tempfile.mkdtemp(prefix="wolf_teste_")
        if args and args[0] == "--ouvir":
            for i, v in enumerate(voices, 1):
                step = f"falar com {v['name']}"
                path = os.path.join(tmp, f"voz_{i}.wav")
                synthesize(f"Voz {i}. {v['name'].replace('Microsoft', '')}. Aqui é o Wolf.", path, v, speed)
                audio, rate = read_wav(path)
                write_wav(path, apply_effect(audio, rate, effect), rate)
                print(f"  tocando {i}. {v['name']}")
                play(path, wait=True)
            print("\nGostou de alguma? No visor, aperte N até chegar nela (a escolha fica salva).")
            return
        text = " ".join(args) or "Teste de voz. Aqui é o Wolf, pronto para a missão."
        raw, out = os.path.join(tmp, "bruto.wav"), os.path.join(tmp, "wolf.wav")
        step = "gerar a fala"
        synthesize(text, raw, voice, speed)
        step = "ler o áudio"
        audio, rate = read_wav(raw)
        print(f"Voz: {voice['name']} | efeito: {effect} | velocidade: {speed} | "
              f"{len(audio) / rate:.1f} s a {rate} Hz")
        step = "aplicar o efeito"
        write_wav(out, apply_effect(audio, rate, effect), rate)
        step = "tocar o som"
        print("Tocando a voz normal e depois a do Wolf...")
        if not (play(raw, wait=True) and play(out, wait=True)):
            print("  (nenhum player de áudio encontrado)")
        print(f"OK. Arquivos em {tmp}")
    except Exception:  # noqa: BLE001
        print(f"\nFALHOU na etapa: {step}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    self_test(sys.argv[1:])
