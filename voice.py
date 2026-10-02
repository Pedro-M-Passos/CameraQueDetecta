"""Voz do Wolf: fala as respostas com um efeito robótico, sem internet e sem instalar nada.

Como funciona:
  1. Um sintetizador de voz que já vem no sistema gera um .wav com o texto:
     - Windows: a voz do próprio Windows (System.Speech, chamada pelo PowerShell).
       Usa uma voz em português se houver uma instalada.
     - Linux: espeak-ng (se estiver instalado).
  2. O áudio passa pelos efeitos (numpy): voz mais grave, modulação metálica, eco curto
     e filtro de rádio.
  3. O som toca em segundo plano (winsound no Windows, aplay no Linux).

Tudo roda numa thread própria, então o vídeo não trava enquanto o Wolf fala.

Variáveis de ambiente:
    WOLF_VOICE=0        começa com a voz desligada (a tecla M liga/desliga)
    WOLF_VOICE_NAME     parte do nome da voz do Windows a usar (ex.: "Maria" ou "Daniel")
"""

import atexit
import base64
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

# Efeitos (ajuste a gosto)
PITCH = 0.86            # < 1 deixa a voz mais grave (e um pouco mais lenta)
RING_HZ = 55            # frequência da modulação metálica
RING_MIX = 0.35         # quanto da modulação entra (0 = nenhuma, 1 = só robô)
ECHO_MS = 14            # eco curto que dá o som "de capacete"
ECHO_GAIN = 0.35
BAND = (250, 3800)      # filtro de rádio: só passa esta faixa de frequências (Hz)
BITS = 10               # leve "bitcrush" (menos bits = mais digital)
VOLUME = 0.9

RATE_WINDOWS = 1        # velocidade da fala no Windows (-10 a 10)

_PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo }
$pick = $null
if ($env:WOLF_VOICE_NAME) { $pick = $voices | Where-Object { $_.Name -like "*$($env:WOLF_VOICE_NAME)*" } | Select-Object -First 1 }
if (-not $pick) { $pick = $voices | Where-Object { $_.Culture.Name -like 'pt*' } | Select-Object -First 1 }
if ($pick) { $s.SelectVoice($pick.Name) }
$s.Rate = [int]$env:WOLF_TTS_RATE
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($env:WOLF_TTS_OUT, $fmt)
$s.Speak($env:WOLF_TTS_TEXT)
$s.Dispose()
if ($pick) { Write-Output $pick.Name } else { Write-Output "padrao" }
"""

# Lista as vozes instaladas (usado pelo autoteste: python voice.py)
_PS_LIST_VOICES = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.GetInstalledVoices() | ForEach-Object { "$($_.VoiceInfo.Name) | $($_.VoiceInfo.Culture.Name) | ativa=$($_.Enabled)" }
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
# Síntese
# ---------------------------------------------------------------------------

def synthesize(text, path):
    """Gera um .wav com o texto. Retorna o nome da voz usada, ou levanta RuntimeError."""
    if sys.platform == "win32":
        if not shutil.which("powershell"):
            raise RuntimeError("PowerShell não encontrado no PATH")
        env = dict(os.environ, WOLF_TTS_TEXT=text, WOLF_TTS_OUT=path, WOLF_TTS_RATE=str(RATE_WINDOWS))
        result = run_powershell(_PS_SCRIPT, env)
        if result.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) < 100:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError("a voz do Windows falhou: " + (detail[0] if detail else "sem detalhes"))
        return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "padrao"
    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if not engine:
        raise RuntimeError("nenhum sintetizador encontrado (instale o espeak-ng)")
    subprocess.run([engine, "-v", "pt-br", "-s", "165", "-w", path, text],
                   check=True, capture_output=True, timeout=60)
    return os.path.basename(engine)


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


# ---------------------------------------------------------------------------
# Efeitos
# ---------------------------------------------------------------------------

def robotize(audio, rate):
    """Aplica os efeitos da voz do Wolf e devolve o áudio novo."""
    if audio.size == 0:
        return audio
    # 1) Mais grave: "toca mais devagar" reamostrando o sinal.
    n = int(len(audio) / PITCH)
    audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)

    # 2) Modulação em anel: mistura a voz com ela mesma multiplicada por um seno grave.
    t = np.arange(len(audio)) / rate
    audio = audio * (1 - RING_MIX) + audio * np.sin(2 * np.pi * RING_HZ * t) * RING_MIX

    # 3) Eco curto, que soa como se a voz saísse de dentro de um capacete.
    delay = int(rate * ECHO_MS / 1000)
    if 0 < delay < len(audio):
        echo = np.zeros_like(audio)
        echo[delay:] = audio[:-delay] * ECHO_GAIN
        audio = audio + echo

    # 4) Filtro de rádio: corta graves e agudos extremos (no domínio da frequência).
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / rate)
    spectrum[(freqs < BAND[0]) | (freqs > BAND[1])] *= 0.08
    audio = np.fft.irfft(spectrum, n=len(audio))

    # 5) Leve redução de bits para um toque digital.
    steps = 2 ** (BITS - 1)
    audio = np.round(audio * steps) / steps

    peak = float(np.max(np.abs(audio))) or 1.0
    return audio / peak * VOLUME


# ---------------------------------------------------------------------------
# Reprodução
# ---------------------------------------------------------------------------

def play(path):
    """Toca o .wav sem bloquear (um som novo interrompe o anterior)."""
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True
    player = shutil.which("aplay") or shutil.which("paplay")
    if player:
        subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    return False


class Voice:
    """Fila de falas do Wolf. speak() volta na hora; a síntese e o som rodam numa thread."""

    def __init__(self, enabled=None, printer=print, on_error=None):
        if enabled is None:
            enabled = os.environ.get("WOLF_VOICE", "1") != "0"
        self.enabled = enabled
        self.speaking_until = 0.0     # instante (time.monotonic) em que a fala atual termina
        self._print = printer
        self.on_error = on_error      # chamado uma vez com a mensagem se a voz falhar
        self._queue = queue.Queue()
        self._dir = tempfile.mkdtemp(prefix="wolf_voz_")
        atexit.register(shutil.rmtree, self._dir, True)
        self._count = 0
        self._warned = False
        self._voice_name = None
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
            # Se chegou fala nova antes da anterior sair, a anterior é descartada.
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put(text)

    def render(self, text):
        """Sintetiza + aplica efeitos. Retorna (caminho do .wav, duração em segundos)."""
        self._count += 1
        raw = os.path.join(self._dir, f"bruto_{self._count % 4}.wav")
        out = os.path.join(self._dir, f"wolf_{self._count % 4}.wav")
        name = synthesize(text, raw)
        if self._voice_name != name:
            self._voice_name = name
            self._print(f"[Wolf] voz: {name}")
        audio, rate = read_wav(raw)
        audio = robotize(audio, rate)
        write_wav(out, audio, rate)
        return out, len(audio) / rate

    def _loop(self):
        while True:
            text = self._queue.get()
            if not self.enabled:
                continue
            try:
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
# Autoteste: python voice.py ["texto"]
# ---------------------------------------------------------------------------

def self_test(text):
    """Testa cada etapa da voz e mostra onde parou."""
    import traceback

    print(f"Sistema: {sys.platform} | Python {sys.version.split()[0]}")
    step = "listar vozes"
    try:
        if sys.platform == "win32":
            print("PowerShell:", shutil.which("powershell") or "NÃO ENCONTRADO")
            result = run_powershell(_PS_LIST_VOICES)
            print("Vozes instaladas (nome | idioma | ativa):")
            print("  " + ("\n  ".join(result.stdout.strip().splitlines()) or "(nenhuma)"))
            if result.returncode != 0:
                print("  erro:", result.stderr.strip())
        else:
            print("Sintetizador:", shutil.which("espeak-ng") or shutil.which("espeak") or "NÃO ENCONTRADO")

        tmp = tempfile.mkdtemp(prefix="wolf_teste_")
        raw, out = os.path.join(tmp, "bruto.wav"), os.path.join(tmp, "wolf.wav")
        step = "gerar a fala"
        print("Voz usada:", synthesize(text, raw))
        step = "ler o áudio"
        audio, rate = read_wav(raw)
        print(f"Áudio gerado: {len(audio) / rate:.1f} s a {rate} Hz")
        step = "aplicar os efeitos"
        write_wav(out, robotize(audio, rate), rate)
        step = "tocar o som"
        print("Tocando a voz normal e depois a do Wolf...")
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(raw, winsound.SND_FILENAME)
            winsound.PlaySound(out, winsound.SND_FILENAME)
        elif not play(out):
            print("  (nenhum player de áudio encontrado)")
        print(f"OK. Arquivos em {tmp}")
    except Exception:  # noqa: BLE001
        print(f"\nFALHOU na etapa: {step}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    self_test(" ".join(sys.argv[1:]) or "Teste de voz. Aqui é o Wolf, pronto para a missão.")
