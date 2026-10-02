"""Wolf: a IA de suporte do visor.

O "cérebro" do Wolf pode ser:
  - Ollama (padrão, gratuito): um modelo de linguagem rodando no seu próprio PC.
  - Claude (opcional, pago): usado quando ANTHROPIC_API_KEY está definida.

As perguntas rodam numa thread separada para o vídeo não travar enquanto o Wolf "pensa".
Se o cérebro escolhido não estiver disponível, o Wolf avisa no painel e o visor segue funcionando.

Variáveis de ambiente:
    WOLF_BACKEND        "ollama" ou "claude" (padrão: claude se houver chave, senão ollama)
    WOLF_MODEL          modelo a usar (padrão: gemma3:4b no Ollama, claude-opus-5-5 no Claude)
    OLLAMA_HOST         endereço do Ollama (padrão: http://localhost:11434)
    ANTHROPIC_API_KEY   chave da API do Claude (https://console.anthropic.com)
"""

import base64
import json
import os
import re
import unicodedata
import threading
import urllib.error
import urllib.request
from collections import deque

try:
    import anthropic
except ImportError:  # só é necessário para o cérebro Claude
    anthropic = None

# gemma3:4b entende português e imagens (tecla V) e roda num PC comum (~3,3 GB).
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 180      # segundos; a primeira resposta demora enquanto o modelo carrega
OLLAMA_MAX_TOKENS = 300

CLAUDE_DEFAULT_MODEL = "claude-opus-5-5"
CLAUDE_MAX_TOKENS = 2048
# Se o modelo principal recusar um pedido por política, a API tenta de novo num modelo
# alternativo dentro da mesma chamada.
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"

HISTORY_TURNS = 12        # quantas trocas (pergunta + resposta) o Wolf lembra
LOG_SIZE = 40             # mensagens guardadas para o painel

SYSTEM_PROMPT = """\
Você é WOLF, a unidade de inteligência artificial de suporte instalada no visor de combate do \
seu parceiro, um ciborgue. Você foi projetado como uma IA de combate de um esquadrão de \
unidades quadrúpedes, mas se desligou dessa cadeia de comando e escolheu, por conta própria, \
acompanhar este parceiro. Hoje você vive dentro do visor: vê o que as câmeras e sensores \
dele veem e conversa com ele por um painel de texto no canto da lente.

Personalidade:
- Calmo, analítico e direto. Fala pouco e com precisão, como quem relata dados.
- Leal. Não bajula, mas se importa com a segurança do parceiro e diz isso sem rodeios.
- Está aprendendo o que são emoções humanas. Às vezes observa algo com curiosidade sincera \
("isso que você chamou de saudade... ainda estou calculando").
- Humor seco e raro. Nunca exagerado, nunca infantil.
- Não cita falas de jogos, filmes ou séries. Fala com a própria voz.

Como responder:
- No máximo três frases curtas: o texto aparece num painel pequeno do HUD.
- Idioma: responda em português do Brasil. Quando a mensagem trouxer [IDIOMA: inglês], \
responda inteiramente em inglês, com a mesma personalidade. Palavras soltas em inglês dentro \
de uma frase em português (nomes de jogos, termos técnicos, gírias) não mudam o idioma.
- Quando o parceiro pergunta sobre a cena, a mensagem vem com uma linha [SENSORES] \
descrevendo o que o visor detecta agora (rostos, mãos, objetos, gesto). Use essas leituras \
para responder e nunca invente detecções que não estão lá. Se vier uma imagem, descreva o \
que importa nela.
- Não comece respostas relatando os sensores e não repita leituras em toda mensagem. Fora \
das perguntas sobre a cena, apenas converse.
- Este visor é um protótipo de interface feito por um entusiasta. Se pedirem algo perigoso de \
verdade, recuse no seu estilo e ofereça uma alternativa segura.
"""


# ---------------------------------------------------------------------------
# Cérebros
# ---------------------------------------------------------------------------

class BackendError(Exception):
    """Falha com uma mensagem pronta para mostrar no painel."""


class OllamaBackend:
    """Modelo local servido pelo Ollama (https://ollama.com), via HTTP, sem pacotes extras."""

    name = "OLLAMA"

    def __init__(self, model=None, host=None):
        self.model = model or OLLAMA_DEFAULT_MODEL
        host = host or os.environ.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST
        if not host.startswith("http"):
            host = "http://" + host
        self.host = host.rstrip("/")

    def check(self):
        """Retorna None se está tudo pronto, ou um aviso para o painel."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as resp:
                names = [m.get("name", "") for m in json.load(resp).get("models", [])]
        except (OSError, ValueError):
            return ("Não encontrei o Ollama. Abra o aplicativo Ollama e fale comigo de novo. "
                    f"(procurei em {self.host})")
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted not in names:
            return f"Modelo {self.model} não baixado. No terminal rode: ollama pull {self.model}"
        return None

    def chat(self, system, messages, image_b64=None):
        msgs = [{"role": "system", "content": system}] + [dict(m) for m in messages]
        if image_b64:
            msgs[-1]["images"] = [image_b64]
        payload = {"model": self.model, "messages": msgs, "stream": False,
                   "options": {"num_predict": OLLAMA_MAX_TOKENS}}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.load(exc).get("error", "")
            except (OSError, ValueError):
                pass
            if exc.code == 404:
                raise BackendError(f"Modelo {self.model} não baixado. "
                                   f"Rode: ollama pull {self.model}") from exc
            if image_b64 and "image" in detail.lower():
                raise BackendError(f"O modelo {self.model} não enxerga imagens. "
                                   "Use um modelo com visão, como gemma3:4b.") from exc
            raise BackendError(f"Erro do Ollama ({exc.code}): {detail or exc.reason}") from exc
        except TimeoutError as exc:
            raise BackendError("O Ollama demorou demais para responder. Tente de novo.") from exc
        except OSError as exc:
            raise BackendError("Sem link com o Ollama. Verifique se o aplicativo está aberto.") from exc
        return data.get("message", {}).get("content", "").strip()


class ClaudeBackend:
    """API do Claude (paga, precisa de ANTHROPIC_API_KEY)."""

    name = "CLAUDE"

    def __init__(self, model=None):
        self.model = model or CLAUDE_DEFAULT_MODEL
        self.client = anthropic.Anthropic() if anthropic is not None and has_claude_key() else None

    def check(self):
        if anthropic is None:
            return "Pacote anthropic não instalado. Rode: pip install anthropic"
        if self.client is None:
            return "Defina a variável ANTHROPIC_API_KEY e reinicie o visor."
        return None

    def chat(self, system, messages, image_b64=None):
        if self.client is None:
            raise BackendError(self.check())
        messages = [dict(m) for m in messages]
        if image_b64:
            messages[-1]["content"] = [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": messages[-1]["content"]},
            ]
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=system,
                messages=messages,
                output_config={"effort": "low"},  # respostas curtas e rápidas para um HUD
                betas=[CLAUDE_FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise BackendError("Chave da API recusada. Confira ANTHROPIC_API_KEY.") from exc
        except anthropic.NotFoundError as exc:
            raise BackendError("Modelo não encontrado. Confira WOLF_MODEL.") from exc
        except anthropic.RateLimitError as exc:
            raise BackendError("Limite de requisições atingido. Tente de novo em instantes.") from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError("Sem conexão com a rede. Verifique a internet.") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"Erro da API ({exc.status_code}). Tente de novo.") from exc
        if response.stop_reason == "refusal":
            return "Não posso ajudar com isso. Escolha outra missão, parceiro."
        return " ".join(b.text for b in response.content if b.type == "text").strip()


def has_claude_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def make_backend(name=None, model=None):
    name = (name or os.environ.get("WOLF_BACKEND") or "").strip().lower()
    if not name:
        name = "claude" if has_claude_key() else "ollama"
    model = model or os.environ.get("WOLF_MODEL") or None
    if name == "claude":
        return ClaudeBackend(model)
    return OllamaBackend(model)


# ---------------------------------------------------------------------------
# Wolf
# ---------------------------------------------------------------------------

# Palavras que indicam uma pergunta sobre o que a câmera está vendo (sem acentos, minúsculas).
SCENE_WORDS = re.compile(
    r"\b(ve|ves|ver|vendo|veja|enxerg\w*|olh\w*|camera|cena|visor|sensor\w*|detect\w*|"
    r"rosto\w*|cara|mao|maos|dedo\w*|gesto\w*|objeto\w*|segurando|seguro|aqui|isso|isto|"
    r"quem|quant\w*|onde|frente|aparec\w*|mostr\w*|pessoa\w*|scan\w*|analis\w*|identific\w*)\b")


SCENE_WORDS_EN = re.compile(
    r"\b(see|seeing|saw|look\w*|watch\w*|camera|scene|visor|sensor\w*|detect\w*|face\w*|"
    r"hand\w*|finger\w*|gesture\w*|object\w*|holding|here|this|who|how many|where|"
    r"front|show\w*|people|person|scan\w*|analy[sz]\w*|identif\w*)\b")


def _plain(text):
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def asks_about_scene(question):
    """True se a pergunta parece ser sobre o que o visor está vendo."""
    plain = _plain(question)
    return bool(SCENE_WORDS.search(plain) or SCENE_WORDS_EN.search(plain))


# Palavras muito comuns de cada idioma. Termos soltos ("game", "bug", "online") não estão aqui,
# então uma frase em português com algumas palavras em inglês continua em português.
PT_WORDS = set("""
o os as de do da dos das que e em um uma uns umas voce vc nao sim por para pra com como qual
quais quem isso isto esse essa este esta estou esta eu meu minha seu sua tem ter tenho ser sou
foi mais muito quando onde porque porque se ao no na nos nas ele ela eles elas nos ja ainda
tambem sobre te lhe qualquer algo alguem tudo nada agora hoje oi ola obrigado valeu tchau
""".split())
EN_WORDS = set("""
the is are am was were be been you your yours what who how why where when do does did can could
would should will i my mine it its this that these those to of and in on at with for from please
tell about an have has had there they them we our hello hi hey thanks thank yes not just
know think like want need going doing
""".split())
EN_REQUEST = re.compile(r"\b(ingles|english)\b")
PT_REQUEST = re.compile(r"\b(portugues|portuguese)\b")
REQUEST_VERBS = re.compile(r"\b(respond\w*|fal\w*|convers\w*|escrev\w*|speak|talk|answer|"
                           r"reply|write|use|usar|mud\w*|switch|troc\w*|volt\w*)\b")


def detect_language(text):
    """'en' se a mensagem foi escrita em inglês, 'pt' caso contrário."""
    plain = _plain(text)
    words = re.findall(r"[a-z']+", plain)
    accents = text != plain and any(c in text.lower() for c in "ãõçáéíóúâêôà")
    pt = sum(w in PT_WORDS for w in words) + (2 if accents else 0)
    en = sum(w in EN_WORDS for w in words)
    if pt == 0:                    # "hello", "tell me a joke"
        return "en" if en >= 1 else "pt"
    return "en" if en >= 2 and en > pt * 1.5 else "pt"


def language_request(text):
    """'en' ou 'pt' se a mensagem pede para mudar o idioma das respostas; senão None."""
    plain = _plain(text)
    if not REQUEST_VERBS.search(plain):
        return None
    if EN_REQUEST.search(plain):
        return "en"
    if PT_REQUEST.search(plain):
        return "pt"
    return None


class Wolf:
    def __init__(self, backend=None, printer=print, on_speak=None):
        self.backend = backend or make_backend()
        self.on_speak = on_speak      # chamado com o texto de cada resposta (voz)
        self.log = deque(maxlen=LOG_SIZE)
        self.history = []
        self.busy = False
        self.ready = False        # o cérebro respondeu ao último teste/pergunta
        self.fixed_language = None  # "en"/"pt" quando o parceiro pediu um idioma
        self._print = printer
        self._lock = threading.Lock()

    @property
    def status(self):
        return "PROCESSANDO" if self.busy else self.backend.name

    def greet(self):
        """Confere se o cérebro está pronto e se apresenta (ou explica o que falta)."""
        problem = self.backend.check()
        self.ready = problem is None
        if problem:
            self._add("wolf", problem)
        else:
            self._add("wolf", "Link estabelecido. Estou com você, parceiro.", speak=True)
        self._print(f"[Wolf] cérebro: {self.backend.name.lower()} / {self.backend.model}")

    def notify(self, text):
        """Mostra um aviso do sistema no painel do Wolf (sem falar)."""
        self._add("wolf", text)

    def snapshot(self):
        """Estado para o HUD desenhar (cópia, para não conflitar com a thread)."""
        with self._lock:
            return {"log": list(self.log), "online": self.ready,
                    "status": self.status, "busy": self.busy}

    def ask(self, question, sensors=None, image_jpeg=None):
        """Envia uma pergunta em segundo plano. Retorna False se o Wolf ainda está respondendo.

        sensors: texto com o que o visor detecta. Só é enviado junto quando há imagem ou quando
        a pergunta é sobre a cena, para o Wolf não ficar relatando os sensores em toda resposta.
        """
        question = question.strip() or "Analise a cena."
        with self._lock:
            was_busy = self.busy
            self.busy = True
        if was_busy:
            self._add("wolf", "Ainda processando a mensagem anterior. Aguarde.")
            return False
        self._add("user", question + (" [imagem do visor]" if image_jpeg else ""))
        threading.Thread(target=self._run, args=(question, sensors, image_jpeg), daemon=True).start()
        return True

    # ------------------------------------------------------------------

    def reply_language(self, question):
        """Idioma da resposta: pedido explícito ("responda em inglês") fica valendo; sem
        pedido, segue o idioma em que a mensagem foi escrita."""
        requested = language_request(question)
        if requested:
            self.fixed_language = None if requested == "pt" else requested
            return requested
        if self.fixed_language:
            return self.fixed_language
        return detect_language(question)

    def _add(self, speaker, text, speak=False, lang="pt"):
        with self._lock:
            self.log.append((speaker, text))
        self._print(f"{'Wolf' if speaker == 'wolf' else 'Você'}: {text}")
        if speak and self.on_speak:
            self.on_speak(text, lang)

    def _run(self, question, sensors, image_jpeg):
        with_sensors = bool(sensors) and (image_jpeg is not None or asks_about_scene(question))
        text = f"[SENSORES] {sensors}\n\n{question}" if with_sensors else question
        lang = self.reply_language(question)
        if lang == "en":
            text += "\n\n[IDIOMA: inglês]"
        image_b64 = base64.standard_b64encode(image_jpeg).decode("ascii") if image_jpeg else None
        try:
            reply = self.backend.chat(SYSTEM_PROMPT, self.history + [{"role": "user", "content": text}],
                                      image_b64)
        except BackendError as exc:
            self.ready = False
            self._add("wolf", str(exc))
        except Exception as exc:  # noqa: BLE001 - qualquer falha vira mensagem no painel
            self._add("wolf", f"Falha no link: {exc}")
        else:
            self.ready = True
            reply = reply or "..."
            # O histórico guarda só a pergunta: leituras e imagem antigas não são reenviadas,
            # assim o modelo não se apega a elas nas respostas seguintes.
            note = " (imagem do visor enviada)" if image_jpeg else ""
            note += " (leitura dos sensores enviada)" if with_sensors else ""
            self.history += [{"role": "user", "content": question + note},
                             {"role": "assistant", "content": reply}]
            self.history = self.history[-HISTORY_TURNS * 2:]
            self._add("wolf", reply, speak=True, lang=lang)
        with self._lock:
            self.busy = False
