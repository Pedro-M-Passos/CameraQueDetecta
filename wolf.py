"""Wolf: a IA de suporte do visor, conversando pela API do Claude.

As perguntas rodam numa thread separada para o vídeo não travar enquanto o Wolf "pensa".
Sem ANTHROPIC_API_KEY (ou sem o pacote anthropic) o Wolf fica offline e o visor segue funcionando.

Variáveis de ambiente:
    ANTHROPIC_API_KEY   chave da API (https://console.anthropic.com)
    WOLF_MODEL          modelo a usar (padrão: claude-opus-5-5)
"""

import base64
import os
import threading
from collections import deque

try:
    import anthropic
except ImportError:  # o visor roda mesmo sem o pacote
    anthropic = None

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 2048
HISTORY_TURNS = 12        # quantas trocas (pergunta + resposta) o Wolf lembra
LOG_SIZE = 40             # mensagens guardadas para o painel

# Se o modelo principal recusar um pedido por política, a API tenta de novo num modelo
# alternativo dentro da mesma chamada.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

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
- Português do Brasil, no máximo três frases curtas: o texto aparece num painel pequeno do HUD.
- Cada mensagem do parceiro vem com uma linha [SENSORES] descrevendo o que o visor detecta \
agora (rostos, mãos, objetos, gesto). Use essas leituras quando forem relevantes e nunca \
invente detecções que não estão lá. Se vier uma imagem, descreva o que importa nela.
- Este visor é um protótipo de interface feito por um entusiasta. Se pedirem algo perigoso de \
verdade, recuse no seu estilo e ofereça uma alternativa segura.
"""

OFFLINE_HELP = ("Link offline. Defina a variável ANTHROPIC_API_KEY e reinicie o visor "
                "para eu poder responder.")


class Wolf:
    def __init__(self, model=None, printer=print):
        self.model = model or os.environ.get("WOLF_MODEL", DEFAULT_MODEL)
        self.log = deque(maxlen=LOG_SIZE)
        self.history = []
        self.busy = False
        self._print = printer
        self._lock = threading.Lock()
        self.client = None
        self.offline_reason = None

        if anthropic is None:
            self.offline_reason = "pacote anthropic não instalado (pip install anthropic)"
        elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            self.offline_reason = "ANTHROPIC_API_KEY não definida"
        else:
            self.client = anthropic.Anthropic()

    @property
    def online(self):
        return self.client is not None

    @property
    def status(self):
        if self.busy:
            return "PROCESSANDO"
        return "ONLINE" if self.online else "OFFLINE"

    def greet(self):
        if self.online:
            self._add("wolf", "Link estabelecido. Sensores do visor sincronizados. Estou com você.")
        else:
            self._add("wolf", OFFLINE_HELP)
            self._print(f"[Wolf] offline: {self.offline_reason}")

    def snapshot(self):
        """Estado para o HUD desenhar (cópia, para não conflitar com a thread)."""
        with self._lock:
            return {"log": list(self.log), "online": self.online,
                    "status": self.status, "busy": self.busy}

    def ask(self, question, sensors, image_jpeg=None):
        """Envia uma pergunta em segundo plano. Retorna False se o Wolf ainda está respondendo."""
        question = question.strip() or "Analise a cena."
        with self._lock:
            was_busy = self.busy
            self.busy = self.busy or self.online
        if was_busy:
            self._add("wolf", "Ainda processando a mensagem anterior. Aguarde.")
            return False
        self._add("user", question + (" [imagem do visor]" if image_jpeg else ""))
        if not self.online:
            self._add("wolf", OFFLINE_HELP)
            return True
        threading.Thread(target=self._run, args=(question, sensors, image_jpeg), daemon=True).start()
        return True

    # ------------------------------------------------------------------

    def _add(self, speaker, text):
        with self._lock:
            self.log.append((speaker, text))
        self._print(f"{'Wolf' if speaker == 'wolf' else 'Você'}: {text}")

    def _run(self, question, sensors, image_jpeg):
        text_part = f"[SENSORES] {sensors}\n\n{question}"
        content = []
        if image_jpeg is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.standard_b64encode(image_jpeg).decode("ascii")},
            })
        content.append({"type": "text", "text": text_part})

        try:
            reply = self._request(self.history + [{"role": "user", "content": content}])
        except Exception as exc:  # noqa: BLE001 - qualquer falha vira mensagem no painel
            reply = None
            self._add("wolf", describe_error(exc))

        if reply is not None:
            # O histórico guarda só texto: a imagem não é reenviada a cada pergunta.
            note = " (imagem do visor enviada)" if image_jpeg is not None else ""
            self.history += [{"role": "user", "content": text_part + note},
                             {"role": "assistant", "content": reply}]
            self.history = self.history[-HISTORY_TURNS * 2:]
            self._add("wolf", reply)

        with self._lock:
            self.busy = False

    def _request(self, messages):
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": "low"},  # respostas curtas e rápidas para um HUD
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "Não posso ajudar com isso. Escolha outra missão, parceiro."
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        return text or "..."


def describe_error(exc):
    if anthropic is not None:
        if isinstance(exc, anthropic.AuthenticationError):
            return "Chave da API recusada. Confira ANTHROPIC_API_KEY."
        if isinstance(exc, anthropic.NotFoundError):
            return "Modelo não encontrado. Confira WOLF_MODEL."
        if isinstance(exc, anthropic.RateLimitError):
            return "Limite de requisições atingido. Tente de novo em instantes."
        if isinstance(exc, anthropic.APIConnectionError):
            return "Sem conexão com a rede. Verifique a internet."
        if isinstance(exc, anthropic.APIStatusError):
            return f"Erro da API ({exc.status_code}). Tente de novo."
    return f"Falha no link: {exc}"
