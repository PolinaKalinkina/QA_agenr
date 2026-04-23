"""HTTP-клиент для GigaChat через официальный SDK (OAuth2-credentials).

Особенности:
- Аутентификация через credentials (base64 'client_id:client_secret') + scope.
- SDK сам обновляет access-токен по истечении.
- Экспоненциальный backoff на сетевых ошибках / 5xx.
- Подсчёт токенов и latency на каждый вызов.
- Thread-safe: SDK-клиент переиспользуется, можно дёргать из ThreadPoolExecutor.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import urllib3
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import httpx
    from gigachat import GigaChat
    from gigachat.exceptions import ResponseError
    from gigachat.models import Chat, Messages, MessagesRole
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Требуется пакет gigachat. Установите: pip install gigachat"
    ) from exc


@dataclass
class LLMResponse:
    """Результат одного вызова модели."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_sec: float
    raw: dict[str, Any]

    @property
    def is_empty(self) -> bool:
        return not self.content or not self.content.strip()


class GigaChatError(Exception):
    """Ошибка взаимодействия с GigaChat API."""


_RETRYABLE_EXC = (httpx.RequestError, httpx.RemoteProtocolError, GigaChatError)


class GigaChatClient:
    """Клиент GigaChat через SDK с OAuth2-credentials."""

    def __init__(
        self,
        credentials: str,
        model: str,
        scope: str = "GIGACHAT_API_CORP",
        verify_ssl: bool = False,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
    ) -> None:
        if not credentials:
            raise ValueError("credentials пуст — задайте GIGACHAT_CREDENTIALS в .env")

        self.model = model
        self.scope = scope
        self.verify_ssl = verify_ssl
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base

        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            warnings.filterwarnings(
                "ignore", category=urllib3.exceptions.InsecureRequestWarning
            )
            logger.warning(
                "SSL verification disabled (verify_ssl=false). "
                "Допустимо только во внутреннем контуре Сбера."
            )

        self._llm = GigaChat(
            credentials=credentials,
            scope=scope,
            model=model,
            verify_ssl_certs=verify_ssl,
            timeout=timeout_seconds,
        )

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        system_message: str | None = None,
    ) -> LLMResponse:
        """Один вызов модели. Возвращает LLMResponse или бросает GigaChatError."""
        messages: list[Messages] = []
        if system_message:
            messages.append(Messages(role=MessagesRole.SYSTEM, content=system_message))
        messages.append(Messages(role=MessagesRole.USER, content=prompt))

        payload = Chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model,
        )

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_backoff_base, min=1, max=20),
            retry=retry_if_exception_type(_RETRYABLE_EXC),
            reraise=True,
        )
        def _do_request() -> LLMResponse:
            started = time.perf_counter()
            try:
                response = self._llm.chat(payload)
            except ResponseError as exc:
                status = getattr(exc, "status_code", None) or _extract_status(exc)
                text = str(exc)[:500]
                if status is not None and status >= 500:
                    msg = f"GigaChat 5xx: {status} {text}"
                    logger.warning(msg)
                    raise GigaChatError(msg) from exc
                msg = f"GigaChat {status or 'error'}: {text}"
                logger.error(msg)
                raise GigaChatError(msg) from exc
            except httpx.RequestError as exc:
                logger.warning(f"GigaChat network error: {exc!r}")
                raise

            elapsed = time.perf_counter() - started

            try:
                choice = response.choices[0]
                content = choice.message.content or ""
            except (IndexError, AttributeError) as exc:
                raise GigaChatError(f"Неожиданная структура ответа: {response!r}") from exc

            usage = response.usage
            return LLMResponse(
                content=content,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                latency_sec=elapsed,
                raw=response.dict() if hasattr(response, "dict") else {},
            )

        try:
            return _do_request()
        except GigaChatError:
            raise
        except Exception as exc:
            raise GigaChatError(f"GigaChat call failed after retries: {exc!r}") from exc

    def close(self) -> None:
        try:
            self._llm.close()
        except Exception:
            pass

    def __enter__(self) -> "GigaChatClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _extract_status(exc: Exception) -> int | None:
    """Пытаемся вытащить HTTP-код из ResponseError, если он не в атрибуте."""
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None
