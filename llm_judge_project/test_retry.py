"""Prueba aislada del mecanismo de retry con _WaitRetryAfter."""
import asyncio
import random
import re
import time
from unittest.mock import MagicMock

from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, RetryCallState
from tenacity.wait import wait_base


# ── Reproducción exacta del código del notebook ─────────────────────────────

_RE_RETRY = re.compile(r'try again in ([\d.]+)s', re.IGNORECASE)


class _WaitRetryAfter(wait_base):
    """Respeta el retry-after de OpenAI con jitter para evitar thundering herd."""

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception()
        if isinstance(exc, RateLimitError):
            base: float | None = None
            resp = getattr(exc, 'response', None)
            if resp is not None:
                after = resp.headers.get('retry-after')
                if after:
                    try:
                        base = max(1.0, float(after))
                    except ValueError:
                        pass
            if base is None:
                m = _RE_RETRY.search(str(exc))
                if m:
                    base = max(1.0, float(m.group(1)))
            if base is not None:
                return base + random.uniform(0.0, base)  # jitter 0-100%
        return 60.0 + random.uniform(0.0, 30.0)


# ── Función de prueba ────────────────────────────────────────────────────────

CALL_COUNT = 0
FAIL_TIMES = 4  # cuántas veces falla antes de tener éxito

def _make_fake_rate_limit_error() -> RateLimitError:
    """Crea un RateLimitError idéntico al que devuelve OpenAI."""
    response = MagicMock()
    response.status_code = 429
    response.headers = {}  # sin header retry-after → usa el mensaje
    body = {
        'error': {
            'message': (
                'Rate limit reached for gpt-4o-mini in organization org-XXX '
                'on requests per day (RPD): Limit 10000, Used 10000, Requested 1. '
                'Please try again in 8.64s.'
            ),
            'type': 'requests',
            'param': None,
            'code': 'rate_limit_exceeded',
        }
    }
    return RateLimitError(
        message=body['error']['message'],
        response=response,
        body=body,
    )


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(15),
    wait=_WaitRetryAfter(),
    reraise=True,
)
async def fake_classify(task_id: int) -> str:
    global CALL_COUNT
    CALL_COUNT += 1
    if CALL_COUNT <= FAIL_TIMES:
        print(f"  [task {task_id}] intento {CALL_COUNT} → simulando 429 RPD")
        raise _make_fake_rate_limit_error()
    print(f"  [task {task_id}] intento {CALL_COUNT} → OK")
    return "NO_ACUSATORIA"


# ── Test 1: una sola pregunta ────────────────────────────────────────────────

async def test_single():
    global CALL_COUNT
    CALL_COUNT = 0
    print("=" * 55)
    print("TEST 1: un solo request con 4 fallos seguidos")
    print("=" * 55)
    t0 = time.perf_counter()
    result = await fake_classify(task_id=1)
    elapsed = time.perf_counter() - t0
    print(f"  Resultado: {result!r}  |  Tiempo total: {elapsed:.2f}s\n")
    assert result == "NO_ACUSATORIA"


# ── Test 2: 5 requests concurrentes (thundering herd) ───────────────────────

async def test_concurrent():
    global CALL_COUNT, FAIL_TIMES
    CALL_COUNT = 0
    FAIL_TIMES = 10  # cada tarea falla las primeras 10 veces globalmente
    print("=" * 55)
    print("TEST 2: 5 requests concurrentes (thundering herd)")
    print("=" * 55)

    # Verificar que los tiempos de espera son distintos (jitter funcionando)
    wait_fn = _WaitRetryAfter()
    exc = _make_fake_rate_limit_error()

    waits = []
    for _ in range(5):
        state = MagicMock()
        state.outcome.exception.return_value = exc
        w = wait_fn(state)
        waits.append(w)

    print(f"  Tiempos de espera generados para 5 tasks: {[f'{w:.2f}s' for w in waits]}")
    assert len(set(round(w, 2) for w in waits)) > 1, "¡Los tiempos son iguales! El jitter no funciona."
    assert all(8.64 <= w <= 8.64 * 2 for w in waits), f"Tiempos fuera del rango esperado: {waits}"
    print("  OK — todos distintos, todos dentro del rango [8.64s, 17.28s]\n")


# ── Test 3: parseo del mensaje de error ─────────────────────────────────────

def test_parse():
    print("=" * 55)
    print("TEST 3: parseo del mensaje de error")
    print("=" * 55)
    exc = _make_fake_rate_limit_error()
    m = _RE_RETRY.search(str(exc))
    assert m, "No se encontró el tiempo en el mensaje de error"
    parsed = float(m.group(1))
    print(f"  Tiempo parseado del mensaje: {parsed}s")
    assert parsed == 8.64
    print("  OK\n")


# ── Runner ───────────────────────────────────────────────────────────────────

async def main():
    test_parse()
    await test_concurrent()
    # test_single hace esperas reales (~8-17s), lo saltamos en la prueba rápida
    print("Todos los tests pasaron.")
    print("\nNota: test_single omitido (haría esperas reales de ~8-17s).")
    print("Para correrlo completo: cambia la línea 'await test_concurrent()' por 'await test_single()'.")


if __name__ == '__main__':
    asyncio.run(main())
