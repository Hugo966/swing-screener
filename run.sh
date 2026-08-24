#!/usr/bin/env bash
# Envoltorio para cron: una region por invocacion.
#
#     ./run.sh us
#
# Pensado para una VM pequena. Tres cosas que no son obvias y por las que existe
# este fichero en vez de llamar al runner directamente desde el crontab:
#
#   - un solo proceso por region a la vez,
#   - el log acotado, porque cuatro corridas diarias durante meses lo engordan,
#   - aviso por Telegram si falla: un cron que muere en silencio no se detecta
#     hasta que echas de menos las alertas tres semanas despues.
set -uo pipefail

REGION="${1:?uso: run.sh <region>}"
# Se resuelve desde la ubicacion del script, no con una ruta fija: asi el mismo
# fichero sirve en cualquier maquina sin editarlo.
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$BASE/logs/${REGION}.log"
PY="$BASE/.venv/bin/python"

mkdir -p "$BASE/logs"
cd "$BASE" || exit 1
[ -x "$PY" ] || { echo "no hay entorno virtual en $BASE/.venv" >&2; exit 1; }

# Un proceso por region: emergentes tarda ~45 min en frio, y sin esto el cron
# del dia siguiente arrancaria encima del anterior.
exec 9>"/tmp/swing-${REGION}.lock"
flock -n 9 || { echo "$(date -Is) ya en marcha, se salta" >>"$LOG"; exit 0; }

echo "=== $(date -Is) inicio $REGION ===" >>"$LOG"
"$PY" -m screener.runner --region "$REGION" >>"$LOG" 2>&1
CODE=$?
echo "=== $(date -Is) fin $REGION codigo=$CODE ===" >>"$LOG"

if [ "$CODE" -ne 0 ]; then
    # El .env es opcional —sin token el screener imprime por consola— asi que
    # el propio aviso de fallo no debe romperse si no hay credenciales.
    [ -f "$BASE/.env" ] && { set -a; . "$BASE/.env"; set +a; }
    if [ -n "${TELEGRAM_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -sS --max-time 20 -X POST \
            "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            --data-urlencode "text=swing-screener: ${REGION} fallo (codigo ${CODE})
$(tail -15 "$LOG")" >/dev/null || true
    fi
fi

if [ "$(wc -l <"$LOG")" -gt 5000 ]; then
    tail -n 3000 "$LOG" >"$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
exit "$CODE"
