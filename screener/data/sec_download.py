"""Descarga de los Financial Statement Data Sets de la SEC.

La SEC publica cada trimestre un ZIP con los XBRL de todos los declarantes
estadounidenses. Lo que lo hace valioso frente a Yahoo es el campo `filed` de
`sub.txt`: la fecha real de presentación, no el cierre del periodo. Usar el
cierre adelanta al mercado unos 58 días de mediana, que es la trampa del §11
del `spec.md`.

Dos cosas que condicionan el diseño:

- **El XBRL se implantó por fases.** 2009 trae 22 declarantes, 2010 unos 500 y
  2011 unos 1.700; la cobertura completa (~9.000) no llega hasta 2012. Antes de
  esa fecha el panel sería una muestra sesgada hacia las mayores, así que el
  arranque por defecto es 2012.
- **`num.txt` pesa ~500 MB por trimestre.** Guardar los ZIP enteros son ~4 GB y
  parsearlos cada vez es inviable, así que `sec_parse` los reduce a un parquet
  con solo las etiquetas que el panel B necesita.

La SEC exige un User-Agent con **una dirección de correo real** —sin ella
devuelve 403— y limita a 10 peticiones por segundo. Ese correo se lee de
`SEC_CONTACT_EMAIL` en el `.env`, no va en el código: es dato personal y el
repositorio puede acabar siendo público.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

log = logging.getLogger("sec")

BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Antes de 2012 la cobertura es demasiado parcial para percentilar.
PRIMER_ANIO = 2012


def user_agent() -> str:
    """User-Agent exigido por la SEC: nombre y correo de contacto.

    Comprobado: sin correo devuelve 403, da igual lo descriptivo que sea el
    resto de la cadena.
    """
    correo = os.getenv("SEC_CONTACT_EMAIL", "").strip()
    if not correo or "@" not in correo:
        raise RuntimeError(
            "Falta SEC_CONTACT_EMAIL en el entorno. La SEC exige una dirección "
            "de correo real en el User-Agent y responde 403 sin ella. "
            "Añádela al .env: SEC_CONTACT_EMAIL=tu@correo.com"
        )
    return f"swing-screener research {correo}"


PAUSA = 0.5      # muy por debajo del límite de 10/s de la SEC
REINTENTOS = 3


def trimestres(desde: int = PRIMER_ANIO, hasta: date | None = None) -> list[str]:
    """Etiquetas `YYYYqN` desde `desde` hasta el trimestre completo más reciente."""
    hoy = hasta or date.today()
    salida = []
    for anio in range(desde, hoy.year + 1):
        for q in (1, 2, 3, 4):
            # El ZIP de un trimestre aparece unas semanas después de cerrarlo.
            fin = date(anio, q * 3, 28)
            if fin >= hoy:
                continue
            salida.append(f"{anio}q{q}")
    return salida


def _descargar(url: str, destino: Path) -> bool:
    """Un fichero, con reintentos. Devuelve False si la SEC no lo tiene."""
    for intento in range(1, REINTENTOS + 1):
        try:
            peticion = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            with urllib.request.urlopen(peticion, timeout=120) as respuesta:
                parcial = destino.with_suffix(destino.suffix + ".parcial")
                parcial.write_bytes(respuesta.read())
                # Renombrado atómico: si el proceso muere a media descarga, el
                # fichero definitivo no llega a existir y no se toma por bueno.
                parcial.rename(destino)
            return True
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return False
            log.warning("%s: HTTP %s (intento %d)", destino.name, err.code, intento)
        except Exception as err:  # noqa: BLE001 — red, cualquier cosa puede pasar
            log.warning("%s: %s (intento %d)", destino.name, err, intento)
        time.sleep(2 ** intento)
    return False


def sincronizar(directorio: Path, desde: int = PRIMER_ANIO) -> list[Path]:
    """Baja los ZIP que falten. Los ya presentes no se vuelven a pedir."""
    user_agent()      # falla pronto si no hay correo, en vez de en cada reintento
    directorio.mkdir(parents=True, exist_ok=True)
    presentes, nuevos = [], 0

    for etiqueta in trimestres(desde):
        destino = directorio / f"{etiqueta}.zip"
        if destino.exists() and destino.stat().st_size > 10_000:
            presentes.append(destino)
            continue
        log.info("descargando %s", etiqueta)
        if _descargar(f"{BASE}/{etiqueta}.zip", destino):
            presentes.append(destino)
            nuevos += 1
        else:
            log.warning("%s no disponible todavía", etiqueta)
        time.sleep(PAUSA)

    log.info("%d trimestres en disco (%d nuevos)", len(presentes), nuevos)
    return presentes


def tickers(directorio: Path, max_edad_dias: int = 30) -> Path:
    """El mapeo oficial CIK -> ticker. Sin él no se puede cruzar con los precios."""
    directorio.mkdir(parents=True, exist_ok=True)
    destino = directorio / "company_tickers.json"
    if destino.exists():
        edad = (time.time() - destino.stat().st_mtime) / 86400
        if edad < max_edad_dias:
            return destino
    _descargar(TICKERS_URL, destino)
    return destino


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    load_dotenv()      # SEC_CONTACT_EMAIL vive en el .env, como el resto
    raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "./.cache/sec")
    tickers(raiz)
    ficheros = sincronizar(raiz)
    total = sum(f.stat().st_size for f in ficheros) / 1e9
    print(f"\n{len(ficheros)} trimestres · {total:.1f} GB en {raiz}")
