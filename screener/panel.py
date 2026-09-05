"""Panel de Streamlit sobre las corridas del screener.

    streamlit run screener/panel.py

Lee dos fuentes, ambas producidas por `runner.py`:

- `state.sqlite` — KPIs y régimen por día (`runs`), ranking por día (`snapshots`)
  e histórico de alertas con su tipo (`alerts`).
- `out/<region>_<fecha>.csv` — el desglose métrica a métrica de esa corrida.

El gráfico central es el de dispersión A_pct vs B_pct: la regla del §7 es un AND
de dos umbrales, así que en el plano se ve como un cuadrante. Es la forma de
comprobar de un vistazo si el corte está donde quieres.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# `streamlit run screener/panel.py` pone screener/ en sys.path, no la raíz del
# proyecto, así que sin esto el import del propio paquete falla.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from screener.config import load_config  # noqa: E402
from screener.metrics import REGISTRY  # noqa: E402
from screener.state import AlertState  # noqa: E402

# --- Paleta (referencia validada; ambos modos) ------------------------------
PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series_a": "#2a78d6",
        "series_b": "#eb6834",
        "inactive": "#c3c2b7",
        # Podio: el color va por puesto (1º amarillo, 2º verde, 3º azul) y es el
        # mismo en los tres podios. La tinta de cada bloque se eligió midiendo
        # contraste, no a ojo: blanco sobre el amarillo da 2,11:1 e ilegible.
        # `medal` es el disco: oro, plata y bronce. Necesita anillo sí o sí — el
        # oro sobre el bloque amarillo da 1,29:1 y se fundiría con él. Un anillo
        # oscuro contrasta con los tres metales y con los tres fondos.
        "podium": [
            {"bg": "#ffc107", "ink": "#0b0b0b",
             "medal": "#d4af37", "sheen": ("#f7e08a", "#a67c1a")},
            {"bg": "#008300", "ink": "#fcfcfb",
             "medal": "#c0c0c0", "sheen": ("#f0f0f0", "#8c8c8c")},
            {"bg": "#2a78d6", "ink": "#fcfcfb",
             "medal": "#cd7f32", "sheen": ("#e8ab77", "#b07138")},
        ],
        "featured_bg": "#f4f3ee",
        "featured_border": "#dedcd2",
        # El amarillo del podio (#ffc107) es demasiado claro para una marca
        # suelta sobre fondo claro: 1,59:1. Para resaltar un punto se usa el
        # amarillo categórico, más oscuro, y además anillo y etiqueta.
        "highlight": "#eda100",
        "zone_buy": "rgba(12, 163, 12, 0.09)",
        "zone_avoid": "rgba(208, 59, 59, 0.11)",
        # Estado de frescura del panel. Sobre `surface` claro (#fcfcfb) dan
        # 4,9:1 y 5,2:1: legibles como texto, no solo como relleno. Los rgba de
        # `zone_*` no sirven aquí porque son translúcidos y esto es tipografía.
        "fresh": "#008300",
        "stale": "#c02626",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series_a": "#3987e5",
        "series_b": "#d95926",
        "inactive": "#4a4a47",
        "podium": [
            {"bg": "#ffc107", "ink": "#0b0b0b",
             "medal": "#d4af37", "sheen": ("#f7e08a", "#a67c1a")},
            {"bg": "#008300", "ink": "#ffffff",
             "medal": "#c0c0c0", "sheen": ("#f0f0f0", "#8c8c8c")},
            {"bg": "#3987e5", "ink": "#ffffff",
             "medal": "#cd7f32", "sheen": ("#e8ab77", "#b07138")},
        ],
        "featured_bg": "#232322",
        "featured_border": "#3a3a37",
        "highlight": "#eda100",
        "zone_buy": "rgba(12, 163, 12, 0.16)",
        "zone_avoid": "rgba(208, 59, 59, 0.20)",
        # Sobre `surface` oscuro (#1a1a19) hacen falta tonos más claros que en
        # el tema claro: estos dan 6,1:1 y 5,4:1.
        "fresh": "#4caf50",
        "stale": "#ef5350",
    },
}


def palette() -> dict[str, str]:
    """Paleta del tema activo del visor; light si Streamlit no lo expone."""
    mode = "light"
    try:
        mode = st.context.theme.type or "light"
    except Exception:  # noqa: BLE001 — versiones sin st.context.theme
        pass
    return PALETTES.get(mode, PALETTES["light"])


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_resource
def get_config():
    return load_config(ROOT / "config.yaml", load_env=False)


@st.cache_resource
def get_state(path: str) -> AlertState:
    return AlertState(path)


@st.cache_data(ttl=60)
def load_runs(path: str, region: str) -> pd.DataFrame:
    rows = get_state(path).runs(region)
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=60)
def load_snapshots(path: str, region: str) -> pd.DataFrame:
    rows = get_state(path).history(region)
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=60)
def load_alerts(path: str, region: str, days: int) -> pd.DataFrame:
    rows = get_state(path).recent(region, days=days)
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=60)
def load_detail(output_dir: str, region: str, day: str) -> pd.DataFrame:
    """CSV de la corrida: es el único sitio con el desglose por métrica."""
    path = Path(output_dir) / f"{region}_{day}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["rank"] = frame["score_final"].rank(ascending=False, method="min").astype(int)
    return frame


# ---------------------------------------------------------------------------
# Recálculo
# ---------------------------------------------------------------------------
def run_region_now(region: str, *, send_alerts: bool) -> int:
    """Lanza el runner y va mostrando su log.

    Se ejecuta como subproceso en vez de llamar a `run_region` en el propio hilo
    de Streamlit por dos razones: el motor usa un pool de hilos propio, y así el
    log sale línea a línea en vez de aparecer entero al final.

    En frío una corrida tarda bastante (descarga precios y fundamentales de todo
    el universo); en caliente va contra la caché y es cuestión de segundos.
    """
    command = [sys.executable, "-m", "screener.runner", "--region", region, "--force"]
    if not send_alerts:
        command.append("--dry-run")

    with st.status(f"Recalculando {region}…", expanded=True) as status:
        output = st.empty()
        lines: list[str] = []
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            # Solo las últimas: el desglose de cada alerta son decenas de líneas.
            output.code("\n".join(lines[-14:]), language="text")
        code = process.wait()

        if code == 0:
            status.update(label=f"{region} actualizada", state="complete", expanded=False)
        else:
            status.update(label=f"{region} falló (código {code})", state="error", expanded=True)
    return code


def update_controls(regions: list[str], current: str) -> None:
    st.subheader("Actualizar")
    target = st.selectbox("Región a recalcular", regions, index=regions.index(current))
    send = st.checkbox(
        "Enviar alertas por Telegram",
        value=False,
        help="Sin marcar corre en --dry-run: no envía nada y no consume el cooldown.",
    )

    if st.button("Recalcular ahora", type="primary", width="stretch"):
        if run_region_now(target, send_alerts=send) == 0:
            st.cache_data.clear()
            st.rerun()

    if st.button("Releer del disco", width="stretch",
                 help="Recarga sin recalcular, por si otra corrida escribió mientras tanto."):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def chrome(chart: alt.Chart, p: dict[str, str]) -> alt.Chart:
    """Cromo recesivo común: rejilla y ejes a un tono de la superficie."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=p["grid"],
            gridWidth=1,
            domainColor=p["axis"],
            tickColor=p["axis"],
            labelColor=p["muted"],
            titleColor=p["muted"],
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor=p["text"], titleColor=p["muted"], labelFontSize=11, titleFontSize=11
        )
        .properties(background=p["surface"])
    )


def selected_symbol(event, table: pd.DataFrame) -> str | None:
    """Ticker de la fila elegida, a partir de lo que devuelve `st.dataframe`.

    El evento trae posiciones de fila, no índices del DataFrame, así que hay que
    resolverlas contra la tabla **tal como se mostró** (ya ordenada y filtrada).
    """
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows")
    if not rows:
        return None
    position = int(rows[0])
    # Al filtrar, la tabla encoge y una selección anterior puede quedar fuera.
    if position >= len(table):
        return None
    return str(table.iloc[position]["symbol"])


def panel_scatter(
    frame: pd.DataFrame,
    a_threshold: float,
    b_threshold: float,
    p: dict,
    highlight: str | None = None,
) -> alt.Chart:
    """A_pct vs B_pct. El AND de los dos umbrales es el cuadrante superior derecho."""
    data = frame.assign(estado=lambda d: d["alert"].map({True: "Alerta", False: "Sin alerta"}))

    base = alt.Chart(data)
    encoding = dict(
        x=alt.X("a_pct:Q", title="A_pct · momentum", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("b_pct:Q", title="B_pct · calidad", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "estado:N",
            title=None,
            scale=alt.Scale(domain=["Alerta", "Sin alerta"], range=[p["series_a"], p["inactive"]]),
            # Fuera del área de trazado: dentro cae justo sobre el cúmulo de alertas.
            legend=alt.Legend(orient="top", direction="horizontal", offset=8),
        ),
        order=alt.Order("alert:N"),  # las alertas se pintan encima
    )
    tooltip = [
        alt.Tooltip("symbol:N", title="Ticker"),
        alt.Tooltip("name:N", title="Nombre"),
        alt.Tooltip("sector:N", title="Sector"),
        alt.Tooltip("a_pct:Q", title="A_pct", format=".0f"),
        alt.Tooltip("b_pct:Q", title="B_pct", format=".0f"),
        alt.Tooltip("score_final:Q", title="Score", format=".1f"),
        alt.Tooltip("rank:Q", title="Puesto"),
        alt.Tooltip("reason:N", title="Motivo"),
    ]

    points = base.mark_point(filled=True, size=45, opacity=0.65, strokeWidth=0).encode(**encoding)
    # Capa invisible de radio amplio: el punto de 8px no puede ser el blanco de hover.
    hover = base.mark_circle(size=420, opacity=0).encode(
        x="a_pct:Q", y="b_pct:Q", tooltip=tooltip
    )
    # Sin etiquetas de ticker: las alertas se apiñan en la esquina superior
    # derecha y cualquier rótulo ahí se pisa con el vecino o se sale del área.
    # La identidad la llevan el tooltip y la tabla de debajo.
    # Los umbrales llevan significado (delimitan la zona de alerta), así que van
    # en tinta apagada y no en el color del eje: con el del eje se confunden con
    # la rejilla, sobre todo en modo oscuro.
    rules = alt.Chart(pd.DataFrame({"a": [a_threshold]})).mark_rule(
        color=p["muted"], strokeWidth=1
    ).encode(x="a:Q") + alt.Chart(pd.DataFrame({"b": [b_threshold]})).mark_rule(
        color=p["muted"], strokeWidth=1
    ).encode(y="b:Q")

    layers = [rules, points, hover]

    if highlight:
        chosen = data[data["symbol"] == highlight]
        if not chosen.empty:
            marked = alt.Chart(chosen)
            # Estas capas van las últimas, así que el elegido se dibuja SIEMPRE
            # por encima de los que tenga solapados. El disco del color de la
            # superficie abre un hueco alrededor para despegarlo de sus vecinos:
            # sin él, un punto en una zona densa se lee como parte del racimo.
            layers.append(
                marked.mark_point(
                    filled=True, size=430, opacity=1.0, color=p["surface"],
                ).encode(x="a_pct:Q", y="b_pct:Q")
            )
            # Anillo y etiqueta además del color: el amarillo solo no basta
            # para localizar un punto entre cientos.
            layers.append(
                marked.mark_point(
                    filled=True, size=260, opacity=1.0,
                    color=p["highlight"], stroke=p["text"], strokeWidth=2,
                ).encode(x="a_pct:Q", y="b_pct:Q")
            )
            layers.append(
                marked.mark_text(
                    dx=12, dy=-12, align="left", fontSize=12,
                    fontWeight="bold", color=p["text"],
                ).encode(x="a_pct:Q", y="b_pct:Q", text="symbol:N")
            )

    return chrome(alt.layer(*layers).properties(height=420), p)


def breakdown_chart(metrics: pd.DataFrame, color: str, title: str, p: dict) -> alt.Chart:
    """Contribución de cada métrica al panel_raw. Una serie: sin leyenda, el título nombra."""
    # Orden explícito: en un gráfico por capas `sort="-x"` no se resuelve y las
    # barras salen alfabéticas.
    order = metrics.sort_values("contribucion", ascending=False)["etiqueta"].tolist()

    base = alt.Chart(metrics)
    bars = base.mark_bar(cornerRadiusEnd=4, color=color, height=11).encode(
        x=alt.X("contribucion:Q", title="contribución al panel"),
        y=alt.Y("etiqueta:N", title=None, sort=order),
        tooltip=[
            alt.Tooltip("etiqueta:N", title="Métrica"),
            alt.Tooltip("percentil:Q", title="Percentil", format=".0f"),
            alt.Tooltip("peso:Q", title="Peso", format=".1f"),
            alt.Tooltip("contribucion:Q", title="Contribución", format=".1f"),
            alt.Tooltip("crudo:N", title="Valor crudo"),
        ],
    )
    # Etiqueta directa solo en las tres primeras: un número en cada barra es ruido.
    labels = (
        base.transform_filter(alt.datum.destacada)
        .mark_text(dx=5, fontSize=10, align="left", color=p["text"])
        .encode(
            x="contribucion:Q",
            y=alt.Y("etiqueta:N", sort=order),
            text=alt.Text("contribucion:Q", format=".1f"),
        )
    )
    return chrome(
        (bars + labels).properties(height=alt.Step(22), title=title), p
    ).configure_title(color=p["text"], fontSize=13, anchor="start")


def history_line(frame: pd.DataFrame, y: str, y_title: str, color: str, p: dict) -> alt.Chart:
    """Serie única en el tiempo: sin leyenda, marca de 2px."""
    # Resolución de día: sin timeUnit, Vega-Lite mete marcas de "12 PM" en una
    # serie que solo tiene un dato por cierre de sesión.
    axis_x = alt.X(
        "fecha:T",
        title=None,
        timeUnit="yearmonthdate",
        axis=alt.Axis(format="%-d %b", tickCount={"interval": "day", "step": 1}),
    )

    base = alt.Chart(frame)
    line = base.mark_line(strokeWidth=2, color=color).encode(
        x=axis_x, y=alt.Y(f"{y}:Q", title=y_title)
    )
    dots = base.mark_point(filled=True, size=60, color=color).encode(
        x=axis_x,
        y=f"{y}:Q",
        tooltip=[
            alt.Tooltip("fecha:T", title="Fecha", format="%-d %b %Y"),
            alt.Tooltip(f"{y}:Q", title=y_title, format=".1f"),
        ],
    )
    return chrome((line + dots).properties(height=220), p)


# ---------------------------------------------------------------------------
# Podio
# ---------------------------------------------------------------------------
# 2º a la izquierda, 1º en el centro y más alto, 3º a la derecha: el orden de
# lectura del podio no es el orden del ranking.
_PODIUM_LAYOUT = ((1, 74), (0, 108), (2, 52))


def medal_html(position: int, step: dict) -> str:
    """Chapa metálica con el número dentro: oro, plata y bronce.

    El brillo es un `conic-gradient` —un barrido de claro a oscuro alrededor del
    disco— en vez del degradado radial de siempre, que es lo que hacía que
    pareciese una pegatina. Encima, un aro interior claro que simula el bisel y
    una sombra corta que la despega del bloque.

    El aro exterior oscuro no es adorno: el oro sobre el bloque amarillo tiene
    1,29:1 de contraste y sin él la medalla se fundiría con él. Y el número va en
    tinta oscura porque la parada más oscura de cada metal mantiene ≥4,5:1 con
    ella, cosa que se comprobó midiendo.
    """
    light, dark = step["sheen"]
    metal = step["medal"]
    sheen = (
        f"conic-gradient(from 210deg, {dark}, {light} 18%, {metal} 34%, "
        f"{dark} 52%, {light} 68%, {metal} 84%, {dark})"
    )
    return (
        f'<div style="width:42px;height:42px;border-radius:50%;background:{sheen};'
        f'box-shadow:inset 0 0 0 2px rgba(255,255,255,0.34),'
        f'inset 0 0 0 3px rgba(11,11,11,0.16),'
        f'0 0 0 1.5px rgba(11,11,11,0.55),'
        f'0 3px 7px rgba(11,11,11,0.30);'
        f'display:flex;align-items:center;justify-content:center;'
        f'color:#0b0b0b;font-weight:700;font-size:19px;line-height:1;'
        f'font-variant-numeric:tabular-nums;'
        f'text-shadow:0 1px 0 rgba(255,255,255,0.45);'
        f'margin-top:-21px">{position + 1}</div>'
    )


def podium_html(top: pd.DataFrame, column: str, p: dict, *, decimals: int = 0) -> str:
    """Podio de tres puestos. El número va escrito: el color no es el único indicio."""
    blocks = []
    for position, height in _PODIUM_LAYOUT:
        if position >= len(top):
            continue
        row = top.iloc[position]
        step = p["podium"][position]
        blocks.append(
            f"""<div style="flex:1;display:flex;flex-direction:column;
                 align-items:center;justify-content:flex-end;min-width:0">
              <div style="font-size:12px;color:{p['muted']};margin-bottom:2px">
                {row[column]:.{decimals}f}</div>
              <div style="font-weight:600;font-size:15px;color:{p['text']};
                   margin-bottom:26px;max-width:100%;overflow:hidden;
                   text-overflow:ellipsis;white-space:nowrap"
                   title="{html_escape(str(row['name']))}">{html_escape(row['symbol'])}</div>
              <div data-rank="{position + 1}"
                   style="width:100%;height:{height}px;background:{step['bg']};
                   border-radius:8px 8px 0 0;display:flex;align-items:flex-start;
                   justify-content:center;overflow:visible">{medal_html(position, step)}</div>
            </div>"""
        )
    return (
        '<div style="display:flex;gap:10px;align-items:flex-end;'
        'margin:4px 0 2px 0">' + "".join(blocks) + "</div>"
    )


def html_escape(value: str) -> str:
    import html as _html

    return _html.escape(str(value))


def runners_up_html(top: pd.DataFrame, column: str, p: dict, decimals: int) -> str:
    """Puestos 4 y 5, etiquetados para que no queden sueltos bajo el podio."""
    rows = []
    for position in range(3, min(5, len(top))):
        row = top.iloc[position]
        rows.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'font-size:13px;color:{p["text"]};padding:4px 2px">'
            f'<span style="min-width:0;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap">'
            f'<span style="background:{p["grid"]};color:{p["muted"]};border-radius:4px;'
            f'padding:1px 6px;font-size:11px;font-weight:600;margin-right:6px">'
            f'TOP {position + 1}</span>'
            f'<b>{html_escape(row["symbol"])}</b> '
            f'<span style="color:{p["muted"]}">{html_escape(str(row["name"])[:24])}</span></span>'
            f'<span style="color:{p["muted"]};padding-left:8px">'
            f'{row[column]:.{decimals}f}</span></div>'
        )
    return "".join(rows)


def render_podium(frame: pd.DataFrame, column: str, title: str, p: dict,
                  *, decimals: int = 0, featured: bool = False) -> None:
    """Top 5: los tres primeros en podio, 4º y 5º debajo.

    Se emite como un único bloque HTML —en vez de varios `st.markdown`— para
    poder envolver el del total en una tarjeta con fondo propio.
    """
    top = frame.nlargest(5, column).reset_index(drop=True)
    if top.empty:
        st.info(f"Sin datos para {title}.")
        return

    if featured:
        card = (f'background:{p["featured_bg"]};'
                f'border:1px solid {p["featured_border"]};'
                f'border-radius:10px;padding:12px 14px 8px 14px')
        heading = (f'<div style="font-size:15px;font-weight:700;color:{p["text"]};'
                   f'border-left:4px solid {p["podium"][0]["bg"]};padding-left:8px;'
                   f'margin-bottom:8px">{html_escape(title)}</div>')
    else:
        card = "padding:12px 4px 8px 4px"
        heading = (f'<div style="font-size:14px;font-weight:600;color:{p["muted"]};'
                   f'margin-bottom:8px">{html_escape(title)}</div>')

    st.markdown(
        f'<div style="{card}">{heading}'
        f'{podium_html(top, column, p, decimals=decimals)}'
        f'<div style="border-top:1px solid {p["grid"]};margin:2px 0 4px 0"></div>'
        f'{runners_up_html(top, column, p, decimals)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Zonas del plano A/B
# ---------------------------------------------------------------------------
def zone_of(a_pct: float, b_pct: float, a_threshold: float, b_threshold: float) -> str:
    """Cuadrante de compra, su simétrico, o el centro.

    El simétrico no es "lo contrario de alertar" sino su reflejo exacto: tan
    abajo en los dos paneles como habría que estar arriba para alertar.
    """
    if a_pct >= a_threshold and b_pct >= b_threshold:
        return "compra"
    if a_pct <= 100 - a_threshold and b_pct <= 100 - b_threshold:
        return "opuesta"
    return ""


def zone_styler(view: pd.DataFrame, p: dict):
    """Fondo verde en la zona de compra, rojo en la simétrica, nada en medio."""
    colors = {"compra": p["zone_buy"], "opuesta": p["zone_avoid"]}

    def paint(row):
        color = colors.get(row["zona"], "")
        return [f"background-color: {color}" if color else ""] * len(row)

    return view.style.apply(paint, axis=1)


# ---------------------------------------------------------------------------
# Vistas
# ---------------------------------------------------------------------------
def metric_frame(row: pd.Series, panel: str) -> pd.DataFrame:
    """Desglose de un ticker para un panel, desde las columnas del CSV."""
    records = []
    for name, spec in REGISTRY.items():
        if spec.panel != panel:
            continue
        pct_col, raw_col = f"{name}__pct", f"{name}__raw"
        if pct_col not in row.index or pd.isna(row[pct_col]):
            continue
        percentile = float(row[pct_col])
        raw = row.get(raw_col)
        records.append(
            {
                "metrica": name,
                "etiqueta": spec.label,
                "percentil": percentile,
                "crudo": "s/d" if pd.isna(raw) else f"{float(raw):+.4g}",
                "imputada": bool(pd.isna(raw)),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    # El peso efectivo no está en el CSV; se reconstruye desde la contribución
    # total conocida: peso_i = peso_base_i renormalizado. Se usa el del config.
    return frame


def enrich_with_weights(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if frame.empty:
        return frame
    total = sum(weights.get(m, 0.0) for m in frame["metrica"])
    scale = 100.0 / total if total else 0.0
    frame = frame.assign(
        peso=[weights.get(m, 0.0) * scale for m in frame["metrica"]],
    )
    frame = frame.assign(contribucion=frame["percentil"] * frame["peso"] / 100.0)
    top = frame["contribucion"].nlargest(3).index
    return frame.assign(destacada=frame.index.isin(top))


@st.cache_data(ttl=60)
def load_frescura(path: str) -> pd.DataFrame:
    """Última corrida de cada región, incluidas las que nunca han corrido."""
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        filas = conn.execute(
            "SELECT region, MAX(run_on) AS run_on, run_at, scored, alerts, regime "
            "FROM runs GROUP BY region"
        ).fetchall()
    return pd.DataFrame([dict(f) for f in filas])


def dias_habiles(desde: date, hasta: date) -> int:
    """Días laborables entre dos fechas.

    Los cron corren de lunes a viernes, así que un lunes la corrida del viernes
    tiene tres días naturales de antigüedad y es perfectamente normal. Contando
    en naturales saltaría una alarma falsa cada lunes.
    """
    dias = 0
    cursor = desde
    while cursor < hasta:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            dias += 1
    return dias


def render_frescura(state_path: str, configuradas: list[str], p: dict) -> None:
    """Estado de actualización de TODAS las regiones configuradas.

    Se compara contra el config, no contra lo que hay en la base: una región que
    nunca corrió, o que lleva días caída, no aparecería en `runs` y desaparecería
    del panel en silencio, que es la peor forma de fallar cuando esto corre
    desatendido en un servidor.
    """
    frescura = load_frescura(state_path).set_index("region")
    hoy = date.today()
    columnas = st.columns(len(configuradas))

    for columna, region in zip(columnas, configuradas):
        with columna:
            if region not in frescura.index:
                st.markdown(
                    f"**{region}**<br><span style='color:{p['stale']}'>sin corridas</span>",
                    unsafe_allow_html=True)
                continue

            fila = frescura.loc[region]
            ultima = date.fromisoformat(str(fila["run_on"]))
            atraso = dias_habiles(ultima, hoy)

            # `run_at` se guarda en UTC porque el servidor va en UTC y los
            # cierres de mercado del config también. Pero quien lo lee está en
            # España, así que se convierte y se etiqueta la zona: un sello sin
            # zona invita a restar dos horas mentalmente y equivocarse.
            if fila.get("run_at"):
                marca = datetime.fromisoformat(str(fila["run_at"]))
                if marca.tzinfo is None:
                    marca = marca.replace(tzinfo=timezone.utc)
                local = marca.astimezone(ZoneInfo("Europe/Madrid"))
                sello = local.strftime("%d/%m %H:%M") + " (España)"
            else:
                sello = str(fila["run_on"])

            if atraso <= 1:
                color, aviso = p["fresh"], ""
            elif atraso <= 3:
                color, aviso = p["highlight"], " ⚠"
            else:
                color, aviso = p["stale"], " ⚠"

            cuando = "hoy" if atraso == 0 else f"hace {atraso} día{'s' if atraso > 1 else ''} hábil{'es' if atraso > 1 else ''}"
            st.markdown(
                f"**{region}**{aviso}<br>"
                f"<span style='color:{color}'>{cuando}</span><br>"
                f"<span style='font-size:0.8em;opacity:0.7'>{sello}<br>"
                f"{int(fila['scored'])} puntuados · {int(fila['alerts'])} alertas</span>",
                unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")
    cfg = get_config()
    p = palette()
    state_path = str(ROOT / Path(cfg.run["state_db"]).name)
    output_dir = str(ROOT / Path(cfg.run["output_dir"]).name)

    st.title("Stock Screener")

    configured = [key for key, r in cfg.regions.items() if r.enabled]
    render_frescura(state_path, configured, p)
    st.divider()

    # --- una sola fila de filtros, arriba, que alcanza a todo -------------
    available = get_state(state_path).regions()
    if not available:
        st.warning("No hay corridas registradas todavía. Lanza la primera desde aquí.")
        with st.sidebar:
            update_controls(configured or list(cfg.regions), configured[0] if configured else "us")
        return

    with st.sidebar:
        st.subheader("Filtros")
        region = st.selectbox("Región", available, key="region")
        runs = load_runs(state_path, region)
        days = sorted(runs["run_on"].tolist(), reverse=True)
        # Clave con la región dentro: si no, Streamlit conserva el estado del
        # widget al cambiar de región y se puede acabar leyendo el CSV de una
        # con la fecha de otra.
        day = st.selectbox("Fecha", days, key=f"fecha_{region}")
        only_alerts = st.checkbox("Solo alertas", value=False, key=f"alertas_{region}")
        only_watchlist = st.checkbox("Solo watchlist", value=False, key=f"watch_{region}")
        st.divider()
        update_controls(sorted(set(configured) | set(available)), region)

    if day not in days:  # defensivo: fecha huérfana de otra región
        day = days[0]

    detail = load_detail(output_dir, region, day)
    matching = runs[runs["run_on"] == day]
    if matching.empty:
        st.error(f"No hay corrida de {region} el {day}.")
        return
    run_row = matching.iloc[0]

    # --- KPIs -------------------------------------------------------------
    columns = st.columns(4)
    columns[0].metric("Universo", f"{int(run_row['universe_size']):,}".replace(",", "."))
    columns[1].metric("Puntuados", f"{int(run_row['scored']):,}".replace(",", "."))
    columns[2].metric("Alertas", int(run_row["alerts"]))
    columns[3].metric("Régimen", f"{run_row['regime']:.3f}")
    if run_row["regime_detail"]:
        st.caption(run_row["regime_detail"])

    if detail.empty:
        st.info(
            f"No hay CSV de esa corrida en `{output_dir}`. Los KPIs y el histórico "
            "vienen de la base de estado, pero el desglose por métrica necesita el CSV."
        )
        return

    sectors = sorted(detail["sector"].dropna().unique().tolist())
    with st.sidebar:
        chosen_sectors = st.multiselect(
            "Sectores", sectors, default=[], key=f"sectores_{region}"
        )

    view = detail
    if only_alerts:
        view = view[view["alert"]]
    if only_watchlist:
        view = view[view["watchlist"]]
    if chosen_sectors:
        view = view[view["sector"].isin(chosen_sectors)]

    tabs = st.tabs(["Ranking", "Detalle de un valor", "Histórico de alertas"])

    # --- Ranking ----------------------------------------------------------
    with tabs[0]:
        a_threshold = float(cfg.alerting["a_threshold"])
        b_threshold = float(cfg.alerting["b_threshold"])
        st.caption(
            f"Alerta = A_pct ≥ {a_threshold:.0f} **y** B_pct ≥ {b_threshold:.0f} "
            f"**y** score ≥ {cfg.alerting['final_cut']}. Las líneas marcan los dos umbrales: "
            "el cuadrante superior derecho es la zona de alerta."
        )
        if view.empty:
            st.info("Ningún valor cumple los filtros seleccionados.")
        else:
            # Los podios van sobre TODA la región, no sobre la vista filtrada:
            # "el mejor momentum de us" no debe cambiar al marcar un sector.
            # El total va en medio y enmarcado: es el que decide la alerta.
            momentum_col, total_col, quality_col = st.columns([1, 1.15, 1])
            with momentum_col:
                render_podium(detail, "a_pct", "Panel A · momentum", p)
            with total_col:
                render_podium(detail, "score_final", "★ Total · score final",
                              p, decimals=1, featured=True)
            with quality_col:
                render_podium(detail, "b_pct", "Panel B · calidad", p)
            st.divider()

            table = view.sort_values("score_final", ascending=False).copy()
            # Hueco reservado: el diagrama va arriba pero se dibuja DESPUÉS de
            # leer la fila elegida, así refleja el clic en la misma pasada en
            # vez de depender de que el estado del widget sobreviva al rerun.
            chart_slot = st.empty()
            table["zona"] = [
                zone_of(a, b, a_threshold, b_threshold)
                for a, b in zip(table["a_pct"], table["b_pct"])
            ]
            counts = table["zona"].value_counts()
            st.caption(
                f"**Marca la casilla ☐ de la izquierda** de una fila para resaltar su "
                f"punto en el diagrama (Streamlit solo detecta el clic ahí, no en "
                f"el resto de la fila). "
                f"Fondo verde: los {int(counts.get('compra', 0))} de la zona de compra "
                f"(A ≥ {a_threshold:.0f} y B ≥ {b_threshold:.0f}). Fondo rojo: los "
                f"{int(counts.get('opuesta', 0))} del cuadrante simétrico "
                f"(A ≤ {100 - a_threshold:.0f} y B ≤ {100 - b_threshold:.0f})."
            )
            columns = ["rank", "symbol", "name", "sector", "a_pct", "b_pct",
                       "score_final", "zona", "reason"]
            event = st.dataframe(
                zone_styler(table[columns], p),
                hide_index=True,
                width="stretch",
                key=f"ranking_{region}_{day}",
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "rank": st.column_config.NumberColumn("#", width="small"),
                    "symbol": "Ticker",
                    "name": "Nombre",
                    "sector": "Sector",
                    "a_pct": st.column_config.NumberColumn("A_pct", format="%.0f"),
                    "b_pct": st.column_config.NumberColumn("B_pct", format="%.0f"),
                    "score_final": st.column_config.NumberColumn("Score", format="%.1f"),
                    "zona": "Zona",
                    "reason": "Motivo",
                },
            )

            chosen = selected_symbol(event, table)
            with chart_slot:
                st.altair_chart(
                    panel_scatter(view, a_threshold, b_threshold, p, highlight=chosen),
                    width="stretch",
                )

    # --- Detalle ----------------------------------------------------------
    with tabs[1]:
        ordered = view.sort_values("score_final", ascending=False)
        if ordered.empty:
            st.info("Ningún valor cumple los filtros seleccionados.")
        else:
            symbol = st.selectbox(
                "Valor", ordered["symbol"].tolist(), key=f"valor_{region}_{day}"
            )
            row = detail[detail["symbol"] == symbol].iloc[0]

            head = st.columns(4)
            head[0].metric("A_pct", f"{row['a_pct']:.0f}")
            head[1].metric("B_pct", f"{row['b_pct']:.0f}")
            head[2].metric("Score final", f"{row['score_final']:.1f}")
            head[3].metric("Puesto", int(row["rank"]))
            st.caption(("🚀 " if row["alert"] else "") + str(row["reason"]))

            weights = cfg.region(region).weights
            left, right = st.columns(2)
            for column, panel_key, title, color in (
                (left, "momentum", "Panel A · Momentum", p["series_a"]),
                (right, "quality", "Panel B · Calidad", p["series_b"]),
            ):
                metrics = enrich_with_weights(metric_frame(row, panel_key), weights[panel_key])
                with column:
                    if metrics.empty:
                        st.info(f"Sin datos del {title}.")
                        continue
                    st.altair_chart(
                        breakdown_chart(metrics, color, title, p), width="stretch"
                    )
                    st.dataframe(
                        metrics.sort_values("contribucion", ascending=False)[
                            ["etiqueta", "percentil", "peso", "contribucion", "crudo", "imputada"]
                        ],
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "etiqueta": "Métrica",
                            "percentil": st.column_config.NumberColumn("Pct", format="%.0f"),
                            "peso": st.column_config.NumberColumn("Peso", format="%.1f"),
                            "contribucion": st.column_config.NumberColumn("Aporta", format="%.1f"),
                            "crudo": "Crudo",
                            "imputada": st.column_config.CheckboxColumn("Imputada"),
                        },
                    )

            history = load_snapshots(state_path, region)
            history = history[history["symbol"] == symbol].assign(
                fecha=lambda d: pd.to_datetime(d["snapshot_on"])
            )
            st.subheader(f"Histórico de {symbol}")
            if len(history) < 2:
                st.info(
                    "Hace falta más de una corrida para dibujar el histórico. "
                    "Se acumula solo: cada corrida guarda su snapshot."
                )
            else:
                st.altair_chart(
                    history_line(history, "score", "score final", p["series_a"], p),
                    width="stretch",
                )

    # --- Alertas ----------------------------------------------------------
    with tabs[2]:
        st.caption(
            "Solo se envía lo **nuevo** y lo que **mejora** de verdad: un valor ya "
            "avisado se silencia mientras siga en el corte sin moverse."
        )
        alerts = load_alerts(state_path, region, days=90)
        if alerts.empty:
            st.info(
                "No hay alertas registradas. En `--dry-run` no se anotan a propósito, "
                "para que una prueba no consuma el cooldown."
            )
        else:
            st.dataframe(
                alerts,
                hide_index=True,
                width="stretch",
                column_config={
                    "symbol": "Ticker",
                    "alerted_on": "Fecha",
                    "a_pct": st.column_config.NumberColumn("A_pct", format="%.0f"),
                    "b_pct": st.column_config.NumberColumn("B_pct", format="%.0f"),
                    "score": st.column_config.NumberColumn("Score", format="%.1f"),
                    "rank": st.column_config.NumberColumn("#", format="%d"),
                    "kind": "Tipo",
                },
            )

        if len(runs) >= 2:
            st.subheader("Régimen de mercado")
            st.altair_chart(
                history_line(
                    runs.assign(fecha=lambda d: pd.to_datetime(d["run_on"])),
                    "regime",
                    "multiplicador",
                    p["series_a"],
                    p,
                ),
                width="stretch",
            )


# Streamlit ejecuta el script con `__name__ == "__main__"`, así que el panel
# arranca igual con `streamlit run`. El guard existe para que importar el módulo
# —lo que hacen los tests— no levante la aplicación entera contra una base de
# estado que en un clon recién hecho todavía no existe.
if __name__ == "__main__":
    main()
