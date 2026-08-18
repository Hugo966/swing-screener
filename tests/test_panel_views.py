"""Podio y zonas del plano A/B del panel."""

from __future__ import annotations

import pandas as pd
import pytest

from screener.panel import PALETTES, podium_html, zone_of, zone_styler


def frame():
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "name": ["Alfa SA", "Beta Ltd", "Gamma", "Delta", "Epsilon", "Zeta"],
            "a_pct": [99.0, 95.0, 90.0, 85.0, 80.0, 10.0],
            "b_pct": [50.0, 60.0, 70.0, 80.0, 90.0, 5.0],
        }
    )


# ---------------------------------------------------------------------------
# Zonas
# ---------------------------------------------------------------------------
def test_buy_zone_needs_both_panels():
    assert zone_of(85, 85, 80, 80) == "compra"
    assert zone_of(80, 80, 80, 80) == "compra"  # el umbral entra
    assert zone_of(99, 79, 80, 80) == ""        # uno solo no basta
    assert zone_of(79, 99, 80, 80) == ""


def test_opposite_zone_is_the_mirror_of_the_buy_zone():
    """Con umbral 80, el simétrico es A<=20 y B<=20: el reflejo exacto."""
    assert zone_of(10, 10, 80, 80) == "opuesta"
    assert zone_of(20, 20, 80, 80) == "opuesta"
    assert zone_of(21, 20, 80, 80) == ""
    assert zone_of(20, 21, 80, 80) == ""


def test_the_middle_has_no_zone():
    for a, b in [(50, 50), (79, 79), (21, 21), (99, 5), (5, 99)]:
        assert zone_of(a, b, 80, 80) == ""


def test_zones_follow_the_configured_threshold():
    """Con 90/90 el simétrico se encoge a 10/10, no se queda en 20."""
    assert zone_of(15, 15, 90, 90) == ""
    assert zone_of(10, 10, 90, 90) == "opuesta"
    assert zone_of(85, 85, 90, 90) == ""


def test_styler_paints_only_the_two_zones():
    view = pd.DataFrame(
        {"symbol": ["A", "B", "C"], "zona": ["compra", "", "opuesta"]}
    )
    p = PALETTES["light"]
    rendered = zone_styler(view, p).to_html()

    assert p["zone_buy"].replace(" ", "") in rendered.replace(" ", "")
    assert p["zone_avoid"].replace(" ", "") in rendered.replace(" ", "")


# ---------------------------------------------------------------------------
# Podio
# ---------------------------------------------------------------------------
def test_podium_puts_the_winner_in_the_middle_and_taller():
    """2º-1º-3º de izquierda a derecha, y el 1º más alto: eso es un podio."""
    html = podium_html(frame().nlargest(3, "a_pct").reset_index(drop=True),
                       "a_pct", PALETTES["light"])

    orden = [html.index(s) for s in ("BBB", "AAA", "CCC")]
    assert orden == sorted(orden), "el orden visual debe ser 2º, 1º, 3º"

    # solo los bloques del podio: las medallas también llevan height
    alturas = [int(h) for h in __import__("re").findall(
        r'data-rank="\d+"\s+style="width:100%;height:(\d+)px', html)]
    assert alturas[1] == max(alturas), "el primero es el bloque más alto"
    assert alturas[0] > alturas[2], "el segundo es más alto que el tercero"


def test_podium_writes_the_rank_so_colour_is_not_the_only_cue():
    html = podium_html(frame().nlargest(3, "b_pct").reset_index(drop=True),
                       "b_pct", PALETTES["light"])
    for rank in ("1", "2", "3"):
        assert f">{rank}</div>" in html


def test_podium_escapes_names_with_html():
    data = pd.DataFrame(
        {"symbol": ["A&B"], "name": ["Smith & Wesson <Brands>"], "a_pct": [99.0]}
    )
    html = podium_html(data, "a_pct", PALETTES["light"])
    assert "&amp;" in html
    assert "<Brands>" not in html


def test_podium_survives_fewer_than_three_names():
    data = frame().head(2)
    html = podium_html(data, "a_pct", PALETTES["light"])
    assert "AAA" in html and "BBB" in html


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_both_themes_define_every_slot_the_panel_uses(mode):
    p = PALETTES[mode]
    for key in ("surface", "text", "muted", "grid", "axis", "series_a", "series_b",
                "inactive", "podium", "zone_buy", "zone_avoid", "highlight",
                "featured_bg", "featured_border"):
        assert key in p, key
    assert len(p["podium"]) == 3
    for step in p["podium"]:
        assert set(step) == {"bg", "ink", "medal", "sheen"}


def test_podium_colours_are_fixed_per_rank_not_per_panel():
    """1º amarillo, 2º verde, 3º azul — igual en los tres podios."""
    a = podium_html(frame().nlargest(3, "a_pct").reset_index(drop=True), "a_pct", PALETTES["light"])
    b = podium_html(frame().nlargest(3, "b_pct").reset_index(drop=True), "b_pct", PALETTES["light"])

    colores = lambda html: __import__("re").findall(
        r'data-rank="\d+"\s+style="width:100%;height:\d+px;background:(#[0-9a-f]{6})', html)
    assert colores(a) == colores(b), "el color depende del puesto, no del panel"
    # el orden visual es 2º, 1º, 3º
    assert colores(a) == ["#008300", "#ffc107", "#2a78d6"]


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_podium_ink_is_readable_on_its_background(mode):
    """Blanco sobre el amarillo da 2,11:1: hay que medirlo, no suponerlo."""
    def luminance(hex_color):
        hex_color = hex_color.lstrip("#")
        channels = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    for step in PALETTES[mode]["podium"]:
        light, dark = sorted((luminance(step["bg"]), luminance(step["ink"])), reverse=True)
        # el número va en 20px bold: umbral de texto grande, 3:1
        assert (light + 0.05) / (dark + 0.05) >= 3.0, step


def test_total_podium_shows_one_decimal():
    data = pd.DataFrame(
        {"symbol": ["A"], "name": ["Alfa"], "score_final": [86.64]}
    )
    assert "86.6" in podium_html(data, "score_final", PALETTES["light"], decimals=1)


# ---------------------------------------------------------------------------
# Resaltado del punto elegido
# ---------------------------------------------------------------------------
def scatter_frame():
    return pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "name": ["Alfa", "Beta", "Gamma"],
        "sector": ["Technology"] * 3,
        "a_pct": [95.0, 50.0, 10.0],
        "b_pct": [90.0, 50.0, 10.0],
        "score_final": [80.0, 25.0, 1.0],
        "rank": [1, 2, 3],
        "alert": [True, False, False],
        "reason": ["ok", "no", "no"],
    })


def _marks(spec):
    """Tipos de marca, bajando a las capas anidadas (las reglas son un layer)."""
    import json

    def walk(node):
        found = []
        if "mark" in node:
            mark = node["mark"]
            found.append(mark["type"] if isinstance(mark, dict) else mark)
        for child in node.get("layer", []):
            found.extend(walk(child))
        return found

    return walk(json.loads(spec))


def test_scatter_without_selection_has_no_highlight_layer():
    from screener.panel import panel_scatter
    spec = panel_scatter(scatter_frame(), 80, 80, PALETTES["light"]).to_json()
    assert _marks(spec).count("text") == 0


def test_selecting_a_row_adds_a_marked_point_and_its_label():
    from screener.panel import panel_scatter
    spec = panel_scatter(scatter_frame(), 80, 80, PALETTES["light"], highlight="BBB").to_json()
    marks = _marks(spec)

    assert marks.count("text") == 1, "la etiqueta con el ticker"
    assert PALETTES["light"]["highlight"] in spec
    # anillo: el color solo no localiza un punto entre cientos
    assert '"stroke"' in spec or "stroke" in spec


def test_highlighting_an_absent_symbol_is_harmless():
    from screener.panel import panel_scatter
    spec = panel_scatter(scatter_frame(), 80, 80, PALETTES["light"], highlight="ZZZ").to_json()
    assert _marks(spec).count("text") == 0


def test_selected_symbol_reads_the_row_position():
    """El evento trae posiciones de fila, no índices del DataFrame."""
    from screener.panel import selected_symbol

    table = scatter_frame()
    assert selected_symbol({"selection": {"rows": [2]}}, table) == "CCC"
    assert selected_symbol({"selection": {"rows": [0]}}, table) == "AAA"
    assert selected_symbol({"selection": {"rows": []}}, table) is None
    assert selected_symbol(None, table) is None


def test_selected_symbol_ignores_a_stale_out_of_range_row():
    """Al filtrar, la tabla encoge y la selección anterior puede quedar fuera."""
    from screener.panel import selected_symbol

    assert selected_symbol({"selection": {"rows": [99]}}, scatter_frame()) is None


def test_selected_symbol_resolves_against_the_displayed_order():
    """La tabla se muestra ordenada por score: la posición 0 es la de arriba."""
    from screener.panel import selected_symbol

    ordenada = scatter_frame().sort_values("score_final", ascending=False)
    assert selected_symbol({"selection": {"rows": [0]}}, ordenada) == "AAA"

    inversa = scatter_frame().sort_values("score_final")
    assert selected_symbol({"selection": {"rows": [0]}}, inversa) == "CCC"


# ---------------------------------------------------------------------------
# Medallas
# ---------------------------------------------------------------------------
def test_each_rank_gets_its_metal():
    """Oro con el 1, plata con el 2, bronce con el 3."""
    html = podium_html(frame().nlargest(3, "a_pct").reset_index(drop=True),
                       "a_pct", PALETTES["light"])
    for metal in ("#d4af37", "#c0c0c0", "#cd7f32"):
        assert metal in html
    # el número sigue escrito dentro de la medalla
    for rank in ("1", "2", "3"):
        assert f">{rank}</div>" in html


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_medals_carry_a_ring(mode):
    """El oro sobre el bloque amarillo da 1,29:1: sin anillo se funde con él."""
    from screener.panel import medal_html

    for position, step in enumerate(PALETTES[mode]["podium"]):
        assert "rgba(11,11,11,0.55)" in medal_html(position, step)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_medal_number_is_readable_on_every_metal(mode):
    def luminance(hex_color):
        hex_color = hex_color.lstrip("#")
        channels = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    for step in PALETTES[mode]["podium"]:
        for stop in (step["medal"], *step["sheen"]):
            high, low = sorted((luminance(stop), luminance("#0b0b0b")), reverse=True)
            assert (high + 0.05) / (low + 0.05) >= 4.5, stop


def test_featured_podium_gets_its_own_card():
    """El total va destacado: fondo propio y título con acento."""
    import streamlit as st

    # comprobación directa sobre el HTML que produce el bloque destacado
    from screener.panel import render_podium

    emitted = []
    original = st.markdown
    st.markdown = lambda body, **kw: emitted.append(body)
    try:
        render_podium(frame(), "a_pct", "Total", PALETTES["light"], featured=True)
        plain = []
        st.markdown = lambda body, **kw: plain.append(body)
        render_podium(frame(), "a_pct", "Total", PALETTES["light"], featured=False)
    finally:
        st.markdown = original

    assert PALETTES["light"]["featured_bg"] in emitted[0]
    assert PALETTES["light"]["featured_bg"] not in plain[0]
    assert "border-left:4px solid" in emitted[0]


def test_highlight_sits_above_everything_else():
    """Si el elegido está solapado, tiene que quedar encima, no debajo."""
    import json

    from screener.panel import panel_scatter

    spec = json.loads(
        panel_scatter(scatter_frame(), 80, 80, PALETTES["light"], highlight="BBB").to_json()
    )
    # las dos últimas capas son el disco resaltado y su etiqueta
    últimas = spec["layer"][-2:]
    assert últimas[-1]["mark"]["type"] == "text"
    assert PALETTES["light"]["highlight"] in json.dumps(últimas)
    # y hay un disco del color de la superficie que lo despega de sus vecinos
    assert PALETTES["light"]["surface"] in json.dumps(spec["layer"][-3])


# ---------------------------------------------------------------------------
# Aislamiento entre regiones
# ---------------------------------------------------------------------------
def test_every_region_dependent_widget_is_keyed_by_region():
    """Sin clave propia, Streamlit conserva el estado al cambiar de región y se
    puede acabar leyendo el CSV de una con la fecha o el filtro de otra."""
    import re
    from pathlib import Path

    source = Path("screener/panel.py").read_text(encoding="utf-8")
    # se busca por la etiqueta: la llamada puede estar partida en varias líneas
    for label in ('st.selectbox("Fecha"', '"Sectores", sectors',
                  '"Solo alertas"', '"Solo watchlist"',
                  '"Valor", ordered'):
        start = source.index(label)
        call = source[start:start + 260]
        assert re.search(r'key=f"[a-z_]+\{region\}', call), label


def test_the_table_selection_key_includes_region_and_date():
    from pathlib import Path

    source = Path("screener/panel.py").read_text(encoding="utf-8")
    assert 'key=f"ranking_{region}_{day}"' in source


def test_the_chart_is_drawn_from_the_same_frame_as_the_table():
    """El diagrama y la tabla salen del mismo `detail`, nunca de dos cargas."""
    from pathlib import Path

    source = Path("screener/panel.py").read_text(encoding="utf-8")
    # una sola carga de CSV por pasada
    assert source.count("load_detail(output_dir, region, day)") == 1
