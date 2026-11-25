"""Ferramentas para o relatório HTML autocontido: estilos, tabelas, figuras embutidas e gráficos SVG."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M_COL = {"Perceptron": "var(--m1)", "Random Forest": "var(--m2)", "LSTM": "var(--m3)"}
MODELS = ["Perceptron", "Random Forest", "LSTM"]

CSS = """
:root{
  --bg:#f6f5f2; --surface:#ffffff; --fg:#1b1b19; --fg2:#53524d; --muted:#7c7b75; --line:#e2e0da; --accent:#1f4e8c; --accent-soft:#e7eef8;
  --ok:#1f7a4d; --warn:#9a6a00; --bad:#b3372f; --warn-soft:#fbf2dc; --bad-soft:#fbe9e7; --ok-soft:#e5f3ec;
  --m1:#2a78d6; --m2:#eb6834; --m3:#1baf7a; --ref:#8a8984; --grid:#e8e6e1; --code:#f0eeea; --shadow:0 1px 2px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#151514; --surface:#1d1d1b; --fg:#ecebe6; --fg2:#bcbab2; --muted:#8f8d86; --line:#33322e; --accent:#7fb0ee; --accent-soft:#1c2a3d;
  --ok:#59c08b; --warn:#e2b04a; --bad:#ee7d73; --warn-soft:#332b17; --bad-soft:#35201e; --ok-soft:#18291f;
  --m1:#3987e5; --m2:#d95926; --m3:#199e70; --ref:#8a8984; --grid:#2c2b28; --code:#262522; --shadow:none; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#151514; --surface:#1d1d1b; --fg:#ecebe6; --fg2:#bcbab2; --muted:#8f8d86; --line:#33322e; --accent:#7fb0ee; --accent-soft:#1c2a3d;
  --ok:#59c08b; --warn:#e2b04a; --bad:#ee7d73; --warn-soft:#332b17; --bad-soft:#35201e; --ok-soft:#18291f;
  --m1:#3987e5; --m2:#d95926; --m3:#199e70; --ref:#8a8984; --grid:#2c2b28; --code:#262522; --shadow:none; color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.62 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;-webkit-text-size-adjust:100%}
.layout{display:grid;grid-template-columns:270px minmax(0,1fr);gap:0;max-width:1320px;margin:0 auto}
nav.toc{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:28px 18px 28px 24px;border-right:1px solid var(--line);font-size:13.5px}
nav.toc .brand{font:600 13px/1.3 system-ui;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
nav.toc ol{list-style:none;margin:0;padding:0}
nav.toc li{margin:1px 0}
nav.toc a{display:block;padding:5px 8px;border-radius:6px;color:var(--fg2);text-decoration:none}
nav.toc a:hover{background:var(--accent-soft);color:var(--accent)}
nav.toc .num{display:inline-block;min-width:1.7em;color:var(--muted);font-variant-numeric:tabular-nums}
main{padding:44px clamp(18px,4vw,56px) 80px;min-width:0}
header.hero{margin-bottom:34px;padding-bottom:26px;border-bottom:1px solid var(--line)}
.eyebrow{font:600 12px/1 system-ui;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);margin-bottom:12px}
h1{font:400 clamp(28px,4.2vw,42px)/1.15 Georgia,"Times New Roman",serif;margin:0 0 12px;text-wrap:balance}
h2{font:400 clamp(23px,3vw,30px)/1.2 Georgia,"Times New Roman",serif;margin:56px 0 14px;padding-top:6px;text-wrap:balance}
h2 .n{font:600 13px/1 system-ui;color:var(--accent);letter-spacing:.06em;display:block;margin-bottom:8px;text-transform:uppercase}
h3{font:600 17px/1.3 system-ui;margin:30px 0 8px}
h4{font:600 14px/1.3 system-ui;margin:20px 0 6px;color:var(--fg2);text-transform:uppercase;letter-spacing:.05em}
p{margin:0 0 14px;max-width:74ch}
.lead{font-size:18px;line-height:1.55;color:var(--fg2);max-width:70ch}
ul,ol.l{margin:0 0 16px;padding-left:1.25em;max-width:74ch}
li{margin:4px 0}
a{color:var(--accent)}
code,.mono{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);padding:1px 5px;border-radius:4px}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow:auto;font:13px/1.55 ui-monospace,Menlo,Consolas,monospace;max-width:100%;margin:0 0 18px}
pre code{background:none;padding:0}
.grid{display:grid;gap:14px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin:20px 0 8px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:var(--shadow);min-width:0}
.kpi .lab{font:600 12px/1.3 system-ui;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.kpi .val{font:400 34px/1.15 Georgia,serif;font-variant-numeric:tabular-nums;margin:6px 0 2px}
.kpi .sub{font-size:13px;color:var(--fg2)}
.kpi .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.callout{border-radius:10px;padding:14px 18px;margin:18px 0;max-width:80ch;border:1px solid var(--line);background:var(--surface)}
.callout.warn{background:var(--warn-soft);border-color:transparent}
.callout.bad{background:var(--bad-soft);border-color:transparent}
.callout.ok{background:var(--ok-soft);border-color:transparent}
.callout.info{background:var(--accent-soft);border-color:transparent}
.callout b.t{display:block;font:600 12px/1.3 system-ui;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}
.callout p:last-child{margin-bottom:0}
.twrap{overflow-x:auto;margin:6px 0 22px;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 12px/1.3 system-ui;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;background:var(--surface);white-space:nowrap}
tr:last-child td{border-bottom:none}
td.n,th.n{text-align:right;white-space:nowrap}
tbody tr:hover td{background:color-mix(in srgb,var(--accent-soft) 55%,transparent)}
td.hl{font-weight:600}
.cap{font-size:13px;color:var(--muted);margin:-12px 0 22px;max-width:80ch}
figure{margin:8px 0 26px}
figure .img{background:#fcfcfb;border:1px solid var(--line);border-radius:10px;padding:10px;overflow:hidden}
figure img{display:block;width:100%;height:auto}
figcaption{font-size:13px;color:var(--muted);margin-top:8px;max-width:80ch}
.chart{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 12px 8px;margin:10px 0 8px;overflow-x:auto}
.chart svg{display:block;max-width:100%;height:auto;margin:0 auto}
.chart text{fill:var(--fg2);font:11.5px system-ui,sans-serif}
.chart .t{fill:var(--fg);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--fg2);margin:0 0 6px 4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.tag{display:inline-block;font:600 11px/1 system-ui;padding:3px 7px;border-radius:99px;letter-spacing:.04em;text-transform:uppercase;vertical-align:1px}
.tag.ok{background:var(--ok-soft);color:var(--ok)} .tag.warn{background:var(--warn-soft);color:var(--warn)} .tag.bad{background:var(--bad-soft);color:var(--bad)} .tag.info{background:var(--accent-soft);color:var(--accent)}
details{border:1px solid var(--line);border-radius:10px;background:var(--surface);margin:10px 0 18px;padding:0}
details>summary{cursor:pointer;padding:11px 16px;font-weight:600;font-size:14.5px}
details>div{padding:0 16px 14px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;align-items:start}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
.themebtn{position:fixed;right:14px;bottom:14px;z-index:5;background:var(--surface);color:var(--fg2);border:1px solid var(--line);border-radius:99px;padding:7px 13px;font:600 12px system-ui;cursor:pointer;box-shadow:var(--shadow)}
@media (max-width:980px){.layout{grid-template-columns:1fr}nav.toc{position:static;height:auto;border-right:none;border-bottom:1px solid var(--line);padding:18px 18px 8px}nav.toc ol{columns:2}}
@media (max-width:560px){nav.toc ol{columns:1}body{font-size:15.5px}}
@media print{nav.toc,.themebtn{display:none}.layout{display:block}body{background:#fff;color:#000}h2{break-before:auto;break-after:avoid}.twrap,figure,.callout,.chart,.kpi{break-inside:avoid}main{padding:0}}
"""


def esc(s) -> str:
    return html.escape(str(s))


def pt(x, d=2, sign=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pc(x, d=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else pt(100 * x, d) + " %"


def table(headers, rows, num=(), hl_col=None, cls="") -> str:
    th = "".join(f'<th class="{"n" if i in num else ""}">{h}</th>' for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f'<td class="{"n" if i in num else ""}{" hl" if i == hl_col else ""}">{c}</td>' for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return f'<div class="twrap {cls}"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def img_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def figure(name: str, caption: str, alt: str) -> str:
    return (f'<figure><div class="img"><img alt="{esc(alt)}" src="{img_data(ROOT / "outputs" / "figures" / name)}"></div>'
            f"<figcaption>{caption}</figcaption></figure>")


def callout(kind: str, title: str, body: str) -> str:
    return f'<div class="callout {kind}"><b class="t">{title}</b>{body}</div>'


def kpi(label, value, sub, color=None) -> str:
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ""
    return f'<div class="kpi"><div class="lab">{dot}{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def legend(items) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


# ------------------------------------------------------------------ gráficos SVG
def _axis_x(x0, x1, y, ticks, fmt, scale):
    out = [f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" stroke="var(--grid)"/>']
    for v in ticks:
        out.append(f'<line x1="{scale(v):.1f}" x2="{scale(v):.1f}" y1="{y}" y2="{y+4}" stroke="var(--ref)"/>')
        out.append(f'<text x="{scale(v):.1f}" y="{y+17}" text-anchor="middle">{fmt(v)}</text>')
    return "".join(out)


def svg_ci(ci) -> str:
    """Três painéis (AUC, Brier, F1): ponto e IC por modelo e linha de referência."""
    panels = [("auc", "AUC", "acaso"), ("brier", "Brier (menor é melhor)", "previsão constante"), ("f1", "F1 da regra principal", "alertar sempre")]
    pw, h = 330, 190
    out = [f'<svg viewBox="0 0 {pw*3} {h}" width="{pw*3}" role="img" aria-label="AUC, Brier e F1 no teste por modelo, com intervalo de confiança de 95 %">']
    for k, (met, title, ref_lbl) in enumerate(panels):
        ox = k * pw
        rows = {m: ci[(ci.modelo == m) & (ci.metrica == met)].iloc[0] for m in MODELS}
        ref = ci[(ci.modelo == "Referência") & (ci.metrica == met)].iloc[0].ponto
        lo = min([r.ic_baixo for r in rows.values()] + [ref]); hi = max([r.ic_alto for r in rows.values()] + [ref])
        pad = (hi - lo) * 0.1
        sx = lambda v, lo=lo, pad=pad, hi=hi, ox=ox: ox + 22 + (v - (lo - pad)) / ((hi + pad) - (lo - pad)) * (pw - 44)
        out.append(f'<text class="t" x="{ox+22}" y="18">{title}</text>')
        rx = sx(ref)
        out.append(f'<line x1="{rx:.1f}" x2="{rx:.1f}" y1="52" y2="{h-38}" stroke="var(--ref)" stroke-dasharray="4 3"/>')
        dd = 3 if met == "brier" else 2
        out.append(f'<text x="{rx:.1f}" y="40" text-anchor="middle">{ref_lbl} {pt(ref, dd)}</text>')
        for i, m in enumerate(MODELS):
            r = rows[m]; y = 66 + i * 38
            out.append(f'<line x1="{sx(r.ic_baixo):.1f}" x2="{sx(r.ic_alto):.1f}" y1="{y}" y2="{y}" stroke="{M_COL[m]}" stroke-width="2.4" stroke-linecap="round"/>')
            out.append(f'<circle cx="{sx(r.ponto):.1f}" cy="{y}" r="5.5" fill="{M_COL[m]}" stroke="var(--surface)" stroke-width="2"/>')
            out.append(f'<text x="{sx(r.ponto):.1f}" y="{y-10}" text-anchor="middle">{pt(r.ponto, dd)}</text>')
        ticks = np.linspace(lo, hi, 3)
        out.append(_axis_x(ox + 22, ox + pw - 22, h - 30, ticks, lambda v: pt(v), sx))
    out.append("</svg>")
    return "".join(out)


def svg_steps(ab, steps) -> str:
    """AUC por passo para os três modelos (média ± desvio entre sementes)."""
    W, H, L, R, T, B = 700, 310, 46, 120, 16, 38
    dp = lambda v: 0.0 if (v is None or v != v) else float(v)
    ys = [r.auc + dp(r.auc_dp) for r in ab.itertuples()] + [r.auc - dp(r.auc_dp) for r in ab.itertuples()] + [0.5]
    lo, hi = min(ys) - 0.03, max(ys) + 0.03
    sx = lambda i: L + 10 + i * (W - L - R - 10) / (len(steps) - 1)
    sy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="AUC no teste por passo de correção, três modelos">']
    for v in np.arange(np.ceil(lo * 10) / 10, np.floor(hi * 10) / 10 + 0.01, 0.1):
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)"/><text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{pt(v)}</text>')
    out.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0.5):.1f}" y2="{sy(0.5):.1f}" stroke="var(--ref)" stroke-dasharray="4 3"/><text x="{L+6}" y="{sy(0.5)-5:.1f}">acaso 0,50</text>')
    for i, s in enumerate(steps):
        out.append(f'<text x="{sx(i):.1f}" y="{H-14}" text-anchor="middle">S{i}</text>')
    ends = []
    for m in MODELS:
        d = ab[ab.modelo == m].sort_values("passo_ordem")
        pts = " ".join(f"{sx(r.passo_ordem):.1f},{sy(r.auc):.1f}" for r in d.itertuples())
        out.append(f'<polyline points="{pts}" fill="none" stroke="{M_COL[m]}" stroke-width="2.2" stroke-linejoin="round"/>')
        for r in d.itertuples():
            if r.auc_dp and r.auc_dp == r.auc_dp:
                out.append(f'<line x1="{sx(r.passo_ordem):.1f}" x2="{sx(r.passo_ordem):.1f}" y1="{sy(r.auc-r.auc_dp):.1f}" y2="{sy(r.auc+r.auc_dp):.1f}" stroke="{M_COL[m]}" stroke-width="1.6" opacity=".5"/>')
            out.append(f'<circle cx="{sx(r.passo_ordem):.1f}" cy="{sy(r.auc):.1f}" r="4.5" fill="{M_COL[m]}" stroke="var(--surface)" stroke-width="2"/>')
        last = d.iloc[-1]
        ends.append([m, sy(last.auc)])
    ends.sort(key=lambda e: e[1])
    for k in range(1, len(ends)):
        if ends[k][1] - ends[k - 1][1] < 15:
            ends[k][1] = ends[k - 1][1] + 15
    for m, y in ends:
        out.append(f'<text x="{W-R+10}" y="{y+4:.1f}" style="fill:var(--fg)">{m}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_prev_year(by_year) -> str:
    W, H, L, R, T, B = 820, 270, 44, 10, 28, 34
    n = len(by_year)
    top = max(by_year.y_original.max(), by_year.y_new.max()) * 1.1
    bw = (W - L - R) / n
    sy = lambda v: T + (1 - v / top) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="Taxa de positivos por ano, alvo original e alvo formal">']
    for v in np.linspace(0, top, 5):
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)"/><text x="{L-6}" y="{sy(v)+4:.1f}" text-anchor="end">{round(v*100)} %</text>')
    for i, r in enumerate(by_year.itertuples()):
        x = L + i * bw
        for j, (val, col) in enumerate(((r.y_original, "var(--ref)"), (r.y_new, "var(--m1)"))):
            out.append(f'<rect x="{x + bw*0.12 + j*bw*0.38:.1f}" y="{sy(val):.1f}" width="{bw*0.36:.1f}" height="{H-B-sy(val):.1f}" rx="2" fill="{col}"/>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{H-14}" text-anchor="middle">{str(int(r.year))[2:]}</text>')
    out.append(f'<text x="{L}" y="14" class="t">Participação de dias positivos por ano (’09 a ’25)</text></svg>')
    return "".join(out)


def svg_timeline(parts) -> str:
    """parts: lista de (rótulo, n, cor_css, texto_curto)."""
    W, H, L = 900, 150, 10
    tot = sum(p[1] for p in parts)
    x = L
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="Linha do tempo da divisão: aquecimento, janela, treino, validação, teste e horizonte">']
    spans = []
    for lab, n, col, short in parts:
        w = max((W - 2 * L - 14) * n / tot, 4.0)
        spans.append((x, w))
        out.append(f'<rect x="{x:.1f}" y="30" width="{w:.1f}" height="34" rx="2" fill="{col}"/>')
        if w > 60:
            out.append(f'<text x="{x+w/2:.1f}" y="52" text-anchor="middle" style="fill:#fff;font-weight:600">{short}</text>')
        x += w
    rows_y = [88, 106, 124]
    for i, ((lab, n, col, short), (sx_, sw)) in enumerate(zip(parts, spans)):
        cx = sx_ + sw / 2
        anchor = "start" if i == 0 else ("end" if i == len(parts) - 1 else "middle")
        tx = sx_ if i == 0 else (sx_ + sw if i == len(parts) - 1 else cx)
        yy = rows_y[i % 3]
        out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="66" y2="{yy-12}" stroke="var(--ref)" stroke-width="0.8"/>')
        out.append(f'<text x="{tx:.1f}" y="{yy}" text-anchor="{anchor}">{lab} ({n})</text>')
    out.append("</svg>")
    return "".join(out)


def svg_equity(dates, curves, labels) -> str:
    """curves: dict nome -> (array base 100, cor, dash)."""
    W, H, L, R, T, B = 760, 320, 46, 150, 14, 34
    allv = np.concatenate([c[0] for c in curves.values()])
    lo, hi = np.floor(allv.min() / 5) * 5, np.ceil(allv.max() / 5) * 5
    n = len(dates)
    sx = lambda i: L + i * (W - L - R) / (n - 1)
    sy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="Curvas de patrimônio em base 100 no período de teste">']
    for v in np.linspace(lo, hi, 6):
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)"/><text x="{L-6}" y="{sy(v)+4:.1f}" text-anchor="end">{int(v)}</text>')
    yrs = sorted({d[:4] for d in dates})
    for y in yrs:
        i = next(k for k, d in enumerate(dates) if d.startswith(y))
        out.append(f'<text x="{sx(i):.1f}" y="{H-12}" text-anchor="middle">{y}</text>')
    ends = []
    for name, (arr, col, dash) in curves.items():
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(arr))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{da}/>')
        ends.append([name, sy(arr[-1]), col])
    ends.sort(key=lambda e: e[1])
    for k in range(1, len(ends)):
        if ends[k][1] - ends[k - 1][1] < 14:
            ends[k][1] = ends[k - 1][1] + 14
    for name, y, col in ends:
        out.append(f'<text x="{W-R+8}" y="{y+4:.1f}" style="fill:{col}">{name}</text>')
    out.append("</svg>")
    return "".join(out)
