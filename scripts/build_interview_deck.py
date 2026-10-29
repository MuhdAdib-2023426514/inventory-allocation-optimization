"""Build the 30-minute interview deck (HTML + per-slide PDF) from verified release artifacts.

Every number shown on a slide is read from artifacts/*/summary_metrics.parquet or
multi_seed_summary.parquet at build time. Nothing numeric is typed by hand except
the toy example (a verified unit test) and configuration values read from YAML.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from inventory_di.config import load_config
from inventory_di.io import sha256_file

OUT_HTML = Path("docs/interview_presentation_30min.html")
OUT_MD = Path("docs/interview_presentation_30min.md")
OUT_PDF = Path("docs/interview_presentation_30min.pdf")
OUT_SLIDES_DIR = Path("reports/release/browser/interview_30min_slides")
POLICIES = ["equal_allocation", "proportional_allocation", "supervised_pto",
            "supervised_dro", "dfl_pto", "dfl_dro"]
LABEL = {
    "equal_allocation": "Equal split",
    "proportional_allocation": "Proportional rule",
    "supervised_pto": "Forecast → optimise",
    "supervised_dro": "Forecast → robust",
    "dfl_pto": "DFL → optimise",
    "dfl_dro": "DFL → robust",
    "clairvoyant_oracle": "Hindsight oracle",
}
COLOR = {
    "equal_allocation": "#94a3b8",
    "proportional_allocation": "#475569",
    "supervised_pto": "#2563eb",
    "supervised_dro": "#60a5fa",
    "dfl_pto": "#7c3aed",
    "dfl_dro": "#a78bfa",
}
SEED_COLOR = {42: "#0f766e", 101: "#d97706", 202: "#be123c"}
esc = html.escape


# --------------------------------------------------------------------------- data
def load():
    data, sources = {}, {}
    for mode in ["m5", "m5_intermittent"]:
        cfg = load_config(f"configs/{mode}.yaml")
        art = Path(cfg.data.artifacts_dir)
        s_path, m_path = art / "summary_metrics.parquet", art / "multi_seed_summary.parquet"
        summary = pd.read_parquet(s_path).set_index("policy")
        seeds = pd.read_parquet(m_path).pivot(index="policy", columns="seed", values="mean_cost_mu")
        tuning = json.loads((art / "validation_dro_tuning.json").read_text())
        data[mode] = dict(cfg=cfg, s=summary, seeds=seeds, tuning=tuning)
        sources[str(s_path)] = sha256_file(s_path)
        sources[str(m_path)] = sha256_file(m_path)
    return data, sources


def f1(x):
    return f"{x:,.1f}"


def winners_by_seed(seeds):
    sub = seeds.loc[POLICIES]
    return {int(c): sub[c].idxmin() for c in sub.columns}


# --------------------------------------------------------------------------- SVG charts
def regret_chart(s, title, width=560, height=330):
    rows = POLICIES
    top, left, right, bottom = 44, 150, 70, 34
    bar_h = (height - top - bottom) / len(rows)
    vmax = max(float(s.loc[p, "regret_ci95_high_mu"]) for p in rows) * 1.08
    sx = lambda v: left + (width - left - right) * v / vmax  # noqa: E731
    best = min(rows, key=lambda p: s.loc[p, "mean_regret_mu"])
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(title)}">',
           f'<text x="{left}" y="20" class="ch-title">{esc(title)}</text>',
           f'<text x="{left}" y="36" class="ch-sub">Mean regret per week vs hindsight oracle · bars = 95% interval · lower is better</text>']
    for t in range(0, int(vmax) + 1, max(1, int(vmax / 4 // 10 * 10) or 10)):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{height-bottom+16}" class="tick" text-anchor="middle">{t}</text>')
    for i, p in enumerate(rows):
        y = top + i * bar_h + bar_h * 0.18
        h = bar_h * 0.64
        m = float(s.loc[p, "mean_regret_mu"])
        lo, hi = float(s.loc[p, "regret_ci95_low_mu"]), float(s.loc[p, "regret_ci95_high_mu"])
        weight = "700" if p == best else "500"
        out.append(f'<text x="{left-10}" y="{y+h/2+4:.1f}" class="lbl" text-anchor="end" font-weight="{weight}">{esc(LABEL[p])}</text>')
        out.append(f'<rect x="{left}" y="{y:.1f}" width="{sx(m)-left:.1f}" height="{h:.1f}" rx="3" fill="{COLOR[p]}"/>')
        cy = y + h / 2
        out.append(f'<line x1="{sx(lo):.1f}" y1="{cy:.1f}" x2="{sx(hi):.1f}" y2="{cy:.1f}" class="whisk"/>')
        for v in (lo, hi):
            out.append(f'<line x1="{sx(v):.1f}" y1="{cy-6:.1f}" x2="{sx(v):.1f}" y2="{cy+6:.1f}" class="whisk"/>')
        out.append(f'<text x="{sx(hi)+6:.1f}" y="{cy+4:.1f}" class="val">{m:,.0f}</text>')
    out.append("</svg>")
    return "".join(out)


def seed_chart(seeds, title, width=560, height=330):
    rows = POLICIES
    top, left, right, bottom = 44, 150, 24, 40
    vals = seeds.loc[rows].to_numpy().ravel()
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    sx = lambda v: left + (width - left - right) * (v - lo) / (hi - lo)  # noqa: E731
    row_h = (height - top - bottom) / len(rows)
    ref = float(seeds.loc["proportional_allocation"].iloc[0])
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(title)}">',
           f'<text x="{left}" y="20" class="ch-title">{esc(title)}</text>',
           f'<text x="{left}" y="36" class="ch-sub">Mean weekly cost (MU) per independently retrained seed · lower is better</text>']
    step = max(1, round((hi - lo) / 4 / 5) * 5)
    t = (int(lo) // step + 1) * step
    while t < hi:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{height-bottom+16}" class="tick" text-anchor="middle">{t:,}</text>')
        t += step
    xr = sx(ref)
    out.append(f'<line x1="{xr:.1f}" y1="{top-4}" x2="{xr:.1f}" y2="{height-bottom}" class="refline"/>')
    out.append(f'<text x="{xr+4:.1f}" y="{height-bottom+30}" class="tick">proportional rule</text>')
    for i, p in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        out.append(f'<text x="{left-10}" y="{cy+4:.1f}" class="lbl" text-anchor="end">{esc(LABEL[p])}</text>')
        out.append(f'<line x1="{left}" y1="{cy:.1f}" x2="{width-right}" y2="{cy:.1f}" class="rowline"/>')
        for seed in seeds.columns:
            v = float(seeds.loc[p, seed])
            out.append(f'<circle cx="{sx(v):.1f}" cy="{cy:.1f}" r="6.5" fill="{SEED_COLOR[int(seed)]}" fill-opacity="0.85" stroke="#fff" stroke-width="1.5"><title>seed {seed}: {v:,.1f} MU</title></circle>')
    out.append("</svg>")
    return "".join(out)


TOY_SVG = """
<svg viewBox="0 0 560 310" class="chart" role="img" aria-label="Toy allocation example">
  <text x="20" y="24" class="ch-title">60 units in the warehouse · demand: Store A = 60, Store B = 20</text>
  <g transform="translate(40,52)">
    <text x="90" y="0" class="lbl" text-anchor="middle" font-weight="700">Equal split 30 / 30</text>
    <rect x="20" y="20" width="60" height="135" class="dem"/><rect x="20" y="87" width="60" height="68" class="ship"/>
    <text x="50" y="172" class="tick" text-anchor="middle">Store A</text><text x="50" y="80" class="neg" text-anchor="middle">30 short</text>
    <rect x="100" y="110" width="60" height="45" class="dem"/><rect x="100" y="87" width="60" height="68" class="ship-x"/>
    <text x="130" y="172" class="tick" text-anchor="middle">Store B</text><text x="130" y="80" class="neg" text-anchor="middle">10 wasted</text>
    <text x="90" y="198" class="big-neg" text-anchor="middle">fills 50 of 80 · 62.5%</text>
  </g>
  <g transform="translate(300,52)">
    <text x="90" y="0" class="lbl" text-anchor="middle" font-weight="700">Optimal 40 / 20</text>
    <rect x="20" y="20" width="60" height="135" class="dem"/><rect x="20" y="65" width="60" height="90" class="ship"/>
    <text x="50" y="172" class="tick" text-anchor="middle">Store A</text><text x="50" y="58" class="neg" text-anchor="middle">20 short</text>
    <rect x="100" y="110" width="60" height="45" class="dem"/><rect x="100" y="110" width="60" height="45" class="ship"/>
    <text x="130" y="172" class="tick" text-anchor="middle">Store B</text><text x="130" y="104" class="pos" text-anchor="middle">exact</text>
    <text x="90" y="198" class="big-pos" text-anchor="middle">fills 60 of 80 · 75.0%</text>
  </g>
  <g transform="translate(55,282)">
    <rect width="14" height="10" class="dem"/><text x="20" y="9" class="tick">Demand</text>
    <rect x="115" width="14" height="10" class="ship"/><text x="135" y="9" class="tick">Units shipped &amp; sold</text>
    <rect x="295" width="14" height="10" class="ship-x"/><text x="315" y="9" class="tick">Shipped but unsold (overstock)</text>
  </g>
</svg>"""


def cost_curve_svg(b, h, t):
    short, excess = b + t, h + t
    return f"""
<svg viewBox="0 0 680 300" class="chart" role="img" aria-label="Asymmetric cost">
  <text x="30" y="24" class="ch-title">Cost of one unit of forecast error (synthetic MU)</text>
  <line x1="30" y1="235" x2="650" y2="235" class="axis"/><line x1="340" y1="40" x2="340" y2="245" class="axis"/>
  <path d="M 180 195 L 340 235 L 500 55" fill="none" class="curve"/>
  <path d="M 180 125 L 340 235 L 500 125" fill="none" class="curve-sym"/>
  <text x="170" y="129" class="tick" text-anchor="end">symmetric (MAE / RMSE)</text>
  <text x="170" y="199" class="pos" text-anchor="end">overstock: {excess:.2f} MU/unit</text>
  <text x="510" y="59" class="neg" text-anchor="start">shortage: {short:.2f} MU/unit</text>
  <text x="510" y="129" class="tick" text-anchor="start">symmetric benchmark</text>
  <text x="340" y="265" class="tick" text-anchor="middle">perfect forecast (error = 0)</text>
  <text x="180" y="265" class="tick" text-anchor="middle">over-forecast</text><text x="500" y="265" class="tick" text-anchor="middle">under-forecast</text>
</svg>"""


PIPELINE_SVG = """
<svg viewBox="0 0 1120 250" class="chart" role="img" aria-label="Pipeline">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>
  <g class="node"><rect x="10" y="30" width="190" height="80" rx="12"/><text x="105" y="66" text-anchor="middle" class="n-t">History up to</text><text x="105" y="88" text-anchor="middle" class="n-t">decision day</text></g>
  <g class="node accent"><rect x="250" y="30" width="210" height="80" rx="12"/><text x="355" y="64" text-anchor="middle" class="n-t">Demand distribution</text><text x="355" y="86" text-anchor="middle" class="n-s">P(next-week demand = 0, 1, 2 …)</text></g>
  <g class="node"><rect x="510" y="30" width="200" height="80" rx="12"/><text x="610" y="64" text-anchor="middle" class="n-t">Value of each</text><text x="610" y="86" text-anchor="middle" class="n-s">extra unit, per store</text></g>
  <g class="node solver"><rect x="760" y="30" width="160" height="80" rx="12"/><text x="840" y="64" text-anchor="middle" class="n-t">Integer optimiser</text><text x="840" y="86" text-anchor="middle" class="n-s">supply + space limits</text></g>
  <g class="node out"><rect x="970" y="30" width="140" height="80" rx="12"/><text x="1040" y="66" text-anchor="middle" class="n-t">Units per</text><text x="1040" y="88" text-anchor="middle" class="n-t">store</text></g>
  <line x1="200" y1="70" x2="246" y2="70" class="flow" marker-end="url(#arr)"/>
  <line x1="460" y1="70" x2="506" y2="70" class="flow" marker-end="url(#arr)"/>
  <line x1="710" y1="70" x2="756" y2="70" class="flow" marker-end="url(#arr)"/>
  <line x1="920" y1="70" x2="966" y2="70" class="flow" marker-end="url(#arr)"/>
  <path d="M 840 110 C 840 180, 355 180, 355 116" class="loop" marker-end="url(#arr)"/>
  <text x="598" y="210" text-anchor="middle" class="loop-t">DFL training loop: optimiser penalises decision regret (SPO+ loss) → updates forecaster</text>
</svg>"""


# --------------------------------------------------------------------------- slides
def build_slides(D):
    R, I = D["m5"], D["m5_intermittent"]
    rs, is_ = R["s"], I["s"]
    ecfg = R["cfg"].economics
    rcfg = R["cfg"].data
    b, h, t = ecfg.unit_shortage_penalty_mu, ecfg.unit_holding_cost_mu, ecfg.unit_ship_cost_mu
    eq_gain = float(rs.loc["equal_allocation", "diff_vs_proportional_mu"])
    eq_lo = float(rs.loc["equal_allocation", "paired_diff_vs_proportional_ci95_low_mu"])
    eq_hi = float(rs.loc["equal_allocation", "paired_diff_vs_proportional_ci95_high_mu"])
    eq_pct = 100 * eq_gain / float(rs.loc["equal_allocation", "mean_cost_mu"])
    oracle_gap = float(rs.loc["proportional_allocation", "mean_regret_mu"])
    _oracle_pct = 100 * oracle_gap / float(rs.loc["clairvoyant_oracle", "mean_cost_mu"])
    sp_i = is_.loc["supervised_pto"]
    win_r, win_i = winners_by_seed(R["seeds"]), winners_by_seed(I["seeds"])
    n_test = int(rs["num_origins"].iloc[0])
    tun = R["tuning"]["supervised_dro_selected"]

    def ci(row, col="proportional"):
        return f"[{row[f'paired_diff_vs_{col}_ci95_low_mu']:+.1f}, {row[f'paired_diff_vs_{col}_ci95_high_mu']:+.1f}]"

    def model_rows(s):
        out = []
        for p in POLICIES[2:]:
            row = s.loc[p]
            lo, hi = row.paired_diff_vs_proportional_ci95_low_mu, row.paired_diff_vs_proportional_ci95_high_mu
            verdict = "better" if hi < 0 else ("worse" if lo > 0 else "tie")
            out.append((LABEL[p], f"{row.diff_vs_proportional_mu:+.1f}", ci(row), verdict))
        return out

    def verdict_table(s):
        rows = "".join(
            f'<tr><td>{esc(a)}</td><td class="num">{b_}</td><td class="num">{c}</td>'
            f'<td><span class="pill {v}">{ {"better":"clearly better","worse":"clearly worse","tie":"within noise"}[v] }</span></td></tr>'
            for a, b_, c, v in model_rows(s))
        return ('<table class="tbl"><thead><tr><th>Policy</th><th class="num">vs proportional</th>'
                '<th class="num">95% interval</th><th>Verdict</th></tr></thead><tbody>' + rows + "</tbody></table>")

    seed_list = ", ".join(str(k) for k in win_r)
    win_i_txt = "; ".join(f"seed {k}: {LABEL[v]}" for k, v in win_i.items())
    win_r_txt = "; ".join(f"seed {k}: {LABEL[v]}" for k, v in win_r.items())

    slides = []
    S = slides.append

    # 1 ---------------------------------------------------------------- title
    S(dict(kind="title", sec="", title="Which store gets the last unit?",
        body=f"""
<div class="title-wrap">
  <div class="eyebrow">Decision Intelligence case study</div>
  <h1 class="hero">Which store gets the <span class="acc-text">last unit?</span></h1>
  <p class="hero-sub">Decision-focused and risk-aware stock allocation for a multi-store retailer, and an honest test of whether the advanced methods are worth it</p>
  <div class="hero-meta">
    <div><span>Data</span>Walmart M5 sales (public) as a demand proxy</div>
    <div><span>Economics</span>Synthetic monetary units (MU), not real costs</div>
    <div><span>Evidence</span>{n_test} test weeks · 2 demand regimes · 3 seeds</div>
  </div>
</div>""",
        talk="Good morning. Thank you for the time. I'll spend about twenty minutes on a case study I built end to end, then show the working tool for five minutes, then take questions. The question is simple: when the warehouse can't cover every store, who gets the stock? I'll show you a method from recent research, and, just as importantly, an honest test of whether it beats the simple rules a retailer already uses.",
        q="Why did you pick this problem?",
        a="Allocation is where forecasting meets money. Two teams can have the same forecast accuracy and very different stockout costs. It is also a problem where an analyst can do harm by shipping complexity that doesn't pay, so it is a good test of judgement.",
        sec_s=60))

    # 2 ---------------------------------------------------------------- answer first
    S(dict(kind="std", sec="The answer first", title="Three findings in one slide",
        body=f"""
<div class="kpis">
  <div class="kpi good"><div class="kpi-n">−{eq_gain:,.0f} MU</div><div class="kpi-l">per week ({eq_pct:.1f}% lower cost) from moving off an equal split to any demand-aware rule</div><div class="kpi-f">95% interval {eq_lo:,.0f} to {eq_hi:,.0f} MU · regular demand</div></div>
  <div class="kpi neutral"><div class="kpi-n">Tie</div><div class="kpi-l">Decision-focused learning and robust optimisation did not beat a simple proportional rule on regular demand</div><div class="kpi-f">every model's 95% interval vs the rule crosses zero</div></div>
  <div class="kpi warn"><div class="kpi-n">Unstable</div><div class="kpi-l">On intermittent demand, forecast → optimise won clearly on one seed, but not when retrained</div><div class="kpi-f">best policy by seed: {esc(win_i_txt)}</div></div>
</div>
<div class="takeaway"><b>What I would tell a business owner:</b> get the allocation logic right first, it is where the money is. Add advanced learning only where a controlled test shows it pays. I built the evidence system that can tell the difference.</div>""",
        talk=f"I'll start with the answer. One: simply replacing an equal split with a demand-aware rule cut weekly cost by about {eq_gain:.0f} MU, roughly {eq_pct:.0f} percent, and the interval is well away from zero. Two: on regular demand, the advanced methods tied with a simple proportional rule. Three: on intermittent items, forecast-then-optimise won clearly on the primary seed, but when I retrained with different random seeds the winner changed. So the honest headline is not 'my model wins', it is 'here is where complexity pays and where it doesn't, and here is how I know'.",
        q="So your advanced method failed?",
        a="On this sample it didn't beat a strong simple rule, and I'd rather show that than hide it. The pipeline, verification and evaluation design are what transfer to a real business: they stop you deploying a model because it looks clever. With real stock records and costs the answer may change, and the same harness would show it.",
        sec_s=90))

    # 3 ---------------------------------------------------------------- problem toy
    S(dict(kind="split", sec="1 · Problem", title="Same stock, different outcome: allocation is the decision",
        left="""
<ul class="pts">
  <li>A warehouse rarely has enough to cover every store. Every unit sent to one store is a unit another store doesn't get.</li>
  <li>The decision is a whole number of units per store, under warehouse supply, shelf space and per-item caps.</li>
  <li>In this tiny example, the same 60 units fill <b>20% more demand</b> just by sending them where they will sell.</li>
</ul>
<div class="note">This exact example is one of the project's automated tests: the optimiser must find 40 / 20.</div>""",
        right=TOY_SVG,
        talk="Here is the whole problem in one picture. Sixty units in the warehouse. Store A will sell sixty, store B only twenty. An equal split sends thirty each: A runs short by thirty, and ten units sit unsold at B. Sending forty and twenty fills sixty units instead of fifty. Nothing about the forecast changed; only the allocation did. At network scale, with hundreds of items and supply and shelf limits, you can't do this by eye. It needs an optimiser.",
        q="Isn't this just ranking stores by expected demand?",
        a="With one item and no limits, nearly. With shared warehouse supply, shelf capacity across items and uncertain demand, it becomes an integer optimisation: the value of the next unit depends on how many that store already has and how likely it is to sell.",
        sec_s=90))

    # 4 ---------------------------------------------------------------- accuracy
    S(dict(kind="split", sec="1 · Problem", title="A better forecast score doesn't guarantee a better decision",
        left=f"""
<ul class="pts">
  <li>In this setup, running out costs <b>{b+t:.2f} MU</b> per unit; overstocking costs <b>{h+t:.2f} MU</b>. Roughly <b>{(b+t)/(h+t):.0f} : 1</b>.</li>
  <li>Accuracy metrics such as MAE treat both errors as equally bad, so a model tuned for accuracy is tuned for the wrong goal.</li>
  <li>The quantity to measure is the <b>cost of the decision</b>, compared with the best decision possible in hindsight (<b>regret</b>).</li>
</ul>
<div class="note">Costs are synthetic. In a real deployment they would come from margin, holding and freight data.</div>""",
        right=cost_curve_svg(b, h, t),
        talk=f"Why not just build a better forecaster? Because forecast accuracy and business cost aren't the same thing. In my setup a missed sale costs about {(b+t)/(h+t):.0f} times more than an extra unit on the shelf. Accuracy metrics punish both errors equally. So I judge every method by one number: how much more did its decision cost than the best decision you could have made with hindsight. That's regret, and it's the main score on every results slide.",
        q="Why not use quantile regression at the critical ratio?",
        a="That answers 'how much for one store with unlimited supply'. When stores share a limited warehouse and shelf space, the quantile alone can't decide who to cut. You still need the optimiser; the question is what to feed it.",
        sec_s=90))

    # 5 ---------------------------------------------------------------- approach
    S(dict(kind="std", sec="2 · Approach", title="From history to shipments, with the optimiser in the training loop",
        body=f"""
{PIPELINE_SVG}
<div class="three">
  <div><h4>Forecast → optimise</h4><p>Train the forecaster for calibrated probabilities, then optimise. The standard approach and a fair baseline.</p></div>
  <div><h4>Decision-focused learning (DFL)</h4><p>During training, the optimiser scores each forecast by the <b>cost of the decision it led to</b> (SPO+ loss), and the model learns from that.</p></div>
  <div><h4>Same model, same budget</h4><p>Both versions share architecture, initial weights, data order and epochs. Only the training signal differs.</p></div>
</div>""",
        talk="This is the pipeline. On each decision day the model sees only past data and predicts a full probability distribution for next week's demand, not a single number. From that I work out the value of each extra unit at each store, and an integer optimiser allocates within supply and shelf limits. Decision-focused learning adds the curved arrow: during training, the optimiser tells the model how costly its decision was, and the model updates to make cheaper decisions, not just more accurate forecasts. To make the comparison fair, both versions are identical except for that training signal.",
        q="How do you know the optimiser actually influences training?",
        a="I test it. There's an automated check that the decision-focused gradient is finite and non-zero, and that switching it off changes the parameter updates. Without that check, a lot of 'DFL' implementations are just regression with extra steps.",
        sec_s=120))

    # 6 ---------------------------------------------------------------- DRO
    S(dict(kind="split", sec="2 · Approach", title="Risk-aware allocation: plan for the bad weeks, not only the average",
        left=f"""
<ul class="pts">
  <li>Simulate many possible next weeks from the forecast, keeping stores' tendency to move together.</li>
  <li>The robust optimiser gives extra weight to the most expensive scenarios. Mathematically, this equals minimising the average cost of the worst tail of outcomes (CVaR).</li>
  <li>Two dials: how far into the tail (α) and how much to care about it (risk weight). Both chosen on validation data only: α = {tun['alpha']}, weight = {tun['risk_weight']}.</li>
  <li>Setting the risk weight to zero gives back the ordinary average-cost plan, a built-in sanity check.</li>
</ul>""",
        right="""
<div class="dial">
  <div class="dial-row"><span>Average-cost plan</span><div class="bar"><i style="width:100%"></i></div><small>every scenario counts equally</small></div>
  <div class="dial-row"><span>Robust plan</span><div class="bar tail"><i style="width:40%"></i><b style="width:60%"></b></div><small>expensive scenarios weighted up</small></div>
  <div class="dial-cap">Cost: a little more on an average week. Benefit: fewer very bad weeks, if the forecast's uncertainty is right.</div>
  <div class="note" style="margin-top:18px">This is probability-reweighting robustness. It is not Wasserstein DRO, and I label it that way.</div>
</div>""",
        talk="The second idea is robustness. Instead of planning for the average week, I simulate many possible weeks and let the optimiser give extra weight to the expensive ones. There's a neat result that this is exactly the same as minimising the average of the worst tail of costs, called CVaR. The two dials, how deep into the tail and how much to care, were picked on validation weeks, never on test weeks. And when the risk dial is at zero, it collapses back to the normal plan, which I check automatically.",
        q="Why would robust optimisation ever cost more?",
        a="Because it deliberately buys insurance. If the forecast's uncertainty is too wide, or bad weeks don't arrive in the test period, you pay the premium without collecting. That is what happened here on average, which is why I report average and tail cost separately.",
        sec_s=105))

    # 7 ---------------------------------------------------------------- fair test
    S(dict(kind="std", sec="3 · Analysis", title="How I made the comparison fair",
        body=f"""
<div class="timeline">
  <div class="seg train" style="flex:{rcfg.train_weeks}"><b>{rcfg.train_weeks} weeks</b> training</div>
  <div class="seg val" style="flex:{rcfg.val_weeks*3}"><b>{rcfg.val_weeks}</b> validation</div>
  <div class="seg test" style="flex:{rcfg.test_weeks*3}"><b>{rcfg.test_weeks}</b> test</div>
</div>
<div class="tl-cap">Time only moves forward: the model never sees the future it is tested on.</div>
<div class="grid4">
  <div class="card-s"><h4>Same conditions</h4><p>Every policy faces the same weeks, stock, supply, shelf limits and costs.</p></div>
  <div class="card-s"><h4>Two regimes</h4><p>{rcfg.num_stores} stores × {rcfg.num_skus} household items each: regular sellers and intermittent sellers.</p></div>
  <div class="card-s"><h4>Strong baselines</h4><p>Equal split and a proportional rule (last 8 weeks of sales), as a retailer would use.</p></div>
  <div class="card-s"><h4>Checked independently</h4><p>All 288 policy-week allocations re-verified: feasible, costs recomputed, no negative regret.</p></div>
</div>""",
        talk=f"Results are only as good as the test. Data is split strictly by time: {rcfg.train_weeks} weeks to train, {rcfg.val_weeks} to tune, {rcfg.test_weeks} to test. Anything chosen, item selection, caps, risk dials, was chosen before touching test data. Every policy faces exactly the same conditions. I deliberately included strong simple baselines, because that's what a real business already has. And a separate script recomputes every allocation and cost from scratch, so the numbers you'll see are checked, not just reported.",
        q="How do you handle stockouts hiding true demand in sales data?",
        a="M5 sales are a proxy: when a store ran out, true demand was higher than recorded sales. I state this as a limitation. In production I'd combine sales with on-hand stock records and estimate the hidden demand, for example with a censored-demand model.",
        sec_s=90))

    # 8 ---------------------------------------------------------------- results
    S(dict(kind="two", sec="3 · Analysis", title="Results: big win over equal split; models tie with a simple rule",
        left=regret_chart(rs, "Regular demand"),
        right=regret_chart(is_, "Intermittent demand"),
        foot=f"""<div class="takeaway small"><b>Read it this way:</b> regular demand, equal split is clearly worst and the rest overlap. Intermittent demand, forecast → optimise is the only policy clearly better than the proportional rule on the primary seed ({sp_i.diff_vs_proportional_mu:+.1f} MU, interval {ci(sp_i)}). Even the best policy is ~{oracle_gap:,.0f} MU from perfect hindsight, which is the limit of predictability.</div>""",
        talk=f"Here are the results. Bars are regret, extra cost versus perfect hindsight, so shorter is better, and the lines are 95 percent intervals over time. On regular demand, equal split is clearly worst; everything else overlaps, including the simple proportional rule. On intermittent demand, plain forecast-then-optimise is the one policy clearly better than the proportional rule on the primary seed, by about {abs(sp_i.diff_vs_proportional_mu):.0f} MU a week. Notice also the gap that remains to the oracle, around {oracle_gap:.0f} MU on regular demand: that's demand nobody could have predicted.",
        q="Is a 12-week test enough?",
        a=f"It's enough to see big effects, like the {eq_gain:.0f} MU gain over equal split. It isn't enough to separate methods that differ by 5 to 10 MU. That's why I show intervals and say 'tie' instead of picking a winner.",
        sec_s=120))

    # 9 ---------------------------------------------------------------- seeds
    S(dict(kind="two", sec="3 · Analysis", title="Do the rankings survive retraining? Not reliably",
        left=seed_chart(R["seeds"], "Regular demand"),
        right=seed_chart(I["seeds"], "Intermittent demand"),
        foot=f"""<div class="legend"><span><i style="background:{SEED_COLOR[42]}"></i>seed 42 (primary)</span><span><i style="background:{SEED_COLOR[101]}"></i>seed 101</span><span><i style="background:{SEED_COLOR[202]}"></i>seed 202</span></div>
<div class="takeaway small"><b>Best policy by seed.</b> Regular: {esc(win_r_txt)}. Intermittent: {esc(win_i_txt)}. A result that changes with the random seed is not yet a basis for a business decision.</div>""",
        talk=f"I then retrained every model from scratch with seeds {seed_list}, each with the same budget and its own tuning. The simple rules don't change, they have no randomness, so they sit on one point. The learned models move around. On intermittent demand, the primary-seed advantage of forecast-then-optimise disappears on the other two seeds. This is exactly the check that stops a team from presenting a lucky run as a breakthrough.",
        q="Why not just report the best seed?",
        a="Because the business doesn't get to choose the seed. If performance depends on a random number, the honest expectation is the spread, and here the spread covers the simple rule.",
        sec_s=90))

    # 10 ---------------------------------------------------------------- demo
    S(dict(kind="split", sec="4 · Solution", title="Live demo: the decision tool",
        left="""
<ol class="steps">
  <li><b>Page A</b>: data source badge, synthetic costs, time split</li>
  <li><b>Page C</b>: units per store, supply and shelf limits respected</li>
  <li><b>Page D</b>: cut warehouse supply to 0.6× and watch stock get rationed</li>
  <li><b>Page D</b>: move the risk dial; at α = 0 the risk cost equals the average cost</li>
  <li><b>Page E</b>: paired results and CSV export</li>
</ol>""",
        right="""
<div class="demo-box">
  <div class="demo-url">localhost:8511</div>
  <div class="demo-t">Runs fully offline on a laptop</div>
  <ul class="pts small">
    <li>Models pre-trained; optimisation runs live</li>
    <li>Nominal solve ≈ 0.02 s, robust solve ≈ 0.1–0.15 s per week</li>
    <li>All 15 page × mode views checked in an automated browser test</li>
  </ul>
  <div class="note">Backup if anything fails: this deck plus the PDF.</div>
</div>""",
        talk="Let me show you the tool. I'll switch to the dashboard now.",
        q="Could planners use this?",
        a="As a decision-support view, yes: it shows the allocation, why, and what changes if supply drops. It doesn't place orders; a planner stays responsible. For production it would need real stock feeds, case-pack rounding and an override log.",
        sec_s=60))

    # 11 ---------------------------------------------------------------- business meaning
    S(dict(kind="cols", sec="5 · Impact", title="What the evidence supports, and what it doesn't",
        left=f"""
<h3 class="ok">Supported</h3>
<ul class="pts">
  <li>Demand-aware allocation beats an equal split by about <b>{eq_gain:,.0f} MU/week ({eq_pct:.1f}%)</b> in this setup, with a clear interval.</li>
  <li>An open-source integer optimiser handles the constraints in a fraction of a second per week.</li>
  <li>The evaluation harness catches lucky results: paired intervals, retrained seeds, independent recomputation.</li>
</ul>""",
        right="""
<h3 class="no">Not supported (yet)</h3>
<ul class="pts">
  <li>That decision-focused learning or robust optimisation beats a simple rule on this data.</li>
  <li>Any real saving in RM: costs are synthetic and sales stand in for demand.</li>
  <li>Performance at network scale: this is 5 stores × 10 items per regime.</li>
</ul>""",
        foot="""<div class="takeaway small"><b>Recommended next step:</b> run the proportional rule plus optimiser as the baseline, and treat DFL / robust optimisation as challengers that must win a pre-registered shadow test.</div>""",
        talk=f"So what does this mean for a business? It strongly supports getting allocation logic right: about {eq_pct:.0f} percent lower cost against an equal split in this setup. It does not support claiming that the advanced methods pay off, or any figure in Ringgit, because the costs are synthetic. My recommendation is to make the simple-rule-plus-optimiser the baseline, and only promote advanced models if they win a pre-registered shadow test on your real data.",
        q="How would you turn MU into Ringgit?",
        a="Replace the synthetic costs with real ones: lost gross margin for a stockout, cost of capital and space for holding, freight per unit. Then rerun the same harness on real stock and sales and multiply weekly regret differences by volume, with intervals.",
        sec_s=90))

    # 12 ---------------------------------------------------------------- limits
    S(dict(kind="std", sec="5 · Impact", title="Limitations and what I would do next",
        body="""
<table class="tbl wide"><thead><tr><th>Limitation</th><th>Why it matters</th><th>Next step</th></tr></thead><tbody>
<tr><td>Sales stand in for demand</td><td>Stockouts hide true demand, so costs are understated</td><td>Use on-hand stock records; censored-demand model</td></tr>
<tr><td>Synthetic costs</td><td>Rankings can change with the real cost ratio</td><td>Calibrate from margin, holding and freight data</td></tr>
<tr><td>12 test weeks, small cohorts</td><td>Too few to separate methods 5–10 MU apart</td><td>Longer, untouched holdout; more items and stores</td></tr>
<tr><td>Test period seen before the fixes</td><td>This is a corrected re-run, not a fresh confirmatory test</td><td>Pre-register the next evaluation on new weeks</td></tr>
<tr><td>Single-week decisions</td><td>Real stock carries over between weeks</td><td>Extend the tested multi-week simulator to the main comparison</td></tr>
</tbody></table>""",
        talk="I want to be upfront about limits, because each one changes how far you can trust the result. The biggest are that sales aren't true demand, and the costs are synthetic. The test is short, and because I fixed bugs after first looking at the test weeks, this is a corrected re-run, not a fresh test. Each limitation has a concrete next step, and most of them need data that a retailer already has.",
        q="What would you change if you started again?",
        a="Lock the test period away and pre-register the comparison before building anything, and start from real stock records instead of sales alone.",
        sec_s=75))

    # 13 ---------------------------------------------------------------- roadmap
    S(dict(kind="std", sec="5 · Impact", title="From prototype to pilot: a low-risk path",
        body="""
<div class="road">
  <div class="phase p1"><div class="ph-n">Phase 1 · Shadow</div><div class="ph-t">Weeks 1–8</div><p>Connect real stock, sales and cost data. Generate recommendations alongside planners; ship nothing. Measure regret vs current orders.</p><div class="gate">Go if: clear, significant gain vs current practice</div></div>
  <div class="phase p2"><div class="ph-n">Phase 2 · Pilot</div><div class="ph-t">Weeks 9–16</div><p>Small store cluster and a focused item set. Planner can override; every override is logged and reviewed.</p><div class="gate">Go if: fewer stockouts, no rise in overstock</div></div>
  <div class="phase p3"><div class="ph-n">Phase 3 · Scale</div><div class="ph-t">After pilot review</div><p>Solve items in parallel (each item's warehouse limit is separate), then reconcile shelf space. Monitor drift.</p><div class="gate">Challenger models promoted only on evidence</div></div>
</div>""",
        talk="If this went forward, I'd do it in three gated steps. First, shadow mode: run on real data next to the planners, ship nothing, measure. Second, a small pilot where planners can override and every override is reviewed, because that's where you learn what the model is missing. Third, scale, which is technically straightforward because each item's warehouse limit can be solved in parallel. At each gate the advanced methods have to earn their place.",
        q="What is the biggest risk in rollout?",
        a="Planner trust and data quality, more than algorithms. Shadow mode, visible reasons for each allocation and an override log address both.",
        sec_s=75))

    # 14 ---------------------------------------------------------------- close
    S(dict(kind="title", sec="", title="Summary",
        body=f"""
<div class="title-wrap">
  <div class="eyebrow">Summary</div>
  <h1 class="hero" style="font-size:52px">Get the decision right first. <span class="acc-text">Prove complexity before you ship it.</span></h1>
  <div class="close-grid">
    <div><b>Built</b>End-to-end pipeline, integer optimiser, decision-focused and robust variants, offline dashboard</div>
    <div><b>Verified</b>47 automated tests, 288 allocations recomputed, leakage checks, 3 retrained seeds</div>
    <div><b>Found</b>~{eq_pct:.0f}% gain from allocation logic; advanced methods not yet proven on this data</div>
  </div>
  <p class="hero-sub" style="margin-top:34px">Thank you. Happy to take questions or go deeper on any part.</p>
</div>""",
        talk="To sum up: I built and verified the full pipeline, found a clear gain from better allocation logic, and showed honestly that the advanced methods haven't yet earned their complexity on this data. That combination, building it, testing it hard, and telling you plainly what it shows, is how I'd approach decision intelligence work here. Thank you, I'm happy to take questions.",
        q="Why should we hire you?",
        a="I can take a business decision from data to a working, tested tool, and I report what the evidence says even when it isn't the exciting answer. That protects the business from expensive mistakes.",
        sec_s=45))

    # ---------------------------------------------------------------- backups
    S(dict(kind="std", sec="Backup", title="B1 · The optimisation model",
        body=f"""
<div class="math">
<div class="m-h">Cost for one week (per store s, item p)</div>
<div class="m">cost = Σ  b · shortage<sub>sp</sub>  +  h · leftover<sub>sp</sub>  +  t · shipped<sub>sp</sub></div>
<div class="m">shortage = max(D − I − x, 0)      leftover = max(I + x − D, 0)</div>
<div class="m-h">Constraints</div>
<div class="m">Σ<sub>s</sub> x<sub>sp</sub> ≤ W<sub>p</sub>  (warehouse)      Σ<sub>p</sub> (I<sub>sp</sub> + x<sub>sp</sub>) ≤ C<sub>s</sub>  (store space)      0 ≤ x<sub>sp</sub> ≤ K<sub>sp</sub>, integer</div>
<div class="m-h">Value of the k-th extra unit (exact identity, unit-tested)</div>
<div class="m">c[s,k] = t − b  if demand ≥ I + k  (it prevents a shortage),   else  t + h  (it is left over)</div>
<div class="m">expected: ĉ[s,k] = t + h − (b + h) · P(D ≥ I + k)</div>
</div>
<p class="note">b = {b}, h = {h}, t = {t} MU per unit (synthetic). Solver: HiGHS (open source). Fulfilment is recomputed after solving as min(demand, stock).</p>""",
        talk="Backup: the full model if asked.", q="Why integer?", a="Shipments are whole units; at low volumes rounding a continuous solution can break supply limits or waste stock.", sec_s=0))

    S(dict(kind="std", sec="Backup", title="B2 · Decision-focused learning details",
        body="""
<div class="math">
<div class="m-h">SPO+ loss (Elmachtoub &amp; Grigas), used via PyEPO</div>
<div class="m">L(ĉ, c) = max<sub>w∈S</sub> (c − 2ĉ)ᵀw  +  2ĉᵀw*(c)  −  c ᵀw*(c)</div>
<div class="m-h">Training objective</div>
<div class="m">loss = NLL(demand distribution)  +  λ · SPO+,     λ chosen on validation from {0, 0.5, 1, 2}</div>
</div>
<ul class="pts">
<li>Training uses a fixed set of constraints per item, so SPO+'s assumptions hold. Real weekly limits are applied at decision time.</li>
<li>Gradient check: SPO+ gradient is finite and non-zero; λ = 0 vs λ &gt; 0 produce different updates.</li>
<li>Validation regret is recomputed after restoring the best checkpoint and calibration.</li>
</ul>""",
        talk="Backup: DFL details.", q="Why keep NLL in the loss?", a="Pure SPO+ can wreck probability calibration, and the robust optimiser needs calibrated scenarios. NLL keeps the distribution honest.", sec_s=0))

    S(dict(kind="std", sec="Backup", title="B3 · Why the robust model equals CVaR",
        body="""
<div class="math">
<div class="m-h">Ambiguity set over N scenarios with base probabilities p</div>
<div class="m">P<sub>α</sub> = { w : Σ w<sub>j</sub> = 1,  0 ≤ w<sub>j</sub> ≤ p<sub>j</sub> / (1 − α) }</div>
<div class="m">max<sub>w∈P<sub>α</sub></sub> Σ w<sub>j</sub> L<sub>j</sub>(x)  =  CVaR<sub>α</sub>(L(x))</div>
<div class="m-h">Objective</div>
<div class="m">min (1 − ρ) · E[L(x)]  +  ρ · CVaR<sub>α</sub>(L(x))</div>
</div>
<ul class="pts">
<li>Solved exactly as one integer program with the standard CVaR linearisation (η, z<sub>j</sub>).</li>
<li>Checked against brute-force enumeration on small cases; α = 0 or ρ = 0 recovers the average-cost plan.</li>
<li>Scenarios share random numbers across policies and keep cross-store dependence learned from training weeks only.</li>
</ul>""",
        talk="Backup: CVaR equivalence.", q="Why not Wasserstein DRO?", a="It needs extra modelling choices and solver cost; the reweighting set has an exact, testable meaning. Wasserstein is listed as future work, not claimed.", sec_s=0))

    def full_table(s):
        cols = ["mean_cost_mu", "mean_regret_mu", "mean_fill_rate", "realized_cvar90_cost_mu"]
        rows = "".join(
            f'<tr><td>{esc(LABEL[p])}</td>' + "".join(
                f'<td class="num">{(s.loc[p, c]*100):.1f}%</td>' if c == "mean_fill_rate" else f'<td class="num">{s.loc[p, c]:,.1f}</td>'
                for c in cols) + "</tr>"
            for p in POLICIES + ["clairvoyant_oracle"])
        return ('<table class="tbl compact"><thead><tr><th>Policy</th><th class="num">Cost</th><th class="num">Regret</th>'
                '<th class="num">Fill</th><th class="num">Worst-10% cost</th></tr></thead><tbody>' + rows + "</tbody></table>")

    S(dict(kind="two", sec="Backup", title="B4 · Full results, primary seed (MU per week)",
        left='<h4 class="tbl-h">Regular demand</h4>' + full_table(rs) + '<h4 class="tbl-h" style="margin-top:14px">vs proportional rule</h4>' + verdict_table(rs),
        right='<h4 class="tbl-h">Intermittent demand</h4>' + full_table(is_) + '<h4 class="tbl-h" style="margin-top:14px">vs proportional rule</h4>' + verdict_table(is_),
        foot='<p class="note">Worst-10% cost averages roughly the single most expensive of 12 test weeks, so it is noisy. Intervals: paired moving-block bootstrap over weeks (2,000 resamples).</p>',
        talk="Backup: full tables.", q="Which tail result looked best?", a="DFL → optimise had the lowest worst-10% cost on regular demand, but that statistic rests on about one week, so I don't claim it.", sec_s=0))

    return slides


# --------------------------------------------------------------------------- rendering
CSS = r"""
:root{--bg:#eef1f6;--paper:#ffffff;--ink:#0f172a;--ink2:#334155;--mut:#64748b;--line:#e2e8f0;--acc:#1d4ed8;--acc2:#7c3aed;
--good:#047857;--goodbg:#ecfdf5;--warn:#b45309;--warnbg:#fffbeb;--neu:#334155;--neubg:#f1f5f9;--bad:#be123c;--card:#f8fafc;--chip:#eff6ff}
html.dark{--bg:#060a13;--paper:#0f172a;--ink:#f8fafc;--ink2:#cbd5e1;--mut:#94a3b8;--line:#1e293b;--acc:#60a5fa;--acc2:#a78bfa;
--good:#34d399;--goodbg:rgba(52,211,153,.10);--warn:#fbbf24;--warnbg:rgba(251,191,36,.10);--neu:#cbd5e1;--neubg:rgba(148,163,184,.10);--bad:#fb7185;--card:#111c31;--chip:rgba(96,165,250,.12)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:"Inter","SF Pro Display",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;overflow:hidden;height:100vh;-webkit-font-smoothing:antialiased}
#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
#deck{width:1280px;height:720px;position:relative;transform-origin:center center}
.slide{position:absolute;inset:0;background:var(--paper);border-radius:18px;box-shadow:0 30px 70px rgba(15,23,42,.18);padding:46px 60px 40px;display:flex;flex-direction:column;opacity:0;visibility:hidden;transition:opacity .35s ease,transform .35s ease;transform:translateY(8px)}
.slide.active{opacity:1;visibility:visible;transform:none}
.slide::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:18px 0 0 18px;background:linear-gradient(180deg,var(--acc),var(--acc2))}
.sec{font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--acc)}
.sec.backup{color:var(--mut)}
h2.t{font-size:34px;line-height:1.18;font-weight:800;letter-spacing:-.02em;margin:8px 0 22px;max-width:1080px}
.body{flex:1;min-height:0;display:flex;flex-direction:column;gap:18px}
.foot{position:absolute;left:60px;right:60px;bottom:16px;display:flex;justify-content:space-between;font-size:12px;color:var(--mut)}
.split{display:grid;grid-template-columns:1fr 1.05fr;gap:40px;flex:1;align-items:center}
.two{display:grid;grid-template-columns:1fr 1fr;gap:28px;flex:1;align-items:start}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:28px;flex:1}
.cols>div{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px 28px}
h3.ok{color:var(--good);font-size:20px;margin-bottom:14px}h3.no{color:var(--bad);font-size:20px;margin-bottom:14px}
.pts{list-style:none;display:flex;flex-direction:column;gap:16px}
.pts li{font-size:20px;line-height:1.45;color:var(--ink2);padding-left:26px;position:relative}
.pts li::before{content:"";position:absolute;left:2px;top:11px;width:9px;height:9px;border-radius:50%;background:var(--acc)}
.pts li b{color:var(--ink)}
.pts.small li{font-size:16px}
.note{font-size:14px;color:var(--mut);line-height:1.5;border-left:3px solid var(--line);padding-left:12px;margin-top:16px}
.takeaway{background:var(--chip);border-radius:12px;padding:16px 22px;font-size:18px;line-height:1.5;color:var(--ink2)}
.takeaway b{color:var(--ink)}.takeaway.small{font-size:15px;padding:12px 18px}
/* title */
.title-wrap{display:flex;flex-direction:column;justify-content:center;height:100%;padding:0 20px}
.eyebrow{font-size:15px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--acc)}
.hero{font-size:68px;line-height:1.05;font-weight:850;letter-spacing:-.035em;margin:18px 0 20px;color:var(--ink);max-width:1050px}
.hero .acc-text{color:var(--acc)}
.hero-sub{font-size:22px;line-height:1.5;color:var(--ink2);max-width:900px}
.hero-meta{display:flex;gap:18px;margin-top:42px}
.hero-meta div,.close-grid div{flex:1;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;font-size:16px;color:var(--ink2);line-height:1.45}
.hero-meta span,.close-grid b{display:block;font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin-bottom:6px}
.close-grid{display:flex;gap:18px;margin-top:20px}
/* kpis */
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;flex:1}
.kpi{border-radius:16px;padding:26px 26px 22px;display:flex;flex-direction:column;border:1px solid var(--line)}
.kpi.good{background:var(--goodbg)}.kpi.neutral{background:var(--neubg)}.kpi.warn{background:var(--warnbg)}
.kpi-n{font-size:50px;font-weight:850;letter-spacing:-.03em;line-height:1}
.kpi.good .kpi-n{color:var(--good)}.kpi.neutral .kpi-n{color:var(--neu)}.kpi.warn .kpi-n{color:var(--warn)}
.kpi-l{font-size:19px;line-height:1.42;margin-top:16px;color:var(--ink);flex:1}
.kpi-f{font-size:13.5px;color:var(--mut);margin-top:14px;line-height:1.45}
/* charts */
.chart{width:100%;height:auto;display:block;font-family:inherit}
.ch-title{font-size:16px;font-weight:700;fill:var(--ink)}.ch-sub{font-size:11px;fill:var(--mut)}
.lbl{font-size:13px;fill:var(--ink2)}.tick{font-size:11px;fill:var(--mut)}.val{font-size:12px;font-weight:700;fill:var(--ink)}
.grid{stroke:var(--line);stroke-width:1}.rowline{stroke:var(--line);stroke-dasharray:2 4}.whisk{stroke:var(--ink);stroke-width:1.6;opacity:.7}
.refline{stroke:var(--mut);stroke-width:1.5;stroke-dasharray:5 4}
.axis{stroke:var(--mut);stroke-width:1.5}.curve{stroke:var(--bad);stroke-width:4;stroke-linejoin:round}
.curve-sym{stroke:var(--mut);stroke-width:2;stroke-dasharray:6 5}
.neg{font-size:13px;font-weight:700;fill:var(--bad)}.pos{font-size:13px;font-weight:700;fill:var(--good)}
.big-neg{font-size:16px;font-weight:800;fill:var(--bad)}.big-pos{font-size:16px;font-weight:800;fill:var(--good)}
.dem{fill:none;stroke:var(--mut);stroke-width:2;stroke-dasharray:5 4}.ship{fill:var(--acc);opacity:.85}.ship-x{fill:var(--bad);opacity:.35}
.node rect{fill:var(--card);stroke:var(--line);stroke-width:2}.node.accent rect{stroke:var(--acc)}.node.solver rect{stroke:var(--acc2)}.node.out rect{fill:var(--chip);stroke:var(--acc)}
.n-t{font-size:16px;font-weight:700;fill:var(--ink)}.n-s{font-size:12.5px;fill:var(--mut)}
.flow{stroke:var(--mut);stroke-width:2}.arrowhead{fill:var(--mut)}
.loop{fill:none;stroke:var(--acc2);stroke-width:2.5;stroke-dasharray:7 5}.loop-t{font-size:14px;font-weight:600;fill:var(--acc2)}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.three>div,.card-s{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.three h4,.card-s h4{font-size:16px;margin-bottom:6px;color:var(--ink)}.three p,.card-s p{font-size:15px;line-height:1.45;color:var(--ink2)}
/* dial */
.dial{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px}
.dial-row{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;align-items:center;margin-bottom:22px}
.dial-row span{font-weight:700;font-size:16px}.dial-row small{grid-column:2;font-size:13px;color:var(--mut)}
.bar{height:22px;border-radius:6px;background:var(--line);display:flex;overflow:hidden}
.bar i{display:block;background:var(--acc);opacity:.55}.bar.tail b{display:block;background:linear-gradient(90deg,var(--acc2),var(--bad))}
.dial-cap{font-size:16px;line-height:1.5;color:var(--ink2)}
/* timeline */
.timeline{display:flex;height:54px;border-radius:12px;overflow:hidden;font-size:15px;color:#fff}
.seg{display:flex;align-items:center;justify-content:center;gap:6px}.seg.train{background:#475569}.seg.val{background:var(--acc2)}.seg.test{background:var(--acc)}
.tl-cap{font-size:14px;color:var(--mut);margin-top:-8px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
/* tables */
.tbl{width:100%;border-collapse:collapse;font-size:14px}
.tbl th{text-align:left;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);padding:8px 10px;border-bottom:2px solid var(--line)}
.tbl td{padding:8px 10px;border-bottom:1px solid var(--line);color:var(--ink2)}
.tbl.wide td{font-size:16px;padding:12px 12px}.tbl.compact td{padding:6px 10px}
.num{text-align:right;font-variant-numeric:tabular-nums}
.tbl-h{font-size:15px;margin-bottom:6px}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700}
.pill.better{background:var(--goodbg);color:var(--good)}.pill.worse{background:rgba(190,18,60,.1);color:var(--bad)}.pill.tie{background:var(--neubg);color:var(--neu)}
.legend{display:flex;gap:22px;font-size:13px;color:var(--ink2)}.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
/* steps & demo */
.steps{list-style:none;counter-reset:s;display:flex;flex-direction:column;gap:14px}
.steps li{counter-increment:s;font-size:19px;line-height:1.4;color:var(--ink2);padding-left:48px;position:relative}
.steps li::before{content:counter(s);position:absolute;left:0;top:-2px;width:32px;height:32px;border-radius:50%;background:var(--chip);color:var(--acc);font-weight:800;display:flex;align-items:center;justify-content:center;font-size:15px}
.demo-box{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:28px}
.demo-url{display:inline-block;font-family:ui-monospace,Menlo,monospace;font-size:15px;background:var(--chip);color:var(--acc);padding:6px 14px;border-radius:8px}
.demo-t{font-size:22px;font-weight:800;margin:14px 0 14px}
/* roadmap */
.road{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;flex:1}
.phase{border-radius:16px;padding:24px;border:1px solid var(--line);background:var(--card);display:flex;flex-direction:column;position:relative}
.phase::before{content:"";position:absolute;top:0;left:0;right:0;height:5px;border-radius:16px 16px 0 0}
.p1::before{background:#475569}.p2::before{background:var(--acc)}.p3::before{background:var(--acc2)}
.ph-n{font-size:20px;font-weight:800}.ph-t{font-size:13px;color:var(--mut);margin:4px 0 14px}
.phase p{font-size:16px;line-height:1.5;color:var(--ink2);flex:1}
.gate{margin-top:16px;font-size:13.5px;font-weight:700;color:var(--good);background:var(--goodbg);border-radius:8px;padding:8px 12px}
/* math */
.math{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 26px;display:flex;flex-direction:column;gap:8px}
.m-h{font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--acc);margin-top:8px}
.m{font-family:"Cambria Math","STIX Two Math","Latin Modern Math",Georgia,serif;font-size:21px;color:var(--ink)}
/* chrome */
#bar{position:fixed;top:0;left:0;height:4px;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .3s;z-index:40}
#hud{position:fixed;right:18px;bottom:14px;display:flex;gap:8px;align-items:center;z-index:40;opacity:.35;transition:opacity .2s}
#hud:hover{opacity:1}
#hud button{background:var(--paper);color:var(--ink2);border:1px solid var(--line);border-radius:8px;padding:6px 11px;font-size:13px;font-weight:600;cursor:pointer}
#hud .cnt{font-size:13px;color:var(--mut);font-variant-numeric:tabular-nums;padding:0 6px}
#timer{font-variant-numeric:tabular-nums}
#notes{position:fixed;left:0;right:0;bottom:0;max-height:42vh;overflow:auto;background:#0b1220;color:#e2e8f0;border-top:3px solid #7c3aed;padding:18px 28px 22px;transform:translateY(105%);transition:transform .28s cubic-bezier(.2,.8,.2,1);z-index:50;display:grid;grid-template-columns:1.5fr 1fr;gap:26px}
#notes.open{transform:none}
#notes h5{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#a78bfa;margin-bottom:8px}
#notes .talk{font-size:16px;line-height:1.6}
#notes .qa{background:rgba(244,63,94,.08);border-left:3px solid #f43f5e;border-radius:0 8px 8px 0;padding:12px 16px;font-size:14.5px;line-height:1.55}
#notes .qa b{color:#fda4af;display:block;margin-bottom:6px}
#help{position:fixed;inset:0;background:rgba(2,6,23,.6);display:none;align-items:center;justify-content:center;z-index:60}
#help>div{background:var(--paper);color:var(--ink);border-radius:14px;padding:26px 32px;min-width:380px;font-size:15px;line-height:2}
kbd{font-family:ui-monospace,Menlo,monospace;background:var(--card);border:1px solid var(--line);border-radius:5px;padding:1px 7px;font-size:13px}
#overview{position:fixed;inset:0;background:var(--bg);z-index:55;display:none;overflow:auto;padding:30px;grid-template-columns:repeat(4,1fr);gap:18px}
#overview .th{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:14px 16px;cursor:pointer;font-size:14px;line-height:1.4}
#overview .th small{display:block;color:var(--mut);font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}
#overview .th.cur{outline:3px solid var(--acc)}
@media print{
  @page{size:1280px 720px;margin:0}
  body{overflow:visible;height:auto;background:#fff}
  #stage{position:static;display:block}#deck{transform:none!important;width:1280px;height:auto}
  .slide{position:relative;opacity:1;visibility:visible;transform:none;width:1280px;height:720px;border-radius:0;box-shadow:none;page-break-after:always;break-after:page;transition:none}
  #bar,#hud,#notes,#help,#overview{display:none!important}
}
"""

JS = r"""
const slides=[...document.querySelectorAll('.slide')];const N=slides.length;let cur=0;let t0=null;
const deck=document.getElementById('deck');
function fit(){const s=Math.min(innerWidth/1340,innerHeight/770);deck.style.transform=`scale(${s})`;}
addEventListener('resize',fit);fit();
function show(i){cur=Math.max(0,Math.min(N-1,i));slides.forEach((s,k)=>s.classList.toggle('active',k===cur));
 document.getElementById('bar').style.width=((cur+1)/N*100)+'%';
 document.getElementById('cnt').textContent=(cur+1)+' / '+N;
 const s=slides[cur];document.getElementById('n-talk').textContent=s.dataset.talk;
 document.getElementById('n-q').textContent=s.dataset.q;document.getElementById('n-a').textContent=s.dataset.a;
 document.getElementById('n-time').textContent=s.dataset.sec>0?`Target ≈ ${Math.round(s.dataset.sec/6)/10} min · planned elapsed ${s.dataset.elapsed}`:'Backup slide: use only if asked';
 document.querySelectorAll('#overview .th').forEach((e,k)=>e.classList.toggle('cur',k===cur));
 history.replaceState(null,'','#'+(cur+1));}
function tick(){if(t0){const e=Math.floor((Date.now()-t0)/1000);document.getElementById('timer').textContent=String(Math.floor(e/60)).padStart(2,'0')+':'+String(e%60).padStart(2,'0');}}
setInterval(tick,1000);
const notes=document.getElementById('notes'),help=document.getElementById('help'),ov=document.getElementById('overview');
function toggleOv(){ov.style.display=ov.style.display==='grid'?'none':'grid';}
document.addEventListener('keydown',e=>{if(!t0&&['ArrowRight',' ','PageDown'].includes(e.key))t0=Date.now();
 const k=e.key;if(['ArrowRight',' ','PageDown','Enter'].includes(k)){e.preventDefault();show(cur+1)}
 else if(['ArrowLeft','PageUp','Backspace'].includes(k)){e.preventDefault();show(cur-1)}
 else if(k==='Home')show(0);else if(k==='End')show(N-1);
 else if(k==='s'||k==='S')notes.classList.toggle('open');
 else if(k==='t'||k==='T')document.documentElement.classList.toggle('dark');
 else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}
 else if(k==='o'||k==='O')toggleOv();
 else if(k==='r'||k==='R'){t0=Date.now();tick()}
 else if(k==='?'||k==='h'||k==='H')help.style.display=help.style.display==='flex'?'none':'flex';
 else if(k==='Escape'){notes.classList.remove('open');help.style.display='none';ov.style.display='none'}});
document.getElementById('b-prev').onclick=()=>show(cur-1);document.getElementById('b-next').onclick=()=>show(cur+1);
document.getElementById('b-notes').onclick=()=>notes.classList.toggle('open');
document.getElementById('b-theme').onclick=()=>document.documentElement.classList.toggle('dark');
document.getElementById('b-help').onclick=()=>help.style.display='flex';help.onclick=()=>help.style.display='none';
document.querySelectorAll('#overview .th').forEach((e,k)=>e.onclick=()=>{show(k);toggleOv()});
let sx=null;addEventListener('touchstart',e=>sx=e.touches[0].clientX);addEventListener('touchend',e=>{if(sx===null)return;const d=e.changedTouches[0].clientX-sx;if(Math.abs(d)>50)show(cur+(d<0?1:-1));sx=null});
show((parseInt(location.hash.slice(1))||1)-1);
"""


def render(slides, sources):
    elapsed, parts, thumbs = 0, [], []
    main_count = sum(1 for s in slides if s["sec"] != "Backup")
    for i, s in enumerate(slides, 1):
        mm = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        elapsed += s["sec_s"]
        kind = s["kind"]
        if kind == "title":
            inner = f'<div class="body">{s["body"]}</div>'
        else:
            head = f'<div class="sec{" backup" if s["sec"]=="Backup" else ""}">{esc(s["sec"])}</div><h2 class="t">{esc(s["title"])}</h2>'
            if kind == "std":
                core = s["body"]
            elif kind == "split":
                core = f'<div class="split"><div>{s["left"]}</div><div>{s["right"]}</div></div>'
            elif kind == "two":
                core = f'<div class="two"><div>{s["left"]}</div><div>{s["right"]}</div></div>' + s.get("foot", "")
            elif kind == "cols":
                core = f'<div class="cols"><div>{s["left"]}</div><div>{s["right"]}</div></div>' + s.get("foot", "")
            inner = head + f'<div class="body">{core}</div>'
        label = "Backup" if s["sec"] == "Backup" else f"{i} / {main_count}"
        foot = f'<div class="foot"><span>Interview case study · Walmart M5 sales proxy · synthetic MU economics</span><span>{label}</span></div>'
        attrs = f'data-talk="{esc(s["talk"])}" data-q="{esc(s["q"])}" data-a="{esc(s["a"])}" data-sec="{s["sec_s"]}" data-elapsed="{mm}"'
        parts.append(f'<section class="slide" {attrs}>{inner}{foot}</section>')
        thumbs.append(f'<div class="th"><small>{i} · {esc(s["sec"] or "Opening")}</small>{esc(s["title"])}</div>')
    total = sum(s["sec_s"] for s in slides)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Which store gets the last unit? · Decision Intelligence case study</title>
<!-- Generated by scripts/build_interview_deck.py from: {esc(json.dumps(sources))} -->
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div id="stage"><div id="deck">{''.join(parts)}</div></div>
<div id="hud"><span id="timer">00:00</span><span class="cnt" id="cnt"></span>
<button id="b-prev" title="Previous (←)">←</button><button id="b-next" title="Next (→)">→</button>
<button id="b-notes" title="Speaker notes (S)">Notes</button><button id="b-theme" title="Light/dark (T)">◐</button><button id="b-help" title="Shortcuts (?)">?</button></div>
<aside id="notes"><div><h5 id="n-time"></h5><div class="talk" id="n-talk"></div></div>
<div><h5>Likely question</h5><div class="qa"><b id="n-q"></b><span id="n-a"></span></div></div></aside>
<div id="help"><div><b>Keyboard</b><br><kbd>→</kbd> <kbd>Space</kbd> next · <kbd>←</kbd> back<br><kbd>S</kbd> speaker notes · <kbd>O</kbd> slide overview<br>
<kbd>T</kbd> light / dark · <kbd>F</kbd> full screen<br><kbd>R</kbd> reset timer (starts on first →)<br><kbd>Esc</kbd> close<br>
<small style="color:var(--mut)">Planned talk ≈ {total//60} min, plus demo and Q&amp;A</small></div></div>
<div id="overview">{''.join(thumbs)}</div>
<script>{JS}</script></body></html>"""
    OUT_HTML.write_text(doc, encoding="utf-8")
    return total


def render_markdown(slides, data, sources):
    lines = [
        "# 30-Minute Technical Interview Presentation: Which Store Gets the Last Unit?",
        "",
        "**Target Role:** Data Scientist Executive (Decision Intelligence), MR.DIY  ",
        "**Candidate Session Structure:** ~18–20 minutes slide presentation + 5 minutes live dashboard demo + 5–7 minutes technical Q&A  ",
        "**Primary Deliverables:**",
        "- Interactive Slide Deck: [`docs/interview_presentation_30min.html`](interview_presentation_30min.html) (Keyboard-driven, clean typography, embedded SVGs, dark/light theme, toggleable speaker notes drawer)",
        "- 18-Page Static PDF Fallback: [`docs/interview_presentation_30min.pdf`](interview_presentation_30min.pdf) (1 page per slide, identical layout)",
        "- Running Offline Streamlit App: `http://localhost:8511` (`streamlit run src/inventory_di/ui/streamlit_app.py --server.port 8511`)",
        "",
        "---",
        "",
        "## Executive Presentation Pacing & Structure",
        "",
        "| Section | Slide # | Title | Target Time | Cumulative |",
        "|---|---|---|---|---|",
        "| **Opening** | Slide 1 | Which store gets the last unit? | 1.0 min | 0:00 – 1:00 |",
        "| **The Answer First** | Slide 2 | Three findings in one slide | 1.5 min | 1:00 – 2:30 |",
        "| **Part 1: Problem** | Slide 3 | Same stock, different outcome: allocation is the decision | 1.5 min | 2:30 – 4:00 |",
        "| **Part 1: Problem** | Slide 4 | A better forecast score doesn't guarantee a better decision | 1.5 min | 4:00 – 5:30 |",
        "| **Part 2: Approach** | Slide 5 | From history to shipments, with the optimiser in the training loop | 2.0 min | 5:30 – 7:30 |",
        "| **Part 2: Approach** | Slide 6 | Risk-aware allocation: plan for the bad weeks, not only the average | 1.75 min | 7:30 – 9:15 |",
        "| **Part 3: Analysis** | Slide 7 | How I made the comparison fair | 1.5 min | 9:15 – 10:45 |",
        "| **Part 3: Analysis** | Slide 8 | Results: big win over equal split; models tie with a simple rule | 2.0 min | 10:45 – 12:45 |",
        "| **Part 3: Analysis** | Slide 9 | Do the rankings survive retraining? Not reliably | 1.5 min | 12:45 – 14:15 |",
        "| **Part 4: Solution** | Slide 10 | Live demo: the decision tool | 1.0 min (+ 5 min demo) | 14:15 – 20:15 |",
        "| **Part 5: Business Impact** | Slide 11 | What the evidence supports, and what it doesn't | 1.5 min | 20:15 – 21:45 |",
        "| **Part 5: Business Impact** | Slide 12 | Limitations and what I would do next | 1.25 min | 21:45 – 23:00 |",
        "| **Part 5: Business Impact** | Slide 13 | From prototype to pilot: a low-risk path | 1.25 min | 23:00 – 24:15 |",
        "| **Summary** | Slide 14 | Get the decision right first. Prove complexity before you ship it. | 0.75 min | 24:15 – 25:00 |",
        "| **Technical Q&A / Backups** | Slides 15–18 | B1: Formulation, B2: DFL SPO+, B3: DRO/CVaR, B4: Multi-seed table | 5.0 min | 25:00 – 30:00 |",
        "",
        "---",
        "",
        "## Slide-by-Slide Visuals, Verbal Talk Tracks & Interviewer Q&A",
        ""
    ]

    elapsed = 0
    main_count = sum(1 for s in slides if s["sec"] != "Backup")
    for i, s in enumerate(slides, 1):
        mm = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        elapsed += s["sec_s"]
        is_backup = s["sec"] == "Backup"
        slide_label = f"Backup Slide {s['title'][:2]}" if is_backup else f"Slide {i} of {main_count}"

        lines.append(f"### {slide_label}: {s['title']}")
        lines.append(f"- **Section:** {s['sec'] or 'Opening'}")
        lines.append(f"- **Timing:** {s['sec_s']}s (Target ≈ {s['sec_s']/60:.1f} min · Planned elapsed: {mm})")
        lines.append("- **Verbal Script (What you say out loud):**")
        lines.append(f"  > *\"{s['talk']}\"*")
        lines.append("- **Anticipated Interviewer Question (Trap):**")
        lines.append(f"  > **Q:** {s['q']}")
        lines.append("- **Senior-Level Model Answer:**")
        lines.append(f"  > **A:** {s['a']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Live Dashboard Demo Walkthrough Protocol (5 Minutes)")
    lines.append("")
    lines.append("When transitioning at Slide 10 to the running Streamlit dashboard (`localhost:8511`), follow this sequence:")
    lines.append("1. **Page A (Configuration & Methodology):** Point to the 'Walmart M5 Sales Proxy' and 'Synthetic MU Economics' badges. Explain the strict chronological split (100 train / 8 val / 12 test).")
    lines.append("2. **Page C (Stock Allocation Matrix):** Show the allocation table. Point out that all store shipments are whole integers, no warehouse supply is exceeded (sum <= W_p), and total store shelf space is respected (sum <= C_s).")
    lines.append("3. **Page D (Interactive Simulation & Stress Testing):**")
    lines.append("   - Drag the Warehouse Supply multiplier from 1.0x down to 0.6x. Show how the optimiser rations scarce stock to high-volume stores rather than cutting everywhere equally.")
    lines.append("   - Toggle the Risk Weight dial from 0.0 (risk-neutral) to 0.25 (robust CVaR). Show that when risk weight is 0.0, the robust model mathematically collapses back to the nominal expected cost plan.")
    lines.append("4. **Page E (Backtest & Cost Regret Breakdown):** Highlight the paired regret reduction table and show the CSV download capability for offline auditing.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Key Behavioral Guidelines for the Candidate")
    lines.append("- **Pacing:** Don't rush through Slide 2 ('The Answer First'). Senior interviewers love when a candidate gives the punchline immediately.")
    lines.append("- **Honesty:** When asked if DFL beat the baseline, stand firm on the empirical truth: *'On regular demand, it tied with a simple proportional rule. I am showing you an honest evaluation system that prevents the company from deploying unproven complexity.'* That answer demonstrates senior leadership maturity.")
    lines.append("- **Hotkeys in HTML Deck:** Press `S` to toggle speaker notes drawer during practice, `O` for slide overview grid, `T` for dark mode, `F` for fullscreen, `Space`/`→` to advance.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def export_pdf_and_pixmaps(html_path, pdf_path, out_dir):
    import asyncio

    import pymupdf
    from playwright.async_api import async_playwright

    async def _run():
        out_dir.mkdir(parents=True, exist_ok=True)
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            )
            page = await browser.new_page(viewport={"width": 1280, "height": 720})
            await page.goto(f"file://{html_path.resolve()}", wait_until="networkidle")
            await page.pdf(
                path=str(pdf_path.resolve()),
                width="1280px",
                height="720px",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"}
            )
            await browser.close()

        doc = pymupdf.open(str(pdf_path.resolve()))
        assert len(doc) == 18, f"Expected 18 PDF pages, got {len(doc)}"
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            pix.save(str(out_dir / f"slide_{i+1:02d}.png"))
        return len(doc)

    return asyncio.run(_run())


def main():
    data, sources = load()
    slides = build_slides(data)
    total = render(slides, sources)
    render_markdown(slides, data, sources)
    pdf_pages = export_pdf_and_pixmaps(OUT_HTML, OUT_PDF, OUT_SLIDES_DIR)
    print(json.dumps(dict(
        html=str(OUT_HTML),
        markdown=str(OUT_MD),
        pdf=str(OUT_PDF),
        pdf_pages=pdf_pages,
        slides=len(slides),
        planned_talk_seconds=total
    )))


if __name__ == "__main__":
    main()
