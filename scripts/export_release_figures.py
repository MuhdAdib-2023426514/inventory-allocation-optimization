"""Export reproducible Plotly figures as offline HTML, SVG and PNG."""

from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from playwright.sync_api import sync_playwright

from inventory_di.config import load_config
from inventory_di.visualization.plots import plot_policy_cost_comparison


def main():
    paths = []
    for mode in ["demo", "m5", "m5_intermittent"]:
        cfg = load_config(f"configs/{mode}.yaml")
        summary = pd.read_parquet(Path(cfg.data.artifacts_dir) / "summary_metrics.parquet")
        fig = plot_policy_cost_comparison(summary)
        fig.update_layout(width=1250, height=760, margin=dict(l=65, r=25, t=110, b=230),
                          legend=dict(orientation="h", x=0, y=1.15),
                          title=f"{cfg.run_id}: held-out cost and origin-tail CVaR90 (synthetic MU)")
        path = Path(cfg.data.reports_dir) / "policy_costs.html"
        fig.write_html(path, include_plotlyjs=True, full_html=True)
        paths.append(path)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=dict(width=1300, height=850))
        for path in paths:
            page.goto(path.resolve().as_uri())
            page.wait_for_function("document.querySelector('.js-plotly-plot')?._fullLayout !== undefined")
            page.locator(".js-plotly-plot").screenshot(path=str(path.with_suffix(".png")))
            svg = page.evaluate("async () => await Plotly.toImage(document.querySelector('.js-plotly-plot'), {format:'svg',width:1250,height:760})")
            path.with_suffix(".svg").write_text(unquote(svg.split(",",1)[1]))
        browser.close()
    print("Exported", len(paths), "offline HTML/SVG/PNG figure sets")


if __name__ == "__main__":
    main()
