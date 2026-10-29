"""Exercise the real app's pages, boundary controls and service modes via AppTest."""

import argparse
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

MODES = {"demo": "Demo Mode (DEMO / SYNTHETIC)", "m5": "M5 Mode (M5 Sales Proxy)",
         "m5_intermittent": "M5 Intermittent Mode (Lumpy/Intermittent SKUs)"}


def run(modes):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py"), default_timeout=120).run()
    rows = []
    def check(mode, action):
        errors = [e.message for e in app.exception] + [e.value for e in app.error]
        rows.append(dict(mode=mode, action=action, errors=errors))
        assert not errors, rows[-1]
    for mode in modes:
        app.radio[0].set_value(MODES[mode]).run()
        for page in app.radio[1].options:
            app.radio[1].set_value(page).run()
            check(mode,page)
        app.radio[1].set_value("Page D — Risk & Robustness").run()
        cases = [("page_d_alpha", 0.), ("page_d_rw", 0.), ("page_d_supply", .6),
                 ("page_d_open_stock", 5), ("page_d_promo_lift", 50), ("page_d_shortage_mult", 2.),
                 ("page_d_alpha", .95), ("page_d_rw", 1.), ("page_d_supply", 1.2)]
        for key,value in cases:
            app.select_slider(key=key).set_value(value).run()
            check(mode,f"{key}={value}")
        assert "3. Simulated Uplift Test Cost (MU)" in [m.label for m in app.metric]
        for service in ["soft", "hard"]:
            app.selectbox(key="service_mode").set_value(service).run()
            check(mode,f"service={service}")
    result = dict(status="PASS", checks=rows, count=len(rows))
    Path("reports/release/app_qa_"+"_".join(modes)+".json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result["status"],checks=result["count"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), action="append")
    args = parser.parse_args()
    run(args.mode or list(MODES))
