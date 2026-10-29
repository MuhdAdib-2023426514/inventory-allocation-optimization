"""Isolated browser QA with non-loopback HTTP blocked; no user browser reuse."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from playwright.sync_api import expect, sync_playwright

from inventory_di.config import load_config

MODES = {"demo": "Demo Mode (DEMO / SYNTHETIC)", "m5": "M5 Mode (M5 Sales Proxy)",
         "m5_intermittent": "M5 Intermittent Mode (Lumpy/Intermittent SKUs)"}
PAGES = ["Page A — Problem & Data", "Page B — Models & Diagnostics", "Page C — Allocation Results",
         "Page D — Risk & Robustness", "Page E — Evidence & Business Impact"]


def settle(page):
    page.wait_for_timeout(350)
    page.wait_for_function("document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') === 'notRunning'", timeout=120000)
    assert page.get_by_test_id("stException").count() == 0
    assert page.get_by_text("This experiment could not be evaluated:", exact=False).count() == 0


def run(modes, slides=False):
    out = Path("reports/release/browser")
    out.mkdir(parents=True, exist_ok=True)
    result = dict(external_http_blocked=True, external_attempts=[], page_errors=[], console_errors=[], pages=[], controls=[], downloads=[])
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--disable-background-networking"])
        context = browser.new_context(viewport=dict(width=1440, height=1000), accept_downloads=True, service_workers="block")
        def route(request):
            host = urlparse(request.request.url).hostname
            if host and host not in {"localhost", "127.0.0.1", "::1"}:
                result["external_attempts"].append(request.request.url)
                request.abort()
            else:
                request.continue_()
        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
        page.on("console", lambda msg: result["console_errors"].append(msg.text) if msg.type == "error" else None)
        page.goto("http://localhost:8511")
        settle(page)
        sidebar = page.get_by_test_id("stSidebar")
        for mode in modes:
            sidebar.get_by_text(MODES[mode], exact=True).click()
            settle(page)
            for label in PAGES:
                started = time.perf_counter()
                sidebar.get_by_text(label, exact=True).click()
                settle(page)
                expect(page.get_by_role("heading", name=label, exact=True)).to_be_visible()
                expect(page.get_by_text("Provenance Badge:", exact=False)).to_be_visible()
                result["pages"].append(dict(mode=mode, page=label, seconds=time.perf_counter()-started))
                page.screenshot(path=str(out / f"{mode}_{label[5]}.png"), full_page=True)
                if label == PAGES[2]:
                    with page.expect_download() as download:
                        page.get_by_role("button", name="Download Allocation Results CSV for Selected Origin").click()
                    saved = out / f"{mode}_allocations.csv"
                    download.value.save_as(saved)
                    csv = pd.read_csv(saved)
                    cfg = load_config(f"configs/{mode}.yaml")
                    allocations = pd.read_parquet(Path(cfg.data.artifacts_dir) / "allocation_results.parquet")
                    expected = allocations[allocations.forecast_origin == csv.forecast_origin.iloc[0]]
                    keys = ["policy", "store_id", "sku_id"]
                    cols = keys + ["allocated_units", "opening_stock_units", "warehouse_supply_units", "is_feasible", "solver_status"]
                    pd.testing.assert_frame_equal(csv[cols].sort_values(keys).reset_index(drop=True),
                                                  expected[cols].sort_values(keys).reset_index(drop=True), check_dtype=False)
                    assert csv[["model_expected_shortage_units", "model_expected_excess_units"]].notna().all().all()
                    result["downloads"].append(str(saved))
                if label == PAGES[3]:
                    for control in ["Tail Ambiguity Parameter (alpha)", "Tail Risk Weight (risk_weight)", "Warehouse Supply Scarcity Multiplier"]:
                        slider = page.get_by_test_id("stSlider").filter(has_text=control).get_by_role("slider")
                        slider.press("Home")
                        settle(page)
                        result["controls"].append(dict(mode=mode, control=control, action="minimum"))
                    metric_values = page.get_by_test_id("stMetricValue").all_text_contents()
                    assert metric_values[0] == metric_values[1], "alpha=0 must make CVaR equal expected loss"
                    expect(page.get_by_text("1. Model-Implied Expected Cost (MU)", exact=True)).to_be_visible()
                    expect(page.get_by_text("3. Realized Historical Test Cost (MU)", exact=True)).to_be_visible()
        if slides:
            page.goto(Path("docs/presentation_slides.html").resolve().as_uri())
            expect(page.locator(".slide")).to_have_count(7)
            for i, slide in enumerate(page.locator(".slide").all(), 1):
                fit = slide.evaluate("el => ({height:el.scrollHeight, client:el.clientHeight, last:el.querySelector('li:last-child').getBoundingClientRect().bottom, footer:el.querySelector('footer').getBoundingClientRect().top})")
                assert fit["height"] <= fit["client"] and fit["last"] < fit["footer"]-8, (i, fit)
                slide.screenshot(path=str(out / f"slide_{i}.png"))
            page.pdf(path="docs/presentation_slides.pdf", width="1280px", height="720px", print_background=True)
            result["slides_fit"] = True
        context.close()
        browser.close()
    result["status"] = "PASS" if not result["page_errors"] and not result["console_errors"] else "FAIL"
    path = out / ("qa_" + "_".join(modes) + ".json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), action="append")
    parser.add_argument("--slides", action="store_true")
    args = parser.parse_args()
    run(args.mode or list(MODES), args.slides)
