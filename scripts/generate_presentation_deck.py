"""Build seven consistent interview slides and notes from corrected release artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from inventory_di.config import load_config
from inventory_di.io import sha256_file

LABELS = dict(equal_allocation="Equal allocation", proportional_allocation="Proportional allocation",
              supervised_pto="Supervised PTO", supervised_dro="Supervised DRO",
              dfl_pto="DFL PTO", dfl_dro="DFL + DRO")


def build_content():
    runs, tables, sources = {}, {}, {}
    for mode in ["demo", "m5", "m5_intermittent"]:
        cfg = load_config(f"configs/{mode}.yaml")
        path = Path(cfg.data.artifacts_dir) / "summary_metrics.parquet"
        runs[mode], tables[mode] = cfg.run_id, pd.read_parquet(path).set_index("policy")
        sources[str(path)] = sha256_file(path)
    regular, intermittent = tables["m5"], tables["m5_intermittent"]
    def winner(table):
        row = table.drop(index="clairvoyant_oracle").sort_values("mean_cost_mu").iloc[0]
        return row.name, float(row.mean_cost_mu)
    rw, rc = winner(regular)
    iw, ic = winner(intermittent)
    delta = regular.loc["dfl_dro"]
    comparison = [[label, f"{regular.loc[policy].mean_cost_mu:.2f}", f"{intermittent.loc[policy].mean_cost_mu:.2f}"] for policy,label in LABELS.items()]
    slides = [
        dict(title="Inventory allocation: from forecasts to decisions", subtitle="Interview research case study | Decision intelligence for multi-store retail", bullets=[
            "Decision: how many units should a warehouse send to each store when stock and shelf space are limited?",
            "Objective: reduce shortage, holding and shipping cost while respecting integer inventory constraints.",
            "Deliverable: a CPU-based experiment pipeline, mathematical checks and an offline decision dashboard.",
            "Evidence: Walmart M5 sales proxies with synthetic operating economics; no claim of actual retailer savings."],
            notes="Open with the business decision rather than model complexity. Explain a scarce unit can be valuable at one store and excess at another. MU means synthetic monetary units. This is an interview research prototype, not an operational replenishment system.", seconds=65),
        dict(title="Data and assumptions define the experiment", subtitle="Chronological evaluation and explicit provenance", bullets=[
            "Regular and intermittent cohorts: 5 stores and 10 HOUSEHOLD SKUs each, selected using training history only.",
            "Each origin predicts the next seven calendar days. Main M5 split: 100 training, 8 validation and 12 test origins.",
            "Features use completed history; calendar events are known in advance. Future realized prices are excluded.",
            "Stock, supply, capacity and MU costs are synthetic. Observed sales can understate latent demand during stockouts.",
            "The v2 runs reuse previously inspected test periods: these are corrected retrospective experiments."],
            notes="Show Page A provenance and actual filters. ADI is daily inverse nonzero-sales frequency; do not claim a threshold absent from the selection rule. Discuss Walmart-to-MR.DIY domain shift. Twelve test weeks support a small case study, not a causal savings estimate.", seconds=75),
        dict(title="Learning and optimization answer different questions", subtitle="A distribution feeds the decision; SPO+ supplies training feedback",
            flow=["Pre-origin features", "Calibrated demand PMF", "Marginal costs / joint scenarios", "Integer allocation"],
            feedback="Training feedback: SPO+ updates the PMF through a fixed feasible optimization template.",
            bullets=[
            "Matched learning: the same architecture, initialization, batches and budget; DFL adds normalized SPO+.",
            "Probability-reweighting DRO: sum(w)=1 and 0 <= w[j] <= p[j]/(1-alpha). Its worst expected loss equals CVaR.",
            "Risk-neutral scenario ablations isolate sampling from risk aversion; weight zero is expected-cost optimization."],
            notes="Define PTO and DFL in plain language. The feasible set is fixed during SPO+ training; uncertain demand maps to objective coefficients. The optimizer participates in learning. This is probability-reweighting DRO, not Wasserstein DRO. Scenario dependence is estimated on training only; PMF temperature is selected on validation.", seconds=90),
        dict(title="Test the decision, not only the forecast", subtitle="Shared conditions and independent mathematical checks", bullets=[
            "Seven required policies plus two diagnostic ablations use the same origin, stock, supply, capacity and realized target.",
            "Primary outcomes: held-out cost, hindsight regret and fill rate; forecast error is a diagnostic.",
            "Three independent seeds use equal budgets and their own validation calibration and risk selection.",
            "Paired moving-block intervals resample time origins. Small samples and few tail weeks limit certainty.",
            "Carryover extension: explicit dispatch, admissions and deferred inventory; the hindsight MILP is replayed and tested against exhaustive tiny cases."],
            notes="The oracle knows future demand only for evaluation and never supplies a realistic recommendation. Explain why conserving total stock alone was insufficient and why matching the state/action contract matters. Show mathematical tests if asked. Negative or inconclusive model results are retained.", seconds=80),
        dict(title="The strongest method depends on the setting", subtitle="Primary seed 42: mean test-origin cost (synthetic MU; lower is better)", table_headers=["Policy", "Regular M5", "Intermittent M5"], table=comparison, bullets=[
            f"Primary-seed non-oracle policies: regular winner {rw} ({rc:.2f} MU); intermittent winner {iw} ({ic:.2f} MU).",
            f"Regular DFL+DRO minus supervised PTO: {delta.diff_vs_supervised_pto_mu:+.2f} MU; paired 95% band [{delta.paired_diff_vs_supervised_ci95_low_mu:.2f}, {delta.paired_diff_vs_supervised_ci95_high_mu:.2f}] MU.",
            "Primary-seed rankings vary across seeds and objectives. Full seed results are retained; universal superiority is not established."],
            notes="Read the table candidly, including heuristic wins and losses by advanced methods. The two stochastic ablations and oracle are available in the full report. Explain a negative paired difference favors DFL+DRO; an interval crossing zero is inconclusive on this sample. These costs are modelled outcomes, not revenue or proven operational savings.", seconds=90),
        dict(title="A decision tool makes the trade-off visible", subtitle="60-second local demonstration; cached models, fresh optimization", bullets=[
            "Page A: identify the M5-proxy badge, synthetic MU economics and chronological data split.",
            "Page C: inspect integer shipments, available supply, store capacity and expected shortage/excess.",
            "Page D: compare model expected cost, model CVaR and realized historical test cost as separate metrics.",
            "Change supply to 0.8 or demand uplift to +25%. Uplift is explicitly a synthetic intervention.",
            "Hard service mode meets a feasible target or separately establishes maximum achievable service. Soft mode permits a cost trade-off.",
            "Page E: show paired results and the baseline winner. Download CSVs; no training runs inside the UI."],
            notes="Leave the dashboard running before the interview. Use one controlled change, then return to defaults. Do not promise a latency threshold: actual solver and pipeline timings are in the artifacts. If the browser fails, use the local PDF or HTML and screenshots; do not rebuild models during the interview.", seconds=70),
        dict(title="Recommendation: validate before an operational pilot", subtitle="A defensible result includes its limitations", bullets=[
            "Choose policy using the business objective and validation evidence; do not assume the most complex model wins.",
            "Limitations: sales censoring, synthetic economics, small selected cohorts, short test horizon and domain shift.",
            "Retrospective reruns do not create a fresh confirmatory holdout. Future evidence should include untouched periods.",
            "Next: obtain real stock and lead-time records, validate costs and case packs, then run a monitored planner-in-the-loop pilot.",
            "Demonstrated: a tested research workflow and offline prototype. Enterprise-wide scale and actual savings remain to be established."],
            notes="Close with a measured proposal. Describe the data needed from an ERP and warehouse system and how to compare against the incumbent policy. Keep humans responsible for orders. Invite questions about calibration, solver feasibility, simple baselines and why robust optimization can cost more.", seconds=70),
    ]
    return dict(run_ids=runs, sources=sources, slides=slides, total_seconds=sum(s["seconds"] for s in slides))


def build_presentation(output_path="docs/presentation_slides.pptx"):
    content = build_content()
    Path("reports/release/presentation_content.json").write_text(json.dumps(content, indent=2))
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    def textbox(slide, x, y, w, h, text, size=20, color=(30,41,59), bold=False):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        shape.text_frame.word_wrap = True
        for i, line in enumerate(text.split("\n")):
            paragraph = shape.text_frame.paragraphs[0] if i == 0 else shape.text_frame.add_paragraph()
            paragraph.text = line
            paragraph.font.name = "Arial"
            paragraph.font.size = Pt(size)
            paragraph.font.bold = bold
            paragraph.font.color.rgb = RGBColor(*color)
            paragraph.space_after = Pt(8)
        return shape
    html_slides, markdown, notes = [], [], []
    for index, data in enumerate(content["slides"], 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        textbox(slide,.65,.4,12,.3,"DECISION INTELLIGENCE / INTERVIEW RESEARCH PROTOTYPE",11,(3,105,161),True)
        textbox(slide,.65,.9,12,.65,data["title"],28,(15,23,42),True)
        textbox(slide,.65,1.65,12,.5,data["subtitle"],15,(71,85,105))
        top = 2.35
        table_html = ""
        flow_html = ""
        if "flow" in data:
            for i, label in enumerate(data["flow"]):
                x = .75 + 3.1*i
                shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.3), Inches(2.6), Inches(.9))
                shape.fill.solid()
                shape.fill.fore_color.rgb = RGBColor(224,242,254)
                shape.line.color.rgb = RGBColor(125,211,252)
                textbox(slide,x+.13,2.46,2.34,.65,label,17,(15,23,42),True)
                if i < 3:
                    textbox(slide,x+2.68,2.5,.4,.5,"→",22,(3,105,161))
            textbox(slide,.75,3.45,11.85,.55,data["feedback"],16,(3,105,161))
            top = 4.2
            flow_html = '<div class="flow">'+"<span>→</span>".join('<div>'+html.escape(label)+'</div>' for label in data["flow"])+"</div><p class='feedback'>"+html.escape(data["feedback"])+"</p>"
        if "table" in data:
            table = slide.shapes.add_table(7,3, Inches(.7), Inches(2.25), Inches(11.9), Inches(2.45)).table
            table.columns[0].width,table.columns[1].width,table.columns[2].width = Inches(5.5),Inches(3.2),Inches(3.2)
            for r, values in enumerate([data["table_headers"]]+data["table"]):
                for c, value in enumerate(values):
                    cell = table.cell(r,c)
                    cell.text = value
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*( (15,23,42) if r == 0 else (241,245,249)))
                    for paragraph in cell.text_frame.paragraphs:
                        paragraph.font.size = Pt(17)
                        paragraph.font.name = "Arial"
                        paragraph.font.color.rgb = RGBColor(*( (255,255,255) if r == 0 else (30,41,59)))
            top = 5.0
            table_html = "<table><thead><tr>"+"".join(f"<th>{html.escape(v)}</th>" for v in data["table_headers"])+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{html.escape(v)}</td>" for v in row)+"</tr>" for row in data["table"])+"</tbody></table>"
        textbox(slide,.75,top,11.85,6.95-top,"\n".join("• "+b for b in data["bullets"]),16 if "table" in data else 19)
        textbox(slide,.65,7.03,12,.25,f"Synthetic MU economics | Corrected v2 experiments | {index}/7",10,(100,116,139))
        slide.notes_slide.notes_text_frame.text = data["notes"]+f"\nTarget duration: {data['seconds']} seconds.\nSources: "+", ".join(content["sources"])
        html_slides.append(f'<section class="slide"><small>INTERVIEW RESEARCH PROTOTYPE · {index}/7</small><h1>{html.escape(data["title"])}</h1><p class="subtitle">{html.escape(data["subtitle"])}</p>'+flow_html+table_html+"<ul>"+"".join(f"<li>{html.escape(b)}</li>" for b in data["bullets"])+"</ul><footer>Synthetic MU economics · Corrected v2 experiments</footer></section>")
        markdown += [f"# Slide {index}: {data['title']}","",data["subtitle"],""]
        if "table" in data:
            markdown += ["| "+" | ".join(data["table_headers"])+" |", "|---|---:|---:|"]+["| "+" | ".join(row)+" |" for row in data["table"]]+[""]
        if "flow" in data:
            markdown += [" → ".join(data["flow"]), "", data["feedback"], ""]
        markdown += ["- "+b for b in data["bullets"]]+[""]
        notes += [f"## Slide {index}: {data['title']} ({data['seconds']} seconds)","",data["notes"],""]
    prs.save(output_path)
    css = """*{box-sizing:border-box}body{margin:0;background:#e2e8f0;font-family:Arial,sans-serif;color:#1e293b}.slide{position:relative;width:1280px;height:720px;padding:42px 64px;background:white;margin:24px auto;overflow:hidden;page-break-after:always}.flow{display:flex;align-items:center;gap:12px;margin:25px 0}.flow div{background:#e0f2fe;border:1px solid #7dd3fc;border-radius:12px;flex:1;padding:20px 12px;min-height:85px;font-size:20px;font-weight:bold}.flow span{font-size:26px;color:#0369a1}.feedback{color:#0369a1;font-size:19px}small{color:#0369a1;font-weight:bold}h1{font-size:34px;margin:22px 0 14px;line-height:1.15}.subtitle{font-size:19px;color:#475569;margin-bottom:28px}li{font-size:22px;line-height:1.3;margin:0 0 12px}table{width:100%;border-collapse:collapse;margin:18px 0}th,td{text-align:left;padding:6px 14px;font-size:17px}th{background:#0f172a;color:white}td{background:#f1f5f9;border-bottom:2px solid white}table+ul li{font-size:18px;margin-bottom:10px}footer{position:absolute;bottom:24px;font-size:13px;color:#64748b}@media print{body{background:white}.slide{margin:0}@page{size:1280px 720px;margin:0}}"""
    Path("docs/presentation_slides.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Inventory decision case study</title><style>'+css+'</style>'+"".join(html_slides)+"</html>")
    Path("docs/presentation_slides.md").write_text("\n".join(markdown))
    Path("docs/PRESENTATION_SPEAKER_NOTES.md").write_text("# Speaker notes\n\nPrepared talk track; candidate delivery rehearsal must be performed personally. Target: "+str(content["total_seconds"]//60)+" minutes "+str(content["total_seconds"]%60)+" seconds.\n\n"+"\n".join(notes))
    print(json.dumps(dict(slides=len(prs.slides),run_ids=content["run_ids"],target_seconds=content["total_seconds"])))


if __name__ == "__main__":
    build_presentation()
