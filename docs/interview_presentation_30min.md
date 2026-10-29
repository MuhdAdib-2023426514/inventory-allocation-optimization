# 30-Minute Technical Interview Presentation: Which Store Gets the Last Unit?

**Target Role:** Data Scientist Executive (Decision Intelligence), MR.DIY  
**Candidate Session Structure:** ~18–20 minutes slide presentation + 5 minutes live dashboard demo + 5–7 minutes technical Q&A  
**Primary Deliverables:**
- Interactive Slide Deck: [`docs/interview_presentation_30min.html`](interview_presentation_30min.html) (Keyboard-driven, clean typography, embedded SVGs, dark/light theme, toggleable speaker notes drawer)
- 18-Page Static PDF Fallback: [`docs/interview_presentation_30min.pdf`](interview_presentation_30min.pdf) (1 page per slide, identical layout)
- Running Offline Streamlit App: `http://localhost:8511` (`streamlit run src/inventory_di/ui/streamlit_app.py --server.port 8511`)

---

## Executive Presentation Pacing & Structure

| Section | Slide # | Title | Target Time | Cumulative |
|---|---|---|---|---|
| **Opening** | Slide 1 | Which store gets the last unit? | 1.0 min | 0:00 – 1:00 |
| **The Answer First** | Slide 2 | Three findings in one slide | 1.5 min | 1:00 – 2:30 |
| **Part 1: Problem** | Slide 3 | Same stock, different outcome: allocation is the decision | 1.5 min | 2:30 – 4:00 |
| **Part 1: Problem** | Slide 4 | A better forecast score doesn't guarantee a better decision | 1.5 min | 4:00 – 5:30 |
| **Part 2: Approach** | Slide 5 | From history to shipments, with the optimiser in the training loop | 2.0 min | 5:30 – 7:30 |
| **Part 2: Approach** | Slide 6 | Risk-aware allocation: plan for the bad weeks, not only the average | 1.75 min | 7:30 – 9:15 |
| **Part 3: Analysis** | Slide 7 | How I made the comparison fair | 1.5 min | 9:15 – 10:45 |
| **Part 3: Analysis** | Slide 8 | Results: big win over equal split; models tie with a simple rule | 2.0 min | 10:45 – 12:45 |
| **Part 3: Analysis** | Slide 9 | Do the rankings survive retraining? Not reliably | 1.5 min | 12:45 – 14:15 |
| **Part 4: Solution** | Slide 10 | Live demo: the decision tool | 1.0 min (+ 5 min demo) | 14:15 – 20:15 |
| **Part 5: Business Impact** | Slide 11 | What the evidence supports, and what it doesn't | 1.5 min | 20:15 – 21:45 |
| **Part 5: Business Impact** | Slide 12 | Limitations and what I would do next | 1.25 min | 21:45 – 23:00 |
| **Part 5: Business Impact** | Slide 13 | From prototype to pilot: a low-risk path | 1.25 min | 23:00 – 24:15 |
| **Summary** | Slide 14 | Get the decision right first. Prove complexity before you ship it. | 0.75 min | 24:15 – 25:00 |
| **Technical Q&A / Backups** | Slides 15–18 | B1: Formulation, B2: DFL SPO+, B3: DRO/CVaR, B4: Multi-seed table | 5.0 min | 25:00 – 30:00 |

---

## Slide-by-Slide Visuals, Verbal Talk Tracks & Interviewer Q&A

### Slide 1 of 14: Which store gets the last unit?
- **Section:** Opening
- **Timing:** 60s (Target ≈ 1.0 min · Planned elapsed: 00:00)
- **Verbal Script (What you say out loud):**
  > *"Good morning. Thank you for the time. I'll spend about twenty minutes on a case study I built end to end, then show the working tool for five minutes, then take questions. The question is simple: when the warehouse can't cover every store, who gets the stock? I'll show you a method from recent research, and, just as importantly, an honest test of whether it beats the simple rules a retailer already uses."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why did you pick this problem?
- **Senior-Level Model Answer:**
  > **A:** Allocation is where forecasting meets money. Two teams can have the same forecast accuracy and very different stockout costs. It is also a problem where an analyst can do harm by shipping complexity that doesn't pay, so it is a good test of judgement.

---

### Slide 2 of 14: Three findings in one slide
- **Section:** The answer first
- **Timing:** 90s (Target ≈ 1.5 min · Planned elapsed: 01:00)
- **Verbal Script (What you say out loud):**
  > *"I'll start with the answer. One: simply replacing an equal split with a demand-aware rule cut weekly cost by about 133 MU, roughly 9 percent, and the interval is well away from zero. Two: on regular demand, the advanced methods tied with a simple proportional rule. Three: on intermittent items, forecast-then-optimise won clearly on the primary seed, but when I retrained with different random seeds the winner changed. So the honest headline is not 'my model wins', it is 'here is where complexity pays and where it doesn't, and here is how I know'."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** So your advanced method failed?
- **Senior-Level Model Answer:**
  > **A:** On this sample it didn't beat a strong simple rule, and I'd rather show that than hide it. The pipeline, verification and evaluation design are what transfer to a real business: they stop you deploying a model because it looks clever. With real stock records and costs the answer may change, and the same harness would show it.

---

### Slide 3 of 14: Same stock, different outcome: allocation is the decision
- **Section:** 1 · Problem
- **Timing:** 90s (Target ≈ 1.5 min · Planned elapsed: 02:30)
- **Verbal Script (What you say out loud):**
  > *"Here is the whole problem in one picture. Sixty units in the warehouse. Store A will sell sixty, store B only twenty. An equal split sends thirty each: A runs short by thirty, and ten units sit unsold at B. Sending forty and twenty fills sixty units instead of fifty. Nothing about the forecast changed; only the allocation did. At network scale, with hundreds of items and supply and shelf limits, you can't do this by eye. It needs an optimiser."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Isn't this just ranking stores by expected demand?
- **Senior-Level Model Answer:**
  > **A:** With one item and no limits, nearly. With shared warehouse supply, shelf capacity across items and uncertain demand, it becomes an integer optimisation: the value of the next unit depends on how many that store already has and how likely it is to sell.

---

### Slide 4 of 14: A better forecast score doesn't guarantee a better decision
- **Section:** 1 · Problem
- **Timing:** 90s (Target ≈ 1.5 min · Planned elapsed: 04:00)
- **Verbal Script (What you say out loud):**
  > *"Why not just build a better forecaster? Because forecast accuracy and business cost aren't the same thing. In my setup a missed sale costs about 5 times more than an extra unit on the shelf. Accuracy metrics punish both errors equally. So I judge every method by one number: how much more did its decision cost than the best decision you could have made with hindsight. That's regret, and it's the main score on every results slide."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why not use quantile regression at the critical ratio?
- **Senior-Level Model Answer:**
  > **A:** That answers 'how much for one store with unlimited supply'. When stores share a limited warehouse and shelf space, the quantile alone can't decide who to cut. You still need the optimiser; the question is what to feed it.

---

### Slide 5 of 14: From history to shipments, with the optimiser in the training loop
- **Section:** 2 · Approach
- **Timing:** 120s (Target ≈ 2.0 min · Planned elapsed: 05:30)
- **Verbal Script (What you say out loud):**
  > *"This is the pipeline. On each decision day the model sees only past data and predicts a full probability distribution for next week's demand, not a single number. From that I work out the value of each extra unit at each store, and an integer optimiser allocates within supply and shelf limits. Decision-focused learning adds the curved arrow: during training, the optimiser tells the model how costly its decision was, and the model updates to make cheaper decisions, not just more accurate forecasts. To make the comparison fair, both versions are identical except for that training signal."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** How do you know the optimiser actually influences training?
- **Senior-Level Model Answer:**
  > **A:** I test it. There's an automated check that the decision-focused gradient is finite and non-zero, and that switching it off changes the parameter updates. Without that check, a lot of 'DFL' implementations are just regression with extra steps.

---

### Slide 6 of 14: Risk-aware allocation: plan for the bad weeks, not only the average
- **Section:** 2 · Approach
- **Timing:** 105s (Target ≈ 1.8 min · Planned elapsed: 07:30)
- **Verbal Script (What you say out loud):**
  > *"The second idea is robustness. Instead of planning for the average week, I simulate many possible weeks and let the optimiser give extra weight to the expensive ones. There's a neat result that this is exactly the same as minimising the average of the worst tail of costs, called CVaR. The two dials, how deep into the tail and how much to care, were picked on validation weeks, never on test weeks. And when the risk dial is at zero, it collapses back to the normal plan, which I check automatically."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why would robust optimisation ever cost more?
- **Senior-Level Model Answer:**
  > **A:** Because it deliberately buys insurance. If the forecast's uncertainty is too wide, or bad weeks don't arrive in the test period, you pay the premium without collecting. That is what happened here on average, which is why I report average and tail cost separately.

---

### Slide 7 of 14: How I made the comparison fair
- **Section:** 3 · Analysis
- **Timing:** 90s (Target ≈ 1.5 min · Planned elapsed: 09:15)
- **Verbal Script (What you say out loud):**
  > *"Results are only as good as the test. Data is split strictly by time: 100 weeks to train, 8 to tune, 12 to test. Anything chosen, item selection, caps, risk dials, was chosen before touching test data. Every policy faces exactly the same conditions. I deliberately included strong simple baselines, because that's what a real business already has. And a separate script recomputes every allocation and cost from scratch, so the numbers you'll see are checked, not just reported."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** How do you handle stockouts hiding true demand in sales data?
- **Senior-Level Model Answer:**
  > **A:** M5 sales are a proxy: when a store ran out, true demand was higher than recorded sales. I state this as a limitation. In production I'd combine sales with on-hand stock records and estimate the hidden demand, for example with a censored-demand model.

---

### Slide 8 of 14: Results: big win over equal split; models tie with a simple rule
- **Section:** 3 · Analysis
- **Timing:** 120s (Target ≈ 2.0 min · Planned elapsed: 10:45)
- **Verbal Script (What you say out loud):**
  > *"Here are the results. Bars are regret, extra cost versus perfect hindsight, so shorter is better, and the lines are 95 percent intervals over time. On regular demand, equal split is clearly worst; everything else overlaps, including the simple proportional rule. On intermittent demand, plain forecast-then-optimise is the one policy clearly better than the proportional rule on the primary seed, by about 7 MU a week. Notice also the gap that remains to the oracle, around 279 MU on regular demand: that's demand nobody could have predicted."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Is a 12-week test enough?
- **Senior-Level Model Answer:**
  > **A:** It's enough to see big effects, like the 133 MU gain over equal split. It isn't enough to separate methods that differ by 5 to 10 MU. That's why I show intervals and say 'tie' instead of picking a winner.

---

### Slide 9 of 14: Do the rankings survive retraining? Not reliably
- **Section:** 3 · Analysis
- **Timing:** 90s (Target ≈ 1.5 min · Planned elapsed: 12:45)
- **Verbal Script (What you say out loud):**
  > *"I then retrained every model from scratch with seeds 42, 101, 202, each with the same budget and its own tuning. The simple rules don't change, they have no randomness, so they sit on one point. The learned models move around. On intermittent demand, the primary-seed advantage of forecast-then-optimise disappears on the other two seeds. This is exactly the check that stops a team from presenting a lucky run as a breakthrough."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why not just report the best seed?
- **Senior-Level Model Answer:**
  > **A:** Because the business doesn't get to choose the seed. If performance depends on a random number, the honest expectation is the spread, and here the spread covers the simple rule.

---

### Slide 10 of 14: Live demo: the decision tool
- **Section:** 4 · Solution
- **Timing:** 60s (Target ≈ 1.0 min · Planned elapsed: 14:15)
- **Verbal Script (What you say out loud):**
  > *"Let me show you the tool. I'll switch to the dashboard now."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Could planners use this?
- **Senior-Level Model Answer:**
  > **A:** As a decision-support view, yes: it shows the allocation, why, and what changes if supply drops. It doesn't place orders; a planner stays responsible. For production it would need real stock feeds, case-pack rounding and an override log.

---

### Slide 11 of 14: What the evidence supports, and what it doesn't
- **Section:** 5 · Impact
- **Timing:** 90s (Target ≈ 1.5 min · Planned elapsed: 15:15)
- **Verbal Script (What you say out loud):**
  > *"So what does this mean for a business? It strongly supports getting allocation logic right: about 9 percent lower cost against an equal split in this setup. It does not support claiming that the advanced methods pay off, or any figure in Ringgit, because the costs are synthetic. My recommendation is to make the simple-rule-plus-optimiser the baseline, and only promote advanced models if they win a pre-registered shadow test on your real data."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** How would you turn MU into Ringgit?
- **Senior-Level Model Answer:**
  > **A:** Replace the synthetic costs with real ones: lost gross margin for a stockout, cost of capital and space for holding, freight per unit. Then rerun the same harness on real stock and sales and multiply weekly regret differences by volume, with intervals.

---

### Slide 12 of 14: Limitations and what I would do next
- **Section:** 5 · Impact
- **Timing:** 75s (Target ≈ 1.2 min · Planned elapsed: 16:45)
- **Verbal Script (What you say out loud):**
  > *"I want to be upfront about limits, because each one changes how far you can trust the result. The biggest are that sales aren't true demand, and the costs are synthetic. The test is short, and because I fixed bugs after first looking at the test weeks, this is a corrected re-run, not a fresh test. Each limitation has a concrete next step, and most of them need data that a retailer already has."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** What would you change if you started again?
- **Senior-Level Model Answer:**
  > **A:** Lock the test period away and pre-register the comparison before building anything, and start from real stock records instead of sales alone.

---

### Slide 13 of 14: From prototype to pilot: a low-risk path
- **Section:** 5 · Impact
- **Timing:** 75s (Target ≈ 1.2 min · Planned elapsed: 18:00)
- **Verbal Script (What you say out loud):**
  > *"If this went forward, I'd do it in three gated steps. First, shadow mode: run on real data next to the planners, ship nothing, measure. Second, a small pilot where planners can override and every override is reviewed, because that's where you learn what the model is missing. Third, scale, which is technically straightforward because each item's warehouse limit can be solved in parallel. At each gate the advanced methods have to earn their place."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** What is the biggest risk in rollout?
- **Senior-Level Model Answer:**
  > **A:** Planner trust and data quality, more than algorithms. Shadow mode, visible reasons for each allocation and an override log address both.

---

### Slide 14 of 14: Summary
- **Section:** Opening
- **Timing:** 45s (Target ≈ 0.8 min · Planned elapsed: 19:15)
- **Verbal Script (What you say out loud):**
  > *"To sum up: I built and verified the full pipeline, found a clear gain from better allocation logic, and showed honestly that the advanced methods haven't yet earned their complexity on this data. That combination, building it, testing it hard, and telling you plainly what it shows, is how I'd approach decision intelligence work here. Thank you, I'm happy to take questions."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why should we hire you?
- **Senior-Level Model Answer:**
  > **A:** I can take a business decision from data to a working, tested tool, and I report what the evidence says even when it isn't the exciting answer. That protects the business from expensive mistakes.

---

### Backup Slide B1: B1 · The optimisation model
- **Section:** Backup
- **Timing:** 0s (Target ≈ 0.0 min · Planned elapsed: 20:00)
- **Verbal Script (What you say out loud):**
  > *"Backup: the full model if asked."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why integer?
- **Senior-Level Model Answer:**
  > **A:** Shipments are whole units; at low volumes rounding a continuous solution can break supply limits or waste stock.

---

### Backup Slide B2: B2 · Decision-focused learning details
- **Section:** Backup
- **Timing:** 0s (Target ≈ 0.0 min · Planned elapsed: 20:00)
- **Verbal Script (What you say out loud):**
  > *"Backup: DFL details."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why keep NLL in the loss?
- **Senior-Level Model Answer:**
  > **A:** Pure SPO+ can wreck probability calibration, and the robust optimiser needs calibrated scenarios. NLL keeps the distribution honest.

---

### Backup Slide B3: B3 · Why the robust model equals CVaR
- **Section:** Backup
- **Timing:** 0s (Target ≈ 0.0 min · Planned elapsed: 20:00)
- **Verbal Script (What you say out loud):**
  > *"Backup: CVaR equivalence."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Why not Wasserstein DRO?
- **Senior-Level Model Answer:**
  > **A:** It needs extra modelling choices and solver cost; the reweighting set has an exact, testable meaning. Wasserstein is listed as future work, not claimed.

---

### Backup Slide B4: B4 · Full results, primary seed (MU per week)
- **Section:** Backup
- **Timing:** 0s (Target ≈ 0.0 min · Planned elapsed: 20:00)
- **Verbal Script (What you say out loud):**
  > *"Backup: full tables."*
- **Anticipated Interviewer Question (Trap):**
  > **Q:** Which tail result looked best?
- **Senior-Level Model Answer:**
  > **A:** DFL → optimise had the lowest worst-10% cost on regular demand, but that statistic rests on about one week, so I don't claim it.

---

## Live Dashboard Demo Walkthrough Protocol (5 Minutes)

When transitioning at Slide 10 to the running Streamlit dashboard (`localhost:8511`), follow this sequence:
1. **Page A (Configuration & Methodology):** Point to the 'Walmart M5 Sales Proxy' and 'Synthetic MU Economics' badges. Explain the strict chronological split (100 train / 8 val / 12 test).
2. **Page C (Stock Allocation Matrix):** Show the allocation table. Point out that all store shipments are whole integers, no warehouse supply is exceeded (sum <= W_p), and total store shelf space is respected (sum <= C_s).
3. **Page D (Interactive Simulation & Stress Testing):**
   - Drag the Warehouse Supply multiplier from 1.0x down to 0.6x. Show how the optimiser rations scarce stock to high-volume stores rather than cutting everywhere equally.
   - Toggle the Risk Weight dial from 0.0 (risk-neutral) to 0.25 (robust CVaR). Show that when risk weight is 0.0, the robust model mathematically collapses back to the nominal expected cost plan.
4. **Page E (Backtest & Cost Regret Breakdown):** Highlight the paired regret reduction table and show the CSV download capability for offline auditing.

---

## Key Behavioral Guidelines for the Candidate
- **Pacing:** Don't rush through Slide 2 ('The Answer First'). Senior interviewers love when a candidate gives the punchline immediately.
- **Honesty:** When asked if DFL beat the baseline, stand firm on the empirical truth: *'On regular demand, it tied with a simple proportional rule. I am showing you an honest evaluation system that prevents the company from deploying unproven complexity.'* That answer demonstrates senior leadership maturity.
- **Hotkeys in HTML Deck:** Press `S` to toggle speaker notes drawer during practice, `O` for slide overview grid, `T` for dark mode, `F` for fullscreen, `Space`/`→` to advance.