# Antigravity starting instruction

Read `AGENTS.md` fully and treat it as the project's implementation contract. Inspect the existing repository before creating or modifying files.

Build the **Decision-Focused and Distributionally Robust Inventory Allocation** application end to end using the phases and acceptance criteria in `AGENTS.md`. Your first actions should be to:

1. Identify the current repository structure and Python environment.
2. Create/update `docs/IMPLEMENTATION_STATUS.md` with a short concrete plan.
3. Bootstrap a CPU-first open-source environment and verify a tiny MILP and PyEPO SPO+ backward pass.
4. Implement the fully labeled synthetic-demo data path first, including exact inventory-cost accounting and feasible baselines.
5. Continue through probabilistic forecasting, genuine decision-focused training, explicit scenario-reweighting DRO, rolling backtesting, dashboard, and presentation documentation.

Execute commands and tests as you implement. Do not stop at scaffolding, an architecture proposal, or an unverified code snippet. If the M5 data are not present, continue with synthetic-demo mode and give clear instructions for adding the licensed/competition dataset. Never invent evaluation metrics or claim M5 results without executing on M5 data. Keep the research claims mathematically accurate and implementation status honest.

After each milestone, report what works, what was tested, and the next step. Prefer a verified smaller working system to an ambitious incomplete one.
