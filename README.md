# EvoTx

EvoTx is a transaction-level framework for detecting and improving smart-contract attack classifiers. It combines structured evidence packets, rule/plan based judging, and a few-shot evolution loop that learns from false positives, false negatives, and uncertain cases.

The project is organized around a simple contract:

```text
transaction evidence
-> packet views
-> Rule + Plan runtime
-> Judge verdicts
-> Reviewer signals
-> local Rule/Plan candidates
-> validation and promotion
```

The current experiments focus on two settings:

- `between_attacks`: detect one attack family against other attack families.
- `benign_vs_attack`: detect one attack family against benign transactions.

Supported attack labels include `price_manipulation`, `access_control`, `insufficient_validation`, `flashloans`, `reentrancy`, `token_semantic_exploitation`, `protocol_accounting_exploitation`, and `market_manipulation`.

## Repository Layout

```text
evotx/
  adapters/                  LLM provider adapters
  core/                      Rule, Plan, and shared schemas
  runtime/                   packet construction, Judge runtime, followup tools
  evolution/                 Reviewer, Rule/Plan updaters, validation logic
  utils/                     result slimming, reporting, compatibility helpers

experiments/
  train_fewshot.py           few-shot evolution entry point
  evaluate.py                fixed Rule/Plan evaluation entry point
  aggregate_cross_attack_overlap.py
  summarize_cross_attack_hotmap.py

ps/
  0827/                      current unified PowerShell experiment scripts
  v6/                        v6-compatible experiment scripts
  export/                    summary and table export scripts

data/
  csv/                       dataset CSVs
  cache/                     packet/source caches
  rules/<episode>/           saved Rule versions
  plans/<episode>/           saved Plan versions
  results/<episode>/         round artifacts, final reports, eval outputs
  log/<episode>/             execution logs
```

## Setup

Use Python 3.10+ on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Most experiments call LLM providers and optional source-code lookup services. Put credentials in a local `.env` file at the project root. Do not commit `.env`.

Common variables:

```dotenv
# GLM / Zhipu
ZHIPU_API_KEY=...
GLM_API_KEY=...
GLM_EN_API_KEY=...
ZHIPU_BASE_URL=...
GLM_BASE_URL=...
GLM_EN_BASE_URL=...
ZHIPU_ANTHROPIC_BASE_URL=...
GLM_EN_ANTHROPIC_BASE_URL=...
GLM_EN_ENABLE_ANTHROPIC=1

# OpenAI-compatible route
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...

# Other optional providers
ANTHROPIC_API_KEY=...
ANTHROPIC_BASE_URL=...
MINIMAX_API_KEY=...
MINIMAX_BASE_URL=...
XUNFEI_API_KEY=...
XUNFEI_BASE_URL=...
VOLCANO_API_KEY=...
ARK_API_KEY=...

# Explorer/source lookup
ETHERSCAN_API_KEY=...
EXPLORER_API_KEY=...
ETHERSCAN_PROXY_URL=...
SOURCE_CODE_PROXY_URL=...

# Optional defaults
EVOTX_LLM_PROVIDER=glm-en
LLM_PROVIDER=glm-en
API_TIMEOUT_MS=600000
```

You can also pass provider/model/key parameters directly to the PowerShell scripts. Script parameters take precedence where supported.

## Data Layout

The default public experiment scripts read CSVs from:

```text
data/csv/v6.5/<Attack directory>/
```

Each attack directory should contain files using the normalized attack prefix:

```text
<attack>_Fewshot_pos.csv
<attack>_Fewshot_neg.csv
<attack>_evaluation_pos.csv
<attack>_evaluation_neg.csv
<attack>_evaluation_hard_pos.csv
<attack>_evaluation_hard_neg.csv
<attack>_guard.csv
```

Example:

```text
data/csv/v6.5/Access control/access_control_Fewshot_pos.csv
data/csv/v6.5/Access control/access_control_evaluation_neg.csv
```

The CSVs should identify the transaction and label fields expected by `experiments/train_fewshot.py` and `experiments/evaluate.py`. Runtime evidence packets and source-code caches are written under `data/cache/`.

## Few-Shot Evolution

The recommended unified entry point for attack-vs-other-attacks experiments is:

```powershell
.\ps\0827\between_attacks\run_between_attacks_fewshot.ps1 `
  -Episode 20000 `
  -AttackLabel price_manipulation `
  -NumShots 7 `
  -RunVariant evolution `
  -RunEvaluation $true `
  -LlmModel "glm-5.2" `
  -LlmProvider "glm-en" `
  -EnableAnthropic $true `
  -JudgeThinking default `
  -ColdStartThinking default `
  -PlannerThinking default `
  -AggregatorThinking disabled `
  -ReviewThinking default `
  -UpdateThinking default
```

For attack-vs-benign experiments:

```powershell
.\ps\0827\benign_vs_attack\run_attack_vs_benign_fewshot.ps1 `
  -Episode 30000 `
  -AttackLabel access_control `
  -NumShots 7 `
  -RunVariant evolution `
  -NegativeTrainingMode benign_only `
  -RunEvaluation $true `
  -LlmModel "glm-5.2" `
  -LlmProvider "glm-en" `
  -EnableAnthropic $true
```

Key options:

- `-Episode`: output namespace under `data/results`, `data/rules`, `data/plans`, and `data/log`.
- `-AttackLabel`: one of the supported normalized attack labels.
- `-NumShots`: number of rows taken from each few-shot CSV.
- `-RunEvaluation`: runs eval after evolution when set to `$true`.
- `-RuleFile` / `-PlanFile`: start from a specific Rule/Plan artifact instead of cold-start generation.
- `-GenerateInitialPlanWithAgent`: when `$true`, asks the Planner LLM to create an initial plan if no `-PlanFile` is supplied.
- `-RunVariant evolution`: writes normal evolution outputs in the episode root.
- `-RunVariant wo_evolution_v1`: evaluates the initial v1 artifacts and stores eval outputs under a variant subdirectory.

The older v6-compatible scripts remain available:

```text
ps/v6/between_attacks/fewshot/run_v6_between_attacks_fewshot.ps1
ps/v6/between_attacks/eval/run_v6_eval_latest.ps1
ps/v6/benign_vs_attack/fewshot/run_v6_attack_vs_benign.ps1
ps/v6/benign_vs_attack/eval/run_v6_eval_benign_latest.ps1
```

## Evaluation

Evaluate the latest evolved Rule/Plan for attack-vs-other-attacks:

```powershell
.\ps\0827\between_attacks\run_between_attacks_eval_latest.ps1 `
  -Episode 20000 `
  -AttackLabel price_manipulation `
  -RunVariant evolution `
  -LlmModel "glm-5.2" `
  -LlmProvider "glm-en"
```

Evaluate the v1 no-evolution baseline:

```powershell
.\ps\0827\between_attacks\run_between_attacks_eval_latest.ps1 `
  -Episode 20000 `
  -AttackLabel price_manipulation `
  -RunVariant wo_evolution_v1 `
  -LlmModel "glm-5.2" `
  -LlmProvider "glm-en"
```

For attack-vs-benign:

```powershell
.\ps\0827\benign_vs_attack\run_eval_benign_latest.ps1 `
  -Episode 30000 `
  -AttackLabel access_control `
  -RunVariant evolution `
  -NegativeTrainingMode benign_only `
  -LlmModel "glm-5.2" `
  -LlmProvider "glm-en"
```

Normal evolution eval outputs are written directly under:

```text
data/results/<episode>/
```

No-evolution eval outputs are written under:

```text
data/results/<episode>/wo_evolution_v1/
```

Common files include:

```text
eval_results.json
eval_slim_results.json
eval_summary.json
eval_failed.csv
```

## Cross-Attack Overlap

Cross-attack overlap experiments reuse the evaluation runtime to test one detector against examples from other attack families:

```powershell
.\ps\v6\between_attacks\eval\run_v6_cross_attack_hotmap.ps1 `
  -Episode 10126 `
  -AttackLabel insufficient_validation
```

Inputs are prepared under:

```text
data/csv/v6.5/hotmap/
```

Outputs are written into the detector episode directory, for example:

```text
data/results/<episode>/hotmap_<attack>_results.json
data/results/<episode>/hotmap_<attack>_slim_results.json
data/results/<episode>/hotmap_<attack>_summary.json
data/results/<episode>/hotmap_<attack>_by_source.csv
data/results/<episode>/hotmap_<attack>_case_results.csv
```

Aggregate overlap tables are produced with:

```powershell
.\ps\export\export_cross_attack_overlap_table.ps1
```

## Exporting Tables

Few-shot summaries:

```powershell
.\ps\export\export_fewshot_summary_table.ps1
```

Evaluation summaries:

```powershell
.\ps\export\export_eval_summary_table.ps1 -UseEvolution $true
.\ps\export\export_eval_summary_table.ps1 -UseEvolution $false
```

The exporters read episode reports and write CSV tables under `data/results/` unless a script-specific output path is supplied.

## Output Artifacts

For each few-shot episode, the most important files are:

```text
data/results/<episode>/final_report.json
data/results/<episode>/round_XX/current_summary.json
data/results/<episode>/round_XX/candidate_summary.json
data/results/<episode>/round_XX/signal_terminal_audit.json
data/results/<episode>/round_XX/rejected_update_memory.json
data/rules/<episode>/<attack>__latest.json
data/plans/<episode>/<attack>__plan_latest.json
```

The evolution loop promotes a candidate only after it passes the configured train validation and guard checks. Rejected candidates are still saved with terminal reasons so failed evolution can be diagnosed without rerunning the whole episode.

## Notes

- Keep `.env`, provider keys, local cache dumps, and private traces out of public commits.
- Use distinct episode numbers for independent experiments.
- Use `-DryRun` on PowerShell scripts when checking generated arguments without running inference.
- Reproducibility depends on provider behavior, rate limits, source-code availability, and cached packet state. Preserve `data/results/<episode>/`, `data/rules/<episode>/`, `data/plans/<episode>/`, and `data/log/<episode>/` for auditability.
