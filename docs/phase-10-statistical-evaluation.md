# Phase 10 log — Statistical evaluation

**Status:** complete (branch `arena/01a09f4f-brainos-context-lab`)
**Plan section:** Phase 10 — Statistical Evaluation (§16).
**Depends on:** Phase 8 (metric suite and plot series), Phase 9 (controlled experiment harness, repeated trials, paired records, cost controls).

## Objective

From the plan:

> Run multiple trials where generation is stochastic.
> Report:
> - mean
> - standard deviation
> - 95% confidence interval
> For paired benchmark tasks, use paired comparisons.
> Report effect sizes where appropriate.
> Do not report only a single best run.
> 
> Required plots:
> 1. Accuracy vs conversation length.
> 2. Context tokens vs conversation length.
> 3. Accuracy vs context tokens.
> 4. Retrieval precision/recall.
> 5. Token savings.
> 6. Quality-adjusted efficiency.

Phase 9 established the controlled trial runner where the same benchmark tasks and parameters run through every baseline mode. Phase 10 completes the statistical evaluation layer: descriptive statistics (mean, SD, 95% CI) across repeated trials, task-level paired difference tests (paired Student's t-test, Cohen's d, Hedges' g bias-corrected effect size, sign test win/loss/tie analysis), and rendering the six planned figures with confidence intervals and error bars.

## Starting state

- `evaluation/metrics.py` had basic `summarize()` using normal approximation $1.96 \cdot s / \sqrt{n}$ and degradation AUC calculations.
- `evaluation/analysis.py` built headline rows and flat plot series without trial-level statistical aggregation.
- `evaluation/plots.py` contained renderers for the six series without error bars or trial aggregation.
- `evaluation/experiment.py` recorded multi-trial task records paired by task id, but lacked statistical aggregation across trials in its output artifact.
- Constraints carried in: A run is not a trial; repeat then summarize; never present one run's task-level mean as a trial CI; paired comparisons must pair by task across identical controls; render all six planned plots.

## Work completed

### New / changed modules

| File | Purpose |
| --- | --- |
| [`src/evaluation/metrics.py`](../src/evaluation/metrics.py) | Added exact Student's t critical values (`STUDENT_T_CRITICAL_95`, `t_critical`) for small degrees of freedom ($df \in [1, 30]$) and Cornish-Fisher expansion for $df > 30$; exact regularized incomplete beta function (`_incbeta`) and continued fractions (`_betacf`) for two-tailed Student's t p-values (`student_t_p_value`); two-sided binomial sign test p-values (`sign_test_p_value`); Cohen's d effect size (`cohens_d`, paired $d_z = \bar{d}/s_d$ and independent pooled $s_{pooled}$); Hedges' g bias-corrected effect size (`hedges_g` with $J(df) \approx 1 - 3/(4df - 1)$); effect size classification (`effect_size_magnitude`: negligible, small, medium, large); `PairedDifference` dataclass and `paired_difference_test`. |
| [`src/evaluation/analysis.py`](../src/evaluation/analysis.py) | Added `summarize_trial_modes` (summarizes all headline metrics, by_length ladders, and degradation AUC across trials with mean, SD, 95% CI); `compare_modes_paired` (computes paired task differences, t-test, p-value, Cohen's d, Hedges' g, wins/losses/ties across paired tasks); `pairwise_comparisons` (baseline comparisons against Mode A plus key comparisons: BrainOS vs Sliding Window, BrainOS vs RAG, BrainOS+RAG vs BrainOS); `statistical_analysis` (full statistical evaluation bundle); `statistical_plot_series` (builds plot series containing trial mean, SD, CI, error bars). |
| [`src/evaluation/plots.py`](../src/evaluation/plots.py) | Enhanced `_line_plot` and `_bar_plot` to support error bars (`yerr`, capsize=3) from trial standard deviations / confidence intervals; updated `plot_all` to automatically generate statistical plot series with error bars; renders all six required plots (accuracy vs length, tokens vs length, accuracy vs tokens, retrieval precision/recall, token savings, quality-adjusted efficiency). |
| [`src/evaluation/reports.py`](../src/evaluation/reports.py) | Added `statistical_report` building the Phase 10 structured statistical JSON report containing trial summaries, paired comparisons, headline table, and plot series. |
| [`src/evaluation/compare.py`](../src/evaluation/compare.py) | Enhanced CLI to accept experiment artifacts and multiple run JSON files; added `--stats`, `--baseline-mode`, and `--plots-dir`; prints formatted trial statistics (mean ± SD [95% CI]) and paired comparison matrix with difference, CI, t-statistic, p-value, Cohen's d, effect size classification, and W/L/T counts. |
| [`src/evaluation/experiment.py`](../src/evaluation/experiment.py) | Added `ExperimentRun.statistical_summary()` method and added `"statistical_summary"` to `to_dict()`; added `--plots-dir` CLI option; automatically prints trial statistics when `trials > 1`. |
| Tests | Added `test_t_critical_values_and_summarize_with_t`, `test_student_t_p_value_and_sign_test`, `test_cohens_d_and_hedges_g`, `test_paired_difference_test_and_edge_cases` in `tests/unit/test_metrics.py`; added `test_summarize_trial_modes_aggregates_across_repeated_trials`, `test_compare_modes_paired_and_pairwise_comparisons`, `test_statistical_analysis_and_report_schema`, `test_statistical_plot_series_and_rendering_all_six_plots`, `test_compare_cli_with_stats_and_plots_dir` in `tests/evaluation/test_analysis.py`; added `test_experiment_run_statistical_summary_with_multiple_trials` and `test_cli_renders_plots_when_plots_dir_is_supplied` in `tests/evaluation/test_experiment.py`; added `test_live_statistical_evaluation_and_plots` in `tests/integration/test_controlled_experiment_live.py`. **586 → 598 tests**; `ruff check .` clean. |

### Statistical evaluation contract

```text
statistical_report
  metrics_version       "metrics-v1"
  trial_summaries       per mode:
                          trials, label
                          metrics: {metric_name: {count, mean, standard_deviation, confidence_interval_95}}
                          by_length: {length_tier: {length, trials, accuracy: Summary, context_tokens: Summary}}
                          degradation: {trials_with_curve, area_under_curve: Summary, mean_degradation: Summary}
  paired_comparisons[]  per (mode_a, mode_b, metric):
                          metric, sample_size, mean_a, mean_b, mean_difference,
                          standard_deviation_difference, standard_error,
                          confidence_interval_95, t_statistic, p_value,
                          cohens_d, hedges_g, effect_size_magnitude,
                          wins, losses, ties, win_rate, sign_test_p_value
  headline_table[]      compact cross-mode comparison rows with trial statistics
  series                plot series for all six figures with trial SD/CI error bars
```

### Statistical formulas implemented

1. **Student's t Critical Values & 95% CI**:
   $$\text{Margin of error} = t_{0.975, df} \cdot \frac{s}{\sqrt{n}}$$
   where $df = n - 1$. For $df \le 30$, exact lookup values are used; for $df > 30$, higher-order Cornish-Fisher quantile expansion is used.

2. **Paired Difference t-test & Exact Two-Tailed p-value**:
   $$d_i = a_i - b_i, \quad \bar{d} = \frac{1}{n} \sum d_i, \quad s_d = \sqrt{\frac{1}{n-1} \sum (d_i - \bar{d})^2}$$
   $$SE = \frac{s_d}{\sqrt{n}}, \quad t = \frac{\bar{d}}{SE}$$
   $$p = I_{\frac{df}{df + t^2}}\left(\frac{df}{2}, \frac{1}{2}\right)$$
   computed via the regularized incomplete beta function without external dependencies.

3. **Effect Sizes**:
   - **Cohen's $d_z$ (Paired)**: $d_z = \bar{d} / s_d$
   - **Cohen's $d$ (Independent)**: $d = (\bar{a} - \bar{b}) / s_{pooled}$
   - **Hedges' $g$ (Bias-corrected)**: $g = d \cdot \left(1 - \frac{3}{4 \cdot df - 1}\right)$
   - **Classification**: negligible ($|d| < 0.2$), small ($0.2 \le |d| < 0.5$), medium ($0.5 \le |d| < 0.8$), large ($|d| \ge 0.8$).

4. **Sign Test / Binomial Test**:
   $$p = 2 \cdot \sum_{k=0}^{\min(W, L)} \binom{W+L}{k} (0.5)^{W+L}$$

### Measured behaviour (dry run, smoke dataset)

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick --modes all \
  --dry-run --output results/phase10-dry.json --plots-dir results/phase10-plots
PYTHONPATH=src .venv/bin/python -m evaluation.compare results/phase10-dry.json \
  --stats --output results/phase10-stat-report.json
```

```text
=== Trial Statistics (Mean ± SD [95% CI]) ===
Mode: brainos (Mode D — BrainOS memory) — 1 trial(s)
  mean_final_context_tokens       : 193.5714 ± 0.0000 [193.5714, 193.5714]
  mean_token_savings              : 959.1429 ± 0.0000 [959.1429, 959.1429]

=== Paired Comparisons Across Benchmark Tasks ===
Pair                         Metric             Diff (A-B)   95% CI               t-stat   p-val    Cohen's d  Effect     W/L/T   
----------------------------------------------------------------------------------------------------------------------------------
brainos vs full_context      final_context_tokens -959.1429    [-1042.4773, -875.8084] -28.16   <0.0001  -10.64     large      0/7/0   
brainos vs full_context      token_savings      +959.1429    [+875.8084, +1042.4773] 28.16    <0.0001  10.64      large      7/0/0   
brainos vs sliding_window    evidence_in_prompt +0.7143      [+0.2630, +1.1656]   3.87     0.0082   1.46       large      5/0/2   
brainos vs rag               final_context_tokens -77.0000     [-93.4317, -60.5683] -11.47   <0.0001  -4.33      large      0/7/0   
brainos_rag vs brainos       final_context_tokens +173.5714    [+163.8150, +183.3278] 43.53    <0.0001  16.45      large      7/0/0   
```

All 6 plots rendered:
- `results/phase10-plots/accuracy_vs_length.png`
- `results/phase10-plots/tokens_vs_length.png`
- `results/phase10-plots/accuracy_vs_tokens.png`
- `results/phase10-plots/retrieval.png`
- `results/phase10-plots/token_savings.png`
- `results/phase10-plots/quality_adjusted_efficiency.png`

## Constraints carried into later phases

1. **A run is not a trial.** When evaluating stochastic models, use repeated trials ($T \ge 3$) and report trial mean, SD, and 95% CI.
2. **Paired tests pair by task id.** Always evaluate paired differences on matching tasks rather than treating observations as independent samples.
3. **Effect sizes qualify statistical significance.** Report Cohen's d / Hedges' g along with p-values to distinguish practically meaningful differences from sample-size artifacts.
4. **All 6 plots can be generated directly via `--plots-dir`** in both `evaluation.experiment` and `evaluation.compare`.
5. **A single run or single length cannot manufacture empirical variance or AUC curves**; single-observation CIs collapse to $(val, val)$ and single-length AUC remains `null`.

## Validation

```bash
.venv/bin/pytest -q
# 598 passed

.venv/bin/ruff check .
# All checks passed!
```
