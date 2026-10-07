# F-group ablation: no explicit metric comparisons

Nine unchanged inputs: students 01, 02, 10; moves 1, 2, 3.
Keep findings and coach_summary; remove metric_comparisons and explicit per-metric comparison instructions.
Teacher ranges, raw samples and all student measurements remain unchanged.
Run with run_f_group_api.py using a separate output root and model qwen3.8-max-0902.
Keep enable_thinking=true, thinking_budget=4000, temperature=0.2, top_p=0.8, max_tokens=16384.
First run --case-id F001; verify success, then run all cases (the completed F001 is reused).
Do not infer free billing from HTTP success; check the provider's quota/billing console.
The previous alias may differ from this snapshot: not a strict prompt-only causal comparison.
Format/evidence-ID validation is not a diagnosis or numerical-claim accuracy assessment.
