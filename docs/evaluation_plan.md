# Evaluation Plan

## Quantitative (see scripts/run_quant_eval.py)
- Dataset: `data/eval_questions.json` (expand to 20+ queries spanning intents).
- Models: Mistral 7B, Kimi-K2, Llama 3.3 70B, Gemini 2.0 Flash.
- Metrics: latency_ms, prompt/completion/total tokens, response length, optional correctness flag when gold available.
- Steps: populate `context` and `gold_answer` fields, run script with `ENABLE_LLM=1` and `OPENROUTER_API_KEY`, capture CSV/logs.

## Qualitative (see evaluation_rubric.md)
- Sample 15 queries covering intents.
- Score each model on 7 dimensions (Answer Quality, Relevance, Naturalness, Correctness, Instruction Following, Explanation Quality, FPL Domain Knowledge) using 1–5 scale.
- Record notes on hallucinations and strengths/weaknesses per model/query.

## Error/Hallucination Checks
- Compare generated facts against provided KG context; flag deviations.
- Prefer concise, stat-referenced answers; avoid invented fixtures/stats.
