# Evaluation parameters

Board: Kria KV260 (3930 MB RAM). Vision model: `merged_best.onnx`. LLM: Llama 3.2 1B (ExecuTorch).
Board sources: `metrics_board.txt` / `metrics_board.json` (isolated benchmark: 30 images, 3 Llama runs) and the live dashboard session (`docs/kv260_live/`).

## 2. System: FPS, memory, power

### FPS

| Parameter | Isolated benchmark | Live session (all components running) |
|---|---|---|
| Vision latency (mean) | 579.4 ms (median 577.9, p95 588.4) | 1810.9 ms |
| Vision FPS | 1.73 | 0.6 |
| Breakdown | pre 24.9 ms, model 550.3 ms, post 4.0 ms | not broken down |

The live session is slower probably because vision, the dashboard and Llama run at the same time on one board.

### Memory

| Parameter | Value |
|---|---|
| Board total RAM | 3930 MB |
| Vision process peak | 110.5 MB |
| Llama (`llama_main`) peak | 1159.2 MB |

### Power

**Not measured on this board yet.** The figures below are from the manufacturer's K26 SOM datasheet, not from our experiments.

| Parameter | Value | Basis |
|---|---|---|
| SOM typical power | 7.5 W | datasheet |
| SOM maximum power | 15 W | datasheet |
| Energy per vision frame (0.579 s) | 4.3 J (typical) to 8.7 J (max) | datasheet power x measured latency |
| Energy per Llama alert (65.7 s) | 493 J (typical) to 986 J (max) | datasheet power x measured time |

These are an envelope for the SOM, not our workload's actual draw. In the second evaluation we will read the on-board INA260 current sensor (VCC_SOM rail) while idle, during vision, and during Llama, and report measured watts and energy.

## 3. Llama 3.2 1B (ExecuTorch, on board)

Source: `metrics_board.txt`, 3 runs (Hole, Stain, Knot), format check 3/3 ok.

| Parameter | Value |
|---|---|
| Time to first token | 28.6 s (plus ~29 s model load, reloaded on every call) |
| Tokens per second | decode 5.0, prefill 6.6 |
| Total response time | 65.7 s mean, 66.7 s max per alert |
| Tokens | 189 prompt, 36.3 generated (mean) |
