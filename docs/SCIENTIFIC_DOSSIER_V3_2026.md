# GGUF / MODEL COMPRESSION SCIENCE

## Исследовательская идея

Research model for quantization quality vs runtime. The current repository is a specification only, defining a valuable multiobjective question: choose tensor/block quantization profiles to minimize artifact size subject to quality and inference resource constraints. It does not yet include an encoder or published model benchmark.

## Научная формализация

Effective bits per weight must include tensor payload and distinguish metadata/scale/tokenizer overhead. A low average bit-width does not prove lower memory footprint or higher tokens per second on a particular runtime.

## Программа независимой верификации

Implement GGUF parser, calibration data controls, real decoder-compatible artifact materialization and tests. Compare quality degradation, benchmark perplexity, RAM/VRAM, prefill and decode speed under frozen model and hardware identities.

## Перспективное применение и критерии доказательности

Возможное применение следует из предметной архитектуры, но промышленная зрелость требует исполняемого артефакта, сравнимого baseline и независимого воспроизведения. Необходимо разделять спецификацию, реализованный компонент, внутренний тест, sealed OOS и фактическое развёртывание.

## Исходные материалы

- [README.md](../README.md)
