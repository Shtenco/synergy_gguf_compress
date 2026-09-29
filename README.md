# 🧠 SYNERGY GGUF Compress — verified specification dossier

> **Домен:** AI Model Compression  
> **Роль:** спецификация компрессии GGUF/LLM с quality/runtime gates вместо одного ratio.  
> **Статус:** 🧱 **STUB / specification-only** — на проверенном `main` присутствует только `README.md`; executable runtime, tests и evidence artifacts отсутствуют.  
> **Проверено по фактическому root tree:** 29.09.2026.  
> **Архитектурный родитель:** [`synergy_meta_os`](https://github.com/Shtenco/synergy_meta_os)

## 1. 🎯 Проблема и граница проекта

Этот repository резервирует отдельную ответственность внутри SYNERGY: спецификация компрессии GGUF/LLM с quality/runtime gates вместо одного ratio.

Ключевой принцип документации здесь — не путать подробно описанную будущую архитектуру с существующей реализацией:

```text
specification != implementation
architecture diagram != runtime integration
roadmap != completed feature
repository name != security/performance proof
```

Пока в `main` нет кода, strongest honest status — **STUB**.

## 2. 📦 Фактическое состояние `main`

```text
synergy_gguf_compress/
└── README.md
```

Не обнаружены в root tree:

- `src/` / `runtime/`;
- `tests/`;
- build/dependency manifest;
- CI workflow;
- persisted evidence;
- release/deployment artifact.

Это фиксируется как отрицательный, но полезный факт maturity audit.

## 3. 🧠 Целевой design contract

Будущая реализация должна иметь явную цепочку:

```mermaid
flowchart LR
    I[Typed inputs] --> V[Deterministic validation]
    V --> R[Bounded runtime]
    R --> E[Evidence]
    E --> C[Typed contract]
    C --> D[Downstream authority]
```

Runtime не получает дополнительные полномочия только потому, что находится внутри SYNERGY.

## 4. 📥 Планируемые inputs

- exact GGUF/model artifact;
- tensor metadata;
- target bit budget;
- calibration data;
- hardware/runtime target;
- baseline codec/model metrics;

Каждый input должен иметь version/source/provenance там, где это влияет на воспроизводимость или безопасность.

## 5. 📤 Планируемые outputs

- compressed artifact;
- decoder/runtime config;
- artifact hashes;
- compression ratio/bpw;
- PPL/accuracy delta;
- RAM/VRAM/tokens-s metrics;

Output становится federation evidence только после schema validation и, при необходимости, cryptographic/content-addressed binding.

## 6. 🛡️ Authority boundaries

Обязательное системное правило:

```text
component output != downstream authorization
local state      != global/canonical truth
self-reported success != observed evidence
generated artifact != verified artifact
```

Canonical accounting, settlement, risk, governance, identity или deployment authority остаётся у соответствующего domain owner.

## 7. 🔐 Threat / failure model

| Failure mode | Promotion requirement |
|---|---|
| size projection without materialized artifact | mandatory reject/mitigation evidence before promotion |
| tensor error mistaken for model quality | mandatory reject/mitigation evidence before promotion |
| compression slows inference | mandatory reject/mitigation evidence before promotion |
| unsupported GGUF tensor types | mandatory reject/mitigation evidence before promotion |
| metadata corruption | mandatory reject/mitigation evidence before promotion |
| baseline mismatch | mandatory reject/mitigation evidence before promotion |

Threat model обязан становиться executable через negative tests, а не оставаться только prose.

## 8. 🧪 Минимальная test matrix

- [ ] byte/header integrity;
- [ ] load in reference runtime;
- [ ] tensor coverage;
- [ ] full-model PPL;
- [ ] task accuracy;
- [ ] CPU/CUDA tokens-s;
- [ ] RAM/VRAM;
- [ ] round-trip metadata;
- [ ] multiple models;

Дополнительно обязательны malformed-input tests, crash/restart behavior и deterministic fixtures для core logic.

## 9. 📊 Evidence model

Strong evidence package будущего runtime должен связывать:

```text
input/event id
input hash/provenance
software commit
configuration/policy version
result/output
test or execution status
artifact hash
timestamp
limitations
```

Без такого bundle README не должен использовать GREEN/E2E wording.

## 10. 🏗️ Place in SYNERGY

```mermaid
flowchart LR
    SYS[synergy_system] --> P[synergy_meta_os]
    P --> THIS[synergy_gguf_compress]
    THIS --> E[typed evidence]
    E --> NEXT[downstream canonical authority]
```

Стрелка показывает target architecture, **не runtime dependency**.

## 11. 🛠️ Reproducibility текущего состояния

Сегодня воспроизводить runtime нечего. Проверяемый факт:

```bash
git clone https://github.com/Shtenco/synergy_gguf_compress.git
cd synergy_gguf_compress
find . -maxdepth 2 -type f
```

Предметный результат текущего `main`: `README.md`.

Когда появится runtime, README обязан заменить этот раздел exact build/test commands и dependency versions.

## 12. 🗺️ Current vs required repository map

| Surface | Current | Required for CANDIDATE |
|---|---:|---:|
| repository-specific specification | ✅ | ✅ |
| source/runtime | ❌ | ✅ |
| typed schema/contract | ❌ | ✅ |
| unit tests | ❌ | ✅ |
| negative/security tests | ❌ | ✅ |
| CI | ❌ | ✅ |
| evidence artifacts | ❌ | ✅ |
| reproducible build/run | ❌ | ✅ |

## 13. ✅ STUB → CANDIDATE promotion gate

Promotion is allowed only after all of the following exist:

1. executable minimal core;
2. typed input/output schema;
3. deterministic unit tests;
4. at least one negative/failure test per major invariant;
5. pinned dependencies/toolchain;
6. CI command;
7. real repository artifact/evidence;
8. README updated from target tense to implemented tense only for verified pieces.

## 14. ✅ CANDIDATE → E2E_TESTED gate

Requires:

- upstream producer and downstream consumer;
- exact payload binding;
- observed execution;
- persisted evidence;
- verifier;
- no authority bypass;
- reproducible run.

A local unit test is insufficient.

## 15. 📊 Current evidence maturity

```text
Problem/domain definition     ✅
Architecture specification   ✅
Inputs/outputs specification ✅
Threat/failure model          ✅ documented
Executable implementation    ❌
Tests                        ❌
Observed evidence            ❌
E2E                          ❌
Production                   ❌
```

## 16. 🚀 Roadmap

1. formalize v1 typed contract;
2. implement smallest executable core;
3. add deterministic fixtures;
4. add failure/security tests;
5. add CI;
6. generate content-addressed evidence;
7. integrate one upstream/downstream pair;
8. run observed federation gate;
9. only then upgrade maturity.

## 17. 🛑 Что проект сейчас НЕ утверждает

- что runtime существует;
- что integration уже работает;
- что security/performance claims доказаны;
- что target diagrams соответствуют deployed topology;
- что roadmap items completed;
- что STUB может считаться E2E/production.

## 18. 📚 Federation navigation

- [SYNERGY SYSTEM](https://github.com/Shtenco/synergy_system)
- [Атлас 75 репозиториев](https://github.com/Shtenco/synergy_system/blob/main/docs/SYNERGY_REPOSITORY_ATLAS.md)
- [Documentation quality registry](https://github.com/Shtenco/synergy_system/blob/main/registry/DOCUMENTATION_QUALITY.json)

**Итог:** repository теперь имеет подробный предметный specification dossier, но status сознательно остаётся 🧱 STUB до появления кода и evidence.
