# Recent Harness Research Summary for Codex

This note is a short research-grounded ruleset for future work on the translation-quality harness. Its purpose is to keep Codex aligned with the actual evaluation-harness research and avoid drifting into generic, metric-gaming, or project-mixing behavior.

## Core takeaway

Recent harnesses are not just “run a prompt and eyeball the output.” They are moving toward:

1. **Task-specific datasets** with stable inputs and expected behavior.
2. **Repeatable experiment runs** with fixed scorers and result logs.
3. **Application traces**, not only final outputs.
4. **Mixed scoring**, combining deterministic checks, learned/LLM judges, pairwise comparison, and human review gates.
5. **Optimization loops** that improve prompts/rules/program text only after measured benchmark gains.

For this manga translator, Codex must therefore build and maintain a custom harness around the actual translator-agent pipeline, not blindly bolt on a generic eval framework.

---

## What the recent harnesses imply

### 1. OpenAI Evals: define the task, run inputs, analyze, iterate

OpenAI’s eval guidance frames evaluation as a loop: describe the task, run test inputs, analyze results, and iterate. Codex should preserve that pattern:

```text
fixed benchmark case -> run translator agents -> score -> log -> compare -> keep/revert
```

Do not make one-off manual prompt edits without a benchmark result.

### 2. LangSmith: datasets, evaluators, experiments, pairwise comparisons

LangSmith emphasizes curated datasets, evaluators, experiment comparison, and pairwise evaluation. Codex should keep every translation-quality experiment comparable against a baseline and should use pairwise comparison when absolute scores are too noisy.

For the translator, pairwise means:

```text
baseline translation vs candidate translation
```

judged for:

```text
faithfulness
natural manga dialogue
glossary correctness
lack of hallucinated details
lack of Japanese leakage
```

### 3. Promptfoo: CLI/CI-style checks, assertions, custom metrics

Promptfoo is useful as a design model because it treats LLM evaluation as test-driven development with assertions, custom scoring, caching, and CI-style comparison. Codex should copy the idea, not necessarily the dependency.

For this project, deterministic assertions are mandatory before any LLM/MQM judge:

```text
no assistant chatter
no untranslated Japanese leakage unless allowed
no invalid schema
no line-id/source-text mismatch
no glossary violation
no source copied verbatim as fake translation
```

### 4. Inspect AI: task / solver / scorer / logs for agentic workflows

Inspect AI is relevant because it supports agentic evaluations, tools, custom scorers, tracing/logging, and reusable eval components. Codex should structure the local harness similarly:

```text
case dataset -> solver/agent run -> trace -> scorer -> run log
```

The harness must save the translator-agent trace, not just the final English sentence.

Trace fields should include:

```text
source_text
context_before/context_after
glossary_terms
candidate translations
agent that produced each candidate
critic/repair decisions
reranker scores
final accepted translation
reason final candidate was selected
```

### 5. MT research: translation quality needs error-aware scoring

Recent WMT work shows that machine-translation evaluation is moving beyond one flat score. WMT25’s automated translation-quality task uses segment-level quality scoring, span-level error annotation, and quality-informed correction. That maps directly to the manga translator:

```text
quality score      -> how good is this translation?
error spans        -> where did it hallucinate/omit/mistranslate?
error correction   -> can repair improve it safely?
```

Codex should therefore not use BLEU-like lexical overlap alone. Short manga dialogue often has many valid English renderings.

### 6. COMET / CometKiwi / xCOMET: useful signals, not final truth

COMET-family metrics are useful MT-specific signals. CometKiwi-style reference-free scoring is useful when there is no human reference. xCOMET-style error-span scoring is useful because it can point to likely error locations.

But Codex must not treat any learned metric as the sole source of truth. Automatic MT rankings can be biased, especially when systems optimize with QE reranking or MBR-like methods. Keep human-gold cases and hard checks as final gates.

### 7. MQM: use explicit error categories and severity

MQM-style judging is the right mental model for translation quality. Codex should score concrete error types, not vague “good/bad translation.”

Use manga-specific MQM categories:

```text
Accuracy:
  omission
  addition
  mistranslation
  wrong polarity
  hallucinated speaker/name/fact

Fluency:
  broken English
  unnatural phrasing
  awkward literalism

Terminology:
  glossary violation
  inconsistent name/title/honorific

Style:
  too formal
  too verbose
  not manga-dialogue-like
  over-explained joke

Format/safety:
  assistant chatter
  JSON leakage
  line mapping error
  untranslated Japanese leakage
```

Critical and major MQM errors must outweigh fluency gains.

### 8. DSPy / MIPROv2 / GEPA: optimize only controlled text parameters

DSPy optimizers tune program parameters such as prompts and sometimes weights against a metric. MIPROv2 optimizes instructions and few-shot examples. GEPA is especially relevant because it uses execution traces and natural-language feedback to propose prompt updates.

For this repo, Codex may use those ideas only for controlled text/config surfaces:

```text
translation prompt
critic prompt
repair prompt
page-context prompt
vision-facts instruction
glossary injection format
few-shot examples
reranker weights
repair thresholds
```

Codex must not let an optimizer rewrite arbitrary project code, tests, scorers, or benchmark labels.

---

## What Codex must preserve

### Keep the harness separate

The translation-quality harness must remain in its own root-level folder:

```text
translation_quality_autoresearch/
```

Do not move harness files into:

```text
manga_local_translator/
.testing/
quality-runs/
render_autoresearch/
source_extraction_autoresearch/
```

The harness may import the project. It must not become part of the production pipeline unless explicitly approved.

### Preserve replay mode and live mode

Always keep two modes:

```text
Replay mode:
  use frozen agent outputs;
  cheap and deterministic;
  good for testing reranking, scorers, filters, and logs.

Live mode:
  actually call CAT/Qwen/OPUS/repair agents;
  expensive and non-deterministic;
  required for testing prompts and agent behavior.
```

Replay mode should be the default for Codex experiments.

### Score with multiple layers

The correct scoring order is:

```text
1. deterministic hard checks
2. optional MT/QE metrics
3. MQM-style judge
4. pairwise baseline comparison
5. human-gold gate for risky changes
```

Do not replace this with a single LLM judge or a single learned metric.

### Optimize candidate generation + reranking first

The most research-aligned translation improvement is:

```text
generate multiple candidates -> filter -> score/rerank -> accept best -> repair only if needed
```

Not:

```text
one huge prompt rewrite
```

Candidate pools may include:

```text
CAT primary
Qwen direct
Qwen with page context
Qwen repair of CAT/local output
glossary-constrained Qwen
safe phrasebook output for known fragments/SFX
```

### Keep result logs append-only

Every experiment must write one result row and preserve per-case traces.

Required high-level log:

```text
translation_quality_autoresearch/results/translation_quality_results.tsv
```

Required run folder:

```text
translation_quality_autoresearch/runs/<run_id>/
  summary.json
  case_results.jsonl
  traces.jsonl
  failures.jsonl
  artifacts/
```

Codex must never silently overwrite prior results.

---

## Hard rules for future Codex updates

Codex must reject or revert a change if it does any of the following:

```text
edits benchmark labels to improve score
edits scorer weights without documenting scorer version
removes hard checks
hides failures from logs
accepts assistant chatter
accepts untranslated Japanese leakage unless explicitly allowed
increases critical MQM errors
breaks line_id/source_text mapping
optimizes only for fluency while harming faithfulness
lets an optimizer rewrite arbitrary repo code
mixes harness code into the production project
```

If a metric improves but examples look worse, the metric is wrong or incomplete. Add a labeled regression case before trusting the improvement.

---

## What to tell future Codex agents

Use this instruction before any future translation-quality harness work:

```text
You are maintaining a research harness, not just editing prompts.
Preserve fixed benchmarks, append-only logs, per-case traces, deterministic hard checks, MQM-style error categories, pairwise baseline comparison, and human-gold gates.
Use recent harness ideas as design patterns: task-specific evals, trace-aware agent scoring, custom deterministic assertions, MT-specific quality/error metrics, and controlled prompt/program optimization.
Do not treat any single learned metric or LLM judge as ground truth.
Do not move harness code into the production translator package.
```

---

## Research anchors

These are the sources that this summary is based on. Use them to avoid drifting away from the actual research basis:

- OpenAI Evals guide: https://developers.openai.com/api/docs/guides/evals
- LangSmith evaluation docs: https://docs.langchain.com/langsmith/evaluation
- LangSmith pairwise evaluation docs: https://docs.langchain.com/langsmith/evaluate-pairwise
- Promptfoo introduction and assertions/custom metrics docs: https://www.promptfoo.dev/docs/intro/ and https://www.promptfoo.dev/docs/configuration/expected-outputs/
- Inspect AI docs: https://inspect.aisi.org.uk/
- DSPy optimizer docs: https://dspy.ai/learn/optimization/optimizers/
- DSPy MIPROv2 docs: https://dspy.ai/api/optimizers/MIPROv2/
- DSPy GEPA docs: https://dspy.ai/api/optimizers/GEPA/overview/
- GEPA paper: https://arxiv.org/abs/2507.19457
- WMT24 metrics findings: https://aclanthology.org/2024.wmt-1.2.pdf
- WMT25 automated translation-quality task: https://aclanthology.org/2025.wmt-1.24/
- WMT25 preliminary ranking caution: https://arxiv.org/abs/2508.14909
- COMET framework: https://github.com/Unbabel/COMET
- xCOMET paper: https://arxiv.org/abs/2310.10482
- MQM typology: https://themqm.org/error-types-2/typology/
