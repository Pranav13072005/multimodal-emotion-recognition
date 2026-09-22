# Experimental Results

This is the detailed scientific record behind the [`README.md`](README.md) summary —
protocol, numbers, and the reasoning that connects one stage to the next, in the order
the work was actually done. Every notebook referenced below is in
[`notebooks/`](notebooks/) with its real, executed output.

**Model:** `Qwen/Qwen2.5-Omni-3B`, a pretrained multimodal audio-text LLM, used
zero-shot unless a stage says otherwise. No hand-engineered acoustic features
(pitch/VAD/eGeMaPS) are used — the model is given the raw audio waveform and the
ground-truth transcript directly.

**Task:** 6-class speech emotion recognition on IEMOCAP — `happy`, `sad`, `neutral`,
`angry`, `excited`, `frustrated` — kept as six separate classes throughout (happy and
excited are not merged).

**Protocol:** IEMOCAP Sessions 1-3 for training, Session 4 for validation and every
model-selection decision, Session 5 held out completely and touched only once, at the
very end, after all adapters and hyperparameters were already fixed.

---

## 1. Zero-shot baseline

`notebooks/01_baseline_zeroshot.ipynb`

Before touching context or fine-tuning at all, the first question was simply: how well
does this model do on IEMOCAP out of the box? Each Session-5 utterance (N=1622) was
scored independently, with no preceding conversation and no training.

| WA | UA | macro-F1 | weighted-F1 |
|---|---|---|---|
| 38.78% | 38.10% | 34.21% | 33.09% |

This is the number everything downstream is compared against.

## 2. Does dialogue history help? (Hist8)

`notebooks/02_context_ablation_hist8.ipynb`

The next question was whether giving the model the preceding conversation — not just
the isolated target utterance — changes its predictions. Two arms were run over the
identical 1622 target utterances, the only difference being an added 8-turn,
transcript-only window of prior dialogue (speaker identity anonymized to `Speaker_1`/
`Speaker_2`, never gender, and never crossing into another dialogue or session).

| arm | WA | UA | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| No history | 41.68% | 40.13% | 38.28% | 38.76% |
| 8-turn history | 44.88% | 44.78% | 42.16% | 41.85% |

Paired comparison over the same 1622 utterances: **+3.88 macro-F1** with history added.
On average, history clearly helps.

## 3. A closer look — history doesn't help every class

Averages can hide unevenness, so the per-class breakdown was checked before concluding
anything. The `frustrated` class went the *other* way: recall fell from 27.82% (no
history) to 24.41% (8-turn history). So the aggregate gain from Experiment 2 is real,
but it's not uniform — something about longer dialogue context is actively confusing
the model on this one class. That observation is what motivated the next experiment:
does a *shorter* window keep most of the benefit while doing less of this damage?

## 4. A shorter context window (Hist3)

`notebooks/03_context_window_hist3.ipynb`

Same design as Experiment 2, but the window was reduced to the 3 most recent turns.

| arm | WA | UA | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| 3-turn history | 44.45% | 44.26% | 42.03% | 41.72% |

The aggregate gain barely moved (42.03 vs. 42.16 macro-F1 for the 8-turn window), but
the `frustrated`-class harm roughly halved: recall only dropped to 25.98% (vs. 24.41%
with 8 turns), against the same 27.82% no-history baseline. Shorter context keeps
almost all of the benefit while causing noticeably less damage to this one class.

## 5. Is the model's agreement a reliability signal?

`notebooks/04_reliability_calibration.ipynb`

Given the mixed picture above, a natural next question: when the no-history and
3-turn-history predictions *agree*, is that agreement actually trustworthy — a usable
signal that the prediction is more likely correct? This was checked on a 1200-utterance
sample stratified from Sessions 1-4 (disjoint from Session 5, sample size fixed in
advance by a power analysis, not chosen after seeing any result).

| | |
|---|---|
| Agreement rate | 81.75% (981 / 1200) |
| Accuracy when the two arms agree | 46.89% |
| Accuracy when they disagree — no-history side | 26.48% |
| Accuracy when they disagree — 3-turn-history side | 35.62% |
| Separation: agree vs. no-history-disagree | χ² p = 5.4 × 10⁻⁸ |
| Separation: agree vs. history-disagree | χ² p = 3.1 × 10⁻³ |
| Paired comparison, no-history vs. history | +2.54 macro-F1 |

Agreement between the two context settings is a statistically robust reliability
signal — accuracy is nearly twice as high when they agree as when they don't — and this
held up on data that never touched Session 5, independently of the Session-5 result.

## 6. Checking the model's architecture before fine-tuning

`notebooks/05_lora_architecture_smoketest.ipynb`

Before any fine-tuning was attempted, the model's actual structure was inspected
directly rather than assumed. This turned up something worth recording: the top-level
`Qwen2_5OmniForConditionalGeneration` class has no usable `forward()` for computing a
training loss — confirmed by directly inspecting its class hierarchy, not inferred from
an error message. It's built entirely around a custom `generate()` that internally
dispatches to an inner `thinker` submodule. Practically, this means training has to
call `model.thinker(...)` directly, while inference correctly keeps using the normal
`model.generate(...)` interface.

With that resolved, LoRA (rank 16, alpha 16, dropout 0.05) was applied to all 144
attention-projection modules in the language-model decoder
(`thinker.model.layers.{0-35}.self_attn.{q,k,v,o}_proj`):

| trainable parameters | total parameters | percentage |
|---|---|---|
| 7,372,800 | 5,544,493,440 | 0.133% |

Audio, vision, and speech-output pathways were verified frozen (zero trainable
parameters outside the language-model decoder). A 7-stage smoke test — device
placement, forward/loss, backward pass, gradient isolation, optimizer step, checkpoint
save/reload/resume, and post-training generation — passed all 7 stages before any full
training run was attempted.

## 7. Fine-tuning two adapters

`notebooks/06_lora_finetuning_nohist.ipynb` and `notebooks/07_lora_finetuning_hist3.ipynb`

Two LoRA adapters were trained independently, each on only its own input format — one
never saw dialogue history during training, the other always did (3-turn window,
matching Experiment 4). Both trained on Sessions 1-3 (N=4246) and were validated once
per epoch on the full Session 4 set (N=1512), in their own matching format. Training
used AdamW (lr = 1e-4), an effective batch size of 8, fp16, and gradient checkpointing,
for up to 8 epochs with early stopping (patience 3) and checkpoint selection by
macro-F1 on validation data.

| adapter | best epoch | macro-F1 | WA | UA | weighted-F1 |
|---|---|---|---|---|---|
| No-history | 2 | 58.65% | 61.38% | 59.83% | 61.45% |
| 3-turn-history | 3 | 63.74% | 65.48% | 64.97% | 65.76% |

At this point it would be tempting to say "the history-aware adapter is simply
better." The final stage below is specifically designed to check whether that's true,
or whether it's a more specific effect.

## 8. The untouched held-out evaluation

`notebooks/08_final_evaluation_2x2.ipynb`

Both adapters were frozen — selected purely on Session-4 validation, no further tuning
— and then evaluated on the full Session-5 test set (N=1622) in **both** input formats,
not just their own. This 2×2 design is what separates a *training-context* effect from
an *inference-time-context* effect, rather than only comparing the two matched pairs.

| macro-F1 | No-history eval | 3-turn-history eval |
|---|---|---|
| **No-history adapter** | 60.19% | 61.80% |
| **3-turn-history adapter** | 57.79% | **66.78%** |

The interaction between training format and inference format works out to
**+7.38 macro-F1** (computed two algebraically equivalent ways from the same table —
both come out identical, which is a useful internal consistency check). Concretely:
giving a model dialogue context at inference time helps far more when it was *also*
trained with that context (+9.00 macro-F1) than when it wasn't (+1.61 macro-F1). And
the history-aware adapter, evaluated *without* the context it was trained to expect,
actually falls **below** the no-history adapter's own matched result (57.79% vs.
60.19%).

**On interpretation:** the matched-pair comparison in Section 7 (58.65% vs. 63.74%) is
an end-to-end, matched-system comparison — it does not by itself establish that
training with history *causes* a general improvement. What the 2×2 grid shows is more
specific: the benefit of dialogue context is conditional on training and inference
format matching. A history-aware model deprived of context at test time does not simply
retain its advantage — it can do worse than a model that never needed context at all.
Any claim about the value of "context" in this task should be read as a statement about
that interaction, not as a context main effect.
