# Multimodal Emotion Recognition

**Student:** Pranav Jaiganesh  
**Supervisor:** Prof. Chng Eng Siong, Nanyang Technological University  
**Project Type:** Remote Research Internship  
**Duration:** August 2026 – February 2027 or more
**Working With:** Ahire Vrushank Ajay and Hoang Anh  

## Current Direction

Following the initial literature review, the project has moved into an empirical
phase. The current work centers on evaluating a pretrained audio-LLM
(Qwen2.5-Omni-3B) on IEMOCAP for zero-shot speech emotion recognition, and then
studying how two things affect its performance: dialogue context (how much of the
preceding conversation the model is given) and parameter-efficient fine-tuning
(LoRA). All of this follows a strict split — Sessions 1-3 for training, Session 4
for validation and model selection, and Session 5 held out and untouched until the
very end.

Concretely, the work so far has:

- established a zero-shot baseline,
- tested whether giving the model recent conversation history helps, and at what
  window length, including a look at which emotion classes benefit and which don't,
- run an independent check confirming that the model's own agreement between
  context settings is a meaningful reliability signal, and
- fine-tuned two separate LoRA adapters — one with dialogue context, one without —
  and evaluated both, in both settings, on the held-out test set.

The current focus is a specific finding from that final evaluation: the benefit of
giving the model dialogue context at inference time depends heavily on whether it
was trained with that context in the first place. Working out what that implies for
how context should be used in this task is the next step.

Full experimental detail, protocol, and numbers are in [`RESULTS.md`](RESULTS.md).
The executed notebook for every stage is in [`notebooks/`](notebooks/).

## Updates

Progress and findings will be documented here as the work develops.

## Progress Updates

| Update | Date | Summary | Video |
|---|---|---|---|
| 01 | August 2026 | Project setup, literature review, and research direction exploration | To be added |
| 02 | September 2026 | Built a zero-shot IEMOCAP baseline (macro-F1 34.2%), then explored how dialogue context affects performance — an 8-turn history window helped on average but hurt the frustrated class specifically, which motivated trying a shorter 3-turn window that kept most of the benefit with less of the harm. Ran an independent reliability check on separate validation data to confirm the finding held up. Fine-tuned two LoRA adapters (with and without context) and evaluated both on the untouched held-out test set in both settings — giving the model context at inference time helps far more when it was also trained with context (interaction effect of +7.4 macro-F1) | To be added |
