"""Post-run output checks on the 1,000 test translations.

For every system this counts outputs that copy the English source (untranslated),
outputs that contain an English preamble instead of only the translation,
outputs that are far too short or too long, and BLEU with the copied outputs removed.
It also recomputes the sentence-length quality table used in the README.
Run from the repository root:  python analysis/output_checks.py
Writes metrics/output_checks.csv and metrics/chrf_by_length.csv
"""
import re
import pandas as pd
import sacrebleu

SYSTEMS = {
    "transformer": "Basic Transformer",
    "mt5": "mT5-base",
    "llama_fewshot": "LLaMA-3.2-3B-Instruct (3-shot)",
    "llama_lora": "LLaMA-3.2-3B-Instruct + QLoRA",
    "gemma_fewshot": "Gemma-3-1B-it (3-shot)",
    "gemma_lora": "Gemma-3-1B-it + LoRA",
}
PREAMBLE = re.compile(r"\b(here'?s|here is|translation:|translation in french|voici (la|le) traduction|traduction en français)", re.I)
BUCKETS = [(3, 8, "3 to 8"), (9, 15, "9 to 15"), (16, 25, "16 to 25"), (26, 35, "26 to 35"), (36, 999, "36 or more")]

test = pd.read_csv("data/test.tsv", sep="\t", keep_default_na=False)
src, ref = test["en"].tolist(), test["fr"].tolist()
rows, length_rows = [], []
for key, name in SYSTEMS.items():
    hyp = open(f"predictions/{key}.txt", encoding="utf-8").read()[:-1].split("\n")
    assert len(hyp) == len(src), f"{key}: {len(hyp)} lines"
    ratio = [len(h.split()) / max(1, len(s.split())) for h, s in zip(hyp, src)]
    copied = {i for i, (h, s) in enumerate(zip(hyp, src)) if sacrebleu.sentence_chrf(h, [s]).score > 70}
    keep = [i for i in range(len(src)) if i not in copied]
    rows.append({
        "System": name,
        "Untranslated (copied English)": len(copied),
        "English preamble in output": sum(bool(PREAMBLE.search(h)) for h in hyp),
        "Too short (<50% of source length)": sum(r < 0.5 for r in ratio),
        "Too long (>200% of source length)": sum(r > 2.0 for r in ratio),
        "BLEU (all 1,000)": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
        "BLEU (copies removed)": round(sacrebleu.corpus_bleu([hyp[i] for i in keep], [[ref[i] for i in keep]]).score, 2),
    })
    if key in ("transformer", "mt5", "llama_lora", "gemma_lora"):
        row = {"System": name}
        for lo, hi, label in BUCKETS:
            idx = [i for i, s in enumerate(src) if lo <= len(s.split()) <= hi]
            row[label] = round(sum(sacrebleu.sentence_chrf(hyp[i], [ref[i]]).score for i in idx) / len(idx), 2)
        row["Sentences per bucket"] = "/".join(str(sum(lo <= len(s.split()) <= hi for s in src)) for lo, hi, _ in BUCKETS)
        length_rows.append(row)

out = pd.DataFrame(rows)
out.to_csv("metrics/output_checks.csv", index=False)
print(out.to_string(index=False))
lengths = pd.DataFrame(length_rows)
lengths.to_csv("metrics/chrf_by_length.csv", index=False)
print()
print(lengths.to_string(index=False))
