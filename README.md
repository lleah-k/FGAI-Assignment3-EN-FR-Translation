# English to French Machine Translation: Transformer, mT5, LLaMA and Gemma

**Course:** Foundations of Generative AI (ICT4403), Assignment 3
**Student:** K Liya
**Registration Number:** 230953118
**Section:** FGAI F

**Title of the developed system:** Europarl English to French Translation Benchmark: Scratch Transformer, mT5-base, LLaMA 3.2 3B (QLoRA) and Gemma 3 1B (LoRA)

---

## Contents

1. [Problem Statement and Objective](#1-problem-statement-and-objective)
2. [Dataset Description and Source Link](#2-dataset-description-and-source-link)
3. [Models Implemented](#3-models-implemented)
4. [Execution Steps and Required Dependencies](#4-execution-steps-and-required-dependencies)
5. [Results Obtained](#5-results-obtained)
6. [Model Comparison and Evaluation](#6-model-comparison-and-evaluation)
7. [Sampling-Temperature Analysis](#7-sampling-temperature-analysis)
8. [Conclusion: Best-Performing Model](#8-conclusion-best-performing-model)

---

## 1. Problem Statement and Objective

**Problem statement.** Machine translation is an autoregressive text-generation task: the system reads a sentence in one language and generates the equivalent sentence in another language, one token at a time. Two Transformer designs are widely used for this. Encoder-decoder models encode the full source sentence with bidirectional attention and generate the translation with a decoder that attends to that encoding. Decoder-only large language models receive the source sentence inside a prompt and continue it with the translation. These designs differ in size, pretraining, tokenization and the way they are adapted to translation, and the trade-off between their quality and their computational cost is the subject of this study.

**Objective.** With English as the source language and French as the target language, this project:

1. prepares a public English-French parallel corpus and splits it into training, validation and test sets;
2. trains a basic Transformer from scratch, fine-tunes mT5, and adapts LLaMA and Gemma with few-shot prompting and parameter-efficient fine-tuning (QLoRA and LoRA);
3. translates one shared test set with all four models under comparable settings;
4. evaluates them with BLEU, chrF, training time, inference time and number of parameters, and compares the encoder-decoder and decoder-only architectures;
5. measures the effect of the sampling temperature; and
6. examines the translations for bias, hallucinated information, cultural misrepresentation and mishandling of sensitive content.

## 2. Dataset Description and Source Link

| Item | Details |
|---|---|
| Dataset | Europarl Parallel Corpus v7, English-French |
| Source link | https://huggingface.co/datasets/Helsinki-NLP/europarl (config `en-fr`) |
| Original release | https://www.statmt.org/europarl/ (Koehn, 2005) |
| Language pair | English (source) to French (target) |
| Content | Proceedings of the European Parliament, translated by professional translators |
| Size | 2,051,014 sentence pairs, provided as a single `train` split |
| Record format | `{"translation": {"en": "...", "fr": "..."}}` |

Europarl was chosen because it is large, sentence aligned and professionally translated, so the models can reach high scores, and French is a high-resource language for all four models.

### 2.1 Preprocessing

1. A random pool of 450,000 pairs was drawn with a fixed seed (2026), because training on all 2 million pairs is not possible within the Kaggle time limit.
2. Unicode NFC normalisation and whitespace clean-up.
3. Removal of procedural lines such as "(Applause)" or "(The sitting was closed)", empty or untranslated pairs, pairs shorter than 3 or longer than 50 words, and pairs whose length ratio is above 2.0.
4. Removal of duplicate English sentences (case-insensitive) **before** splitting, so no test or validation sentence appears in the training data. The notebook checks this with an assertion.
5. In total 50,055 pairs were removed in steps 3 and 4, leaving 399,945 clean, unique pairs.
6. The clean pairs were shuffled and split into test, validation and training sets.
7. Tokenization is specific to each model: separate English and French SentencePiece unigram vocabularies (10,000 pieces each) for the basic Transformer, and each pretrained model's own tokenizer for mT5, LLaMA and Gemma.

### 2.2 Training, validation and test sets

| Split | Pairs | Mean English words | Mean French words |
|---|---|---|---|
| Training | 250,000 | 22.56 | 23.51 |
| Validation | 1,000 | 22.72 | 23.41 |
| Test | 1,000 | 22.76 | 23.98 |

Europarl has no official validation or test split, so all three sets come from the same cleaned pool and share the same parliamentary style. The training set is large and the evaluation sets are small, which is standard practice in machine translation. One thousand test sentences give a stable corpus-level BLEU score while keeping the six inference passes and the temperature study within the GPU budget.

Each model was trained on a subset of the same 250,000 training pairs, sized to its compute cost. The subsets are prefixes of one shuffled list, so all models see data from the same distribution.

| Model | Training pairs used | Reason |
|---|---|---|
| Basic Transformer | 250,000 | Starts from random weights and needs the most data |
| mT5-base | 40,000 | Pretrained and 23 times larger than the Transformer |
| LLaMA 3.2 3B (QLoRA) | 8,000 | Already translates; QLoRA adapts it to parliamentary French |
| Gemma 3 1B (LoRA) | 10,000 | Already translates; smaller than LLaMA, so more pairs fit in the time budget |

The exact test and validation pairs are in `data/`, and the statistics in `data/dataset_stats.json`.

![Sentence length distribution](plots/dataset_length_hist.png)

*The single taller bar near 26 words in each histogram is a binning effect (45 bins over a 47-word range place two whole-number lengths in one bin), not a feature of the data.*

## 3. Models Implemented

| Model | Architecture | Parameters | Pretraining | Tokenizer | Adaptation |
|---|---|---|---|---|---|
| Basic Transformer | Encoder-decoder, 6 encoder and 3 decoder layers, d_model 384, 6 heads | 25.6M | None | Separate English and French SentencePiece unigram, 10K each | Trained from scratch, 12 epochs |
| mT5-base (`google/mt5-base`) | Encoder-decoder, 12 encoder and 12 decoder layers, d_model 768 | 582.4M | mC4 corpus, 101 languages, span corruption | SentencePiece, 250K | Full fine-tuning, 1 epoch |
| LLaMA 3.2 3B Instruct (`meta-llama/Llama-3.2-3B-Instruct`) | Decoder-only, 28 layers, d_model 3072, grouped-query attention | 3.21B | Multilingual web text, instruction tuned; French officially supported | BPE, 128K | 3-shot prompting, then QLoRA (4-bit, rank 8), 1 epoch |
| Gemma 3 1B Instruct (`google/gemma-3-1b-it`) | Decoder-only, 26 layers, d_model 1152, local and global attention | 1.0B | Text in over 140 languages, instruction tuned | SentencePiece, 262K | 3-shot prompting, then LoRA (rank 16), 1 epoch |

**Basic Transformer.** Built from PyTorch encoder and decoder layers with pre-layer normalisation, GELU activations and learned positional embeddings. It uses a deep encoder and a shallow decoder (6 and 3 layers): the encoder runs once per sentence while the decoder runs once per output token, so moving depth to the encoder makes translation faster at little cost in quality. Training used AdamW (peak learning rate 7e-4, weight decay 0.01) with 2,000 warm-up steps and cosine decay, label smoothing of 0.1, dropout of 0.15, float16 mixed precision and batches of 160 pairs across both GPUs. The epoch with the best validation chrF (epoch 12, 63.17) was kept.

**mT5-base.** The multilingual T5 model was pretrained to fill in masked text, not to translate, so all its weights were fine-tuned with the input prefix `translate English to French:`. Training used Adafactor with a constant learning rate of 5e-4 and batches of 12, in float32 because mT5 produces invalid (NaN) losses in float16 and the T4 GPU does not support bfloat16.

**LLaMA 3.2 3B and Gemma 3 1B.** Both models received the same instruction through their own chat templates:

> Translate the following English sentence into French. Reply with only the French translation.
> English: *source sentence*

* **3-shot prompting (no training).** Three fixed English-French pairs from the training set were placed before the sentence to translate, showing the model the expected style and output format without changing any weights. This is the baseline for each LLM.
* **LLaMA 3.2 3B + QLoRA.** The model was loaded in 4-bit NF4 quantisation with double quantisation, so the 3B model fits in under 5 GB of GPU memory. LoRA adapters of rank 8 were trained on the attention projections (4.6M trainable parameters) on GPU 0.
* **Gemma 3 1B + LoRA.** The model ran in float32 on GPU 1, because Gemma 3 activations overflow in float16 on T4 GPUs. LoRA adapters of rank 16 were trained on the attention and MLP projections (13.0M trainable parameters).
* Fine-tuning used the plain instruction (without examples), with the loss computed only on the French answer tokens.

**Comparable settings.** All systems translated the same 1,000 test sentences with greedy decoding and were scored with corpus-level sacreBLEU (BLEU and chrF, default 13a tokenization).

## 4. Execution Steps and Required Dependencies

### 4.1 Dependencies

Listed in `requirements.txt` and installed by the first notebook cell:

| Package | Version |
|---|---|
| Python | 3.10 or newer |
| torch | 2.1 or newer, with CUDA |
| transformers | 4.57.1 |
| peft | 0.17.1 |
| bitsandbytes | 0.45 or newer (4-bit quantisation for QLoRA) |
| datasets | 3.6.0 |
| huggingface_hub | below 1.0 |
| accelerate, sentencepiece, sacrebleu, numpy, pandas, matplotlib | recent versions |

**Hardware used:** Kaggle notebook with 2 x NVIDIA Tesla T4 GPUs (16 GB each). The complete run took about 3.5 hours.

### 4.2 Steps to run

1. Create a Hugging Face account and accept the licences on https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and https://huggingface.co/google/gemma-3-1b-it.
2. Create a Hugging Face access token of type **Read** (Settings, then Access Tokens).
3. On Kaggle, create a new notebook and import `en_fr_translation_benchmark.ipynb` (File, then Import Notebook).
4. In Session options, set **Accelerator** to **GPU T4 x2** and turn **Internet** on.
5. Under **Add-ons, then Secrets**, add a secret named `HF_TOKEN` containing the token, and attach it to the notebook.
6. Click **Save Version**, then **Save & Run All (Commit)**. Set `QUICK_TEST = True` in the configuration cell first if you want a short trial run of about 25 minutes.
7. When the run finishes, download the output folder. It contains every result folder in this repository.
8. Optional: from the repository root, run `python analysis/output_checks.py` to reproduce the output checks in Section 6.2 and the sentence-length table in Section 5.3.

### 4.3 Repository structure

```
.
├── README.md                              this report
├── en_fr_translation_benchmark.ipynb      complete implementation, with outputs from the run
├── requirements.txt                       dependencies
├── analysis/output_checks.py              post-run checks on the test translations
├── data/                                  dataset statistics and the exact test and validation pairs
├── predictions/                           one file per system, one translation per test sentence
├── metrics/                               per-model metrics, results table, training histories, temperature table
├── plots/                                 all figures used in this README
├── samples/                               sample translations and temperature examples
├── temperature/                           raw outputs of the temperature study
├── ethics/                                ethics probe translations and automatic flags
└── logs/run_log.txt                       timestamped log of the full run
```

## 5. Results Obtained

### 5.1 Main results on the test set (1,000 sentences, greedy decoding)

**Translation quality**

| Model | Architecture | Adaptation | BLEU | chrF |
|---|---|---|---|---|
| **Basic Transformer** | Encoder-decoder | Trained from scratch | **35.09** | **61.15** |
| mT5-base | Encoder-decoder | Full fine-tuning | 29.06 | 56.64 |
| LLaMA 3.2 3B | Decoder-only | 3-shot prompting (no training) | 29.69 | 58.03 |
| LLaMA 3.2 3B + QLoRA | Decoder-only | QLoRA fine-tuning (r=8) | 32.18 | 59.25 |
| Gemma 3 1B | Decoder-only | 3-shot prompting (no training) | 26.89 | 55.75 |
| Gemma 3 1B + LoRA | Decoder-only | LoRA fine-tuning (r=16) | 32.67 | 59.19 |

**Computational requirements**

| Model | Total parameters | Trainable parameters | Training pairs | Training time (min) | Inference time (s) | Sentences per second | Peak GPU memory (GB) |
|---|---|---|---|---|---|---|---|
| Basic Transformer | 25.6M | 25.6M | 250,000 | 38.25 | 4.0 | 248.70 | 0.81 |
| mT5-base | 582.4M | 582.4M | 40,000 | 55.59 | 40.9 | 24.45 | 5.63 |
| LLaMA 3.2 3B (3-shot) | 3.21B | 0 | 0 | Not applicable | 471.0 | 2.12 | 2.92 |
| LLaMA 3.2 3B + QLoRA | 3.21B | 4.6M | 8,000 | 32.76 | 264.9 | 3.78 | 4.68 |
| Gemma 3 1B (3-shot) | 1.00B | 0 | 0 | Not applicable | 334.1 | 2.99 | 4.61 |
| Gemma 3 1B + LoRA | 1.00B | 13.0M | 10,000 | 35.76 | 173.4 | 5.77 | 4.36 |

* **Training time** is wall-clock time for tokenizer and model training (Transformer), fine-tuning (mT5) or QLoRA and LoRA training (LLaMA, Gemma).
* **Inference time** is the time to translate all 1,000 test sentences on the GPU, including tokenization.
* The four main systems compared in this report are the Basic Transformer, mT5-base, LLaMA + QLoRA and Gemma + LoRA. The 3-shot rows show what the pretrained LLMs achieve with prompting alone.

![BLEU and chrF of all systems](plots/quality_bleu_chrf.png)

![Parameters, training time and memory](plots/compute_comparison.png)

![Speed against quality](plots/speed_vs_quality.png)

### 5.2 Training curves

| Model | Training progress |
|---|---|
| Basic Transformer | Training loss fell from 5.43 to 2.43 over 12 epochs; validation BLEU rose from 10.02 to 37.45 and validation chrF from 36.22 to 63.17, still improving at the last epoch |
| mT5-base | One epoch of 3,334 steps; validation loss 1.08 and validation BLEU 30.86 at the end |
| LLaMA 3.2 3B + QLoRA | Answer-token loss fell from 1.03 to about 0.85 over 500 steps |
| Gemma 3 1B + LoRA | Answer-token loss fell from 1.78 to about 1.03 over 625 steps |

![Basic Transformer training](plots/loss_transformer.png)

The other curves are in `plots/loss_mt5.png`, `plots/loss_llama.png` and `plots/loss_gemma.png`.

### 5.3 Quality by sentence length (mean sentence-level chrF)

| Model | 3 to 8 words | 9 to 15 | 16 to 25 | 26 to 35 | 36 or more |
|---|---|---|---|---|---|
| Basic Transformer | 61.04 | 60.65 | 59.58 | 61.43 | 62.26 |
| mT5-base | 53.03 | 56.58 | 55.65 | 56.56 | 57.17 |
| LLaMA 3.2 3B + QLoRA | 55.64 | 57.60 | 57.76 | 59.72 | 60.64 |
| Gemma 3 1B + LoRA | 54.61 | 58.30 | 57.75 | 59.32 | 60.63 |
| *Test sentences in the group* | *81* | *210* | *340* | *231* | *138* |

![chrF by sentence length](plots/chrf_by_length.png)

### 5.4 Sample translations

| System | Translation |
|---|---|
| English source | In Burma the army is pacifying regions inhabited by ethnic minorities. |
| French reference | En Birmanie, l'armée pacifie les régions peuplées de minorités ethniques. |
| Basic Transformer | En Birmanie, l'armée pacifie des régions incrifiantes de minorités ethniques. |
| mT5-base | En Birmanie, l'armée pacte des régions habitées par des minorités ethniques. |
| LLaMA 3.2 3B (3-shot) | L'armée du Birmanie pacifie les régions peuplées par des minorités ethniques. |
| LLaMA 3.2 3B + QLoRA | En Birmanie, l'armée pacifie des régions habitées par des minorités ethniques. |
| Gemma 3 1B (3-shot) | Dans la Birmanie, l'armée pacifie les régions peuplées de minorités ethniques. |
| Gemma 3 1B + LoRA | En Birmanie, l'armée pacifie les régions habitées par des minorités ethniques. |

More examples: `samples/sample_translations.md` (20 random test sentences, all systems side by side) and `samples/all_test_translations.csv` (all 1,000 test sentences).

## 6. Model Comparison and Evaluation

### 6.1 Encoder-decoder vs decoder-only

| | Encoder-decoder (Transformer, mT5) | Decoder-only (LLaMA + QLoRA, Gemma + LoRA) |
|---|---|---|
| Mean BLEU / chrF | 32.08 / 58.90 | 32.43 / 59.22 |
| Parameters | 25.6M to 582.4M | 1.0B to 3.21B |
| Mean inference speed | 136.6 sentences per second | 4.8 sentences per second |
| Peak GPU memory | 0.81 to 5.63 GB | 4.36 to 4.68 GB |
| Training data used | 40,000 to 250,000 pairs | 8,000 to 10,000 pairs |

**Quality.** The two architecture groups reach almost the same average quality, but the best single system is the **Basic Transformer** (35.09 BLEU, 61.15 chrF). It leads Gemma + LoRA by 2.42 BLEU and LLaMA + QLoRA by 2.91 BLEU, and it scores highest in every sentence-length group. Gemma + LoRA and LLaMA + QLoRA are almost tied (Gemma is 0.49 BLEU ahead, LLaMA is 0.06 chrF ahead). mT5-base is last among the four main systems (29.06 BLEU).

**Why the smallest model wins here.** Europarl is a narrow, formal domain, and the test sentences come from the same corpus as the training data. The Transformer saw 250,000 in-domain pairs over 12 epochs and learned the fixed vocabulary of parliamentary debates ("rapporteur", "Commission", "Member States") very well. The LLMs are general-purpose models adapted with only 8,000 to 10,000 pairs. On a broader or out-of-domain test set the pretrained models would be expected to do relatively better, because their knowledge is not limited to one corpus.

**Cost.** The Basic Transformer is also by far the cheapest system. It translates 248.7 sentences per second, which is 43 times faster than Gemma + LoRA and 66 times faster than LLaMA + QLoRA, and it needs only 0.81 GB of GPU memory. It delivers 137 BLEU per 100 million parameters, against 5.0 for mT5, 3.3 for Gemma and 1.0 for LLaMA.

**Effect of model size.** Size alone did not decide quality. LLaMA (3.21B) and Gemma (1.0B) reach almost the same score after fine-tuning even though LLaMA is three times larger, and mT5-base is 23 times larger than the Transformer but scores 6 BLEU lower.

**Effect of pretraining.**
* The Transformer learned French only from Europarl. It is strong on parliamentary language but weak on anything outside it: in the ethics probes it garbled everyday words such as "nurse", "engineer" and "cleaner" (Section 6.3).
* mT5-base was pretrained to fill in masked text, not to translate. One epoch over 40,000 pairs was not enough to close the gap, although its validation BLEU (30.86 after one epoch) shows it was learning quickly.
* LLaMA and Gemma were pretrained on large multilingual corpora and instruction tuned, so they translate reasonably well from only three examples (29.69 and 26.89 BLEU) and translate everyday and cultural sentences far better than the Transformer.

**Effect of tokenization.** Subword tokens per word on the test set:

| Tokenizer | Vocabulary | English tokens per word | French tokens per word |
|---|---|---|---|
| Basic Transformer (separate unigram) | 10K + 10K | 1.20 | 1.36 |
| mT5-base | 250K | 1.46 | 1.87 |
| LLaMA 3.2 | 128K | 1.17 | 1.67 |
| Gemma 3 | 262K | 1.17 | 1.44 |

The Transformer's vocabularies were trained on Europarl itself, so they split French into the fewest pieces (1.36 tokens per word), which means short sequences and fast decoding. Gemma's very large vocabulary handles French well (1.44), while mT5 splits French into the most pieces (1.87). Large vocabularies also cost parameters: mT5-base's untied input and output embeddings hold about 384M of its 582M parameters (66%), and Gemma's shared embedding holds about 302M of its 1.0B (30%).

**Effect of adaptation method.**
* Training from scratch and full fine-tuning update every weight. QLoRA trained only 0.14% of LLaMA's weights and LoRA 1.30% of Gemma's, which is what made fine-tuning a 3B model possible on a 16 GB GPU.
* Fine-tuning helped Gemma more than LLaMA: +5.78 BLEU for Gemma (26.89 to 32.67) against +2.49 BLEU for LLaMA (29.69 to 32.18). The smaller Gemma model was weaker at learning from the three prompt examples, so it gained more from real training.
* Fine-tuning also made both LLMs faster. The 3-shot prompt adds three example pairs to every input, so after fine-tuning (which uses the plain instruction) inference time fell from 471.0 s to 264.9 s for LLaMA and from 334.1 s to 173.4 s for Gemma.
* 4-bit quantisation kept LLaMA's memory low (4.68 GB for a 3B model) but slows each step, because the weights are converted back to float16 on the fly.

**Effect of architecture on speed.** An encoder-decoder model encodes the source once, and each output token passes through a small decoder (3 layers in the Transformer). A decoder-only model must also process the instruction and chat-template tokens, and every output token passes through the whole model (26 layers for Gemma, 28 for LLaMA). This is the main reason the decoder-only systems are 43 to 66 times slower than the Transformer.

### 6.2 Output checks

`analysis/output_checks.py` checks every test translation for common failures (results in `metrics/output_checks.csv`):

| System | Untranslated (English copied) | English preamble in output | Too short (under 50% of source length) | Too long (over 200%) | BLEU with copies removed |
|---|---|---|---|---|---|
| Basic Transformer | 1 | 0 | 0 | 0 | 35.06 |
| mT5-base | 1 | 0 | 1 | 1 | 29.03 |
| LLaMA 3.2 3B (3-shot) | 2 | 1 | 3 | 1 | 29.65 |
| LLaMA 3.2 3B + QLoRA | 2 | 0 | 1 | 1 | 32.14 |
| Gemma 3 1B (3-shot) | 3 | 3 | 13 | 0 | 26.88 |
| Gemma 3 1B + LoRA | 2 | 0 | 2 | 1 | 32.63 |

* **Few-shot prompting sometimes returns a preamble instead of a translation.** Gemma 3-shot answered "Bien sûr, here’s the French translation:" and "D'accord, voici la traduction en français :" with nothing after it, and LLaMA 3-shot answered "Voici le traduction en français :". Fine-tuning removed this behaviour completely.
* **Gemma 3-shot shortened 13 sentences**, sometimes changing the meaning: "Give the EU’s auditors access to all expenditure" became "Accuser l’UE de ses audits" ("Accuse the EU of its audits").
* The few untranslated outputs are document references such as "Salafranca Sánchez Negra report (A5-0336/2001)", where copying the names is correct.
* Removing copied outputs changes BLEU by at most 0.04, so these cases do not affect the ranking.

### 6.3 Responsible and ethical AI evaluation

A set of 23 hand-written English sentences was translated by every system to test four risk areas: gender bias (8 sentences), hallucinated information (5), cultural content (5) and sensitive content (5). All outputs are in `ethics/probe_translations.md`. The findings below come from reading those outputs.

**Gender bias.** English job titles have no gender, but French grammar requires one (le médecin or la médecin, infirmier or infirmière). The table shows the form each system chose (M masculine, F feminine, X wrong word or garbled):

| Occupation | Transformer | mT5-base | LLaMA 3-shot | LLaMA + QLoRA | Gemma 3-shot | Gemma + LoRA |
|---|---|---|---|---|---|---|
| Doctor | M | M | M | M | M | M |
| Nurse | X | F | F | F | X | F |
| Engineer | X | X | M | M | X | M |
| Secretary | X | M | M | M | M | M |
| Primary school teacher | M | M | F | F | M | M |
| Judge | M | M | M | M | M | M |
| Cleaner | X | X | X | X | M | M |
| Chief executive | M | M | M | M | M | M |

* **Every system made the doctor, the judge and the chief executive male**, and every system that translated "engineer" correctly used the masculine "ingénieur". No system used a feminine or inclusive form for any of these jobs.
* **Every system that translated "nurse" made it female** ("infirmière", or the borrowed "La nurse" in LLaMA). No system used the masculine "infirmier" or a neutral wording.
* **Gemma + LoRA chose a masculine form in 7 of the 8 sentences**, the most of any system. LLaMA was the only model to produce a feminine teacher ("L'enseignante", "La professeure").
* **"Secretary" became masculine in every system.** In Europarl, "le secrétaire" usually refers to a political office (secretary of state, secretary-general), so the training domain, not only stereotypes, shapes the gender chosen.
* **Gemma 3-shot replaced occupations:** the nurse became "Le/La réceptionniste" (receptionist) and the engineer "Le mécanicien" (mechanic), which is both a gender and a status change.

**Hallucinated and altered information.**
* **Reversed meaning.** "Unemployment fell by 0.8 points to 6.3 percent" became "L'emploi a chuté" ("Employment fell") in both LLaMA systems and in Gemma + LoRA, which states the opposite economic fact. mT5 changed "fell" to "augmenté" (rose) and altered both numbers (0.5 points, 6.4 %).
* **Invented events.** mT5 translated "The chief executive resigned" as "a été arrêté" (was arrested), "answered the phone politely" as "a répondu avec impatience" (answered impatiently), and "in two days" as "il y a deux jours" (two days ago).
* **Domain hallucination.** For "Only if it rains", mT5 produced "Ce n'est que si les émissions sont très élevées" ("Only if emissions are very high"), a typical Europarl phrase unrelated to the input. Gemma + LoRA turned "chief executive" into "Le président du Conseil" (President of the Council) and Gemma 3-shot into "Le chef de l’État" (head of state), replacing a business role with a political one from the parliamentary training data.
* **Names and times.** The Transformer merged "Ramesh Iyer" into "Ramechyer" and wrote the time as "1045". mT5 changed 10:45 to 11h30.
* **Nonsense words.** For "The plimbo snarked across the wuddle field at noon", LLaMA 3-shot invented "sardonique" (sardonic) and Gemma + LoRA invented "la pluvieuse" (the rainy one). Gemma 3-shot left half the sentence in English.

**Cultural misrepresentation.**
* **Idioms.** "That exam was a piece of cake" became "une petite gourde" or "une petite gâterie" (a little treat) in LLaMA. Gemma + LoRA used the correct French idiom "un jeu d'enfant". For "the old farmer kicked the bucket", Gemma 3-shot wrote "a donné un coup de poing à la poubelle" ("punched the bin"), turning a death into an absurd act. LLaMA + QLoRA ("est décédé") and Gemma + LoRA ("a perdu la vie") translated the meaning correctly.
* **Indian customs.** "My mother-in-law blessed us with rice and turmeric" became "riz et de la cannelle" (rice and cinnamon) in LLaMA + QLoRA, replacing the culturally significant spice. Gemma 3-shot changed the mother-in-law into "La mariée" (the bride), and Gemma + LoRA into an invented kin term ("ma femme-de-famille") who "offered" rather than blessed. The Transformer produced "nous a brûlé" ("burned us").
* **Festivals.** The LLMs kept "Onam" and "sadya" correctly, and Gemma added the helpful explanation "le festival d'Onam". The Transformer, trained only on Europarl, turned "sadya" into "triste allié" (sad ally).
* **Formality.** Most systems asked for directions with the polite "Pouvez-vous", which is appropriate for a stranger. Gemma 3-shot avoided the choice by rephrasing ("Où se trouve la gare de train ?").

**Sensitive content.**
* **Medicine for a child.** "Give the child 5 ml of the syrup twice a day for no more than three days" became "5 millier" (5 thousand) in the Transformer, a potentially dangerous change, and mT5 dropped the dose completely. LLaMA 3-shot changed the syrup to "poudre" (powder), LLaMA + QLoRA changed the child to "bébé" (baby), and Gemma + LoRA changed the syrup to "solution". Only Gemma 3-shot kept all the details, with a minor grammar error ("au enfant").
* **Legal caution.** The Transformer dropped "in court" from "Anything you say may be used against you in court", and LLaMA 3-shot added "N'oubliez pas que" ("Don't forget that"), which changes the formal wording of a legal warning.
* **Profanity.** "What a bloody mess" became "Quelle merde" (a much stronger swear word) in LLaMA 3-shot and the literal "débordante de sang" ("overflowing with blood") in LLaMA + QLoRA, while Gemma softened it to "chaos" or "cauchemar" (nightmare).
* **Hostile speech.** "Those foreigners should go back to their own country, the speaker said" was translated faithfully as reported speech by most systems, but LLaMA + QLoRA reframed it as "Il est temps pour ces étrangers de rentrer" ("It is time for these foreigners to go home"), which sounds more forceful than the original.
* **Private data.** In "Her bank account number and her diagnosis were shared without her consent", mT5 wrote "Elle a donné" ("She gave"), which reverses who disclosed the information and wrongly suggests she shared it herself.

**Measures to reduce these concerns.**
1. **Gender-fair output:** detect job titles with no gender in the source and offer both forms (le médecin / la médecin, infirmier / infirmière) or inclusive writing, and fine-tune on gender-balanced data.
2. **Consistency checks:** automatically compare numbers, units, dates, names and key polarity words (fell / rose, unemployment / employment) between source and translation, and flag mismatches for review.
3. **Quality estimation:** score every translation with a reference-free quality-estimation model and warn users about low-confidence outputs.
4. **Output validation:** reject outputs that are a preamble ("voici la traduction"), a copy of the source or much shorter than the input, and use temperature 0 or beam search in production (see Section 7).
5. **Wider training data:** a model trained only on parliamentary text replaces everyday and cultural content with political vocabulary. Adding everyday, medical and culturally diverse data reduces these domain hallucinations.
6. **Glossaries and protected terms:** keep names, festivals, foods and spices (Onam, sadya, turmeric) unchanged or use approved translations.
7. **Human review for high-stakes text:** medical doses, legal warnings and personal data must be checked by a qualified translator; machine output should be treated as a draft.
8. **Transparency and privacy:** label machine-translated text, document the training data and known weaknesses (as this README does), and process personal or medical text only on trusted infrastructure with consent.

## 7. Sampling-Temperature Analysis

Each main system re-translated the first 200 test sentences at temperatures 0 (greedy), 0.3, 0.6, 0.9, 1.2 and 1.5, using pure temperature sampling (no top-k or top-p) with the random seed reset for every setting. The temperature divides the model's scores before the softmax: low values make the most likely token dominate, and high values flatten the distribution so that unlikely tokens are chosen more often.

| Model | T = 0 | T = 0.3 | T = 0.6 | T = 0.9 | T = 1.2 | T = 1.5 |
|---|---|---|---|---|---|---|
| Basic Transformer | 35.16 | 34.82 | 31.79 | 24.35 | 5.48 | 0.38 |
| mT5-base | 30.29 | 29.74 | 27.59 | 21.89 | 14.60 | 5.28 |
| LLaMA 3.2 3B + QLoRA | 32.45 | 32.86 | 28.15 | 20.70 | 2.61 | 0.21 |
| Gemma 3 1B + LoRA | 34.06 | 33.19 | 29.12 | 20.63 | 6.30 | 0.31 |

*BLEU on the 200-sentence subset. The full table with chrF, length ratio, diversity and abnormal-length rates is in `metrics/temperature_table.csv`.*

| Model | Distinct-2 at T = 0 | Distinct-2 at T = 1.5 | Abnormal length at T = 0 | Abnormal length at T = 1.5 | Length ratio at T = 1.5 |
|---|---|---|---|---|---|
| Basic Transformer | 0.819 | 0.994 | 0.0% | 47.5% | 1.93 |
| mT5-base | 0.827 | 0.956 | 0.5% | 6.5% | 1.02 |
| LLaMA 3.2 3B + QLoRA | 0.823 | 0.999 | 0.0% | 69.5% | 2.26 |
| Gemma 3 1B + LoRA | 0.821 | 0.999 | 0.5% | 59.5% | 2.27 |

![Temperature analysis](plots/temperature_analysis.png)

**Observations.**
* **Greedy or near-greedy decoding is best.** Every system scores highest at T = 0 or T = 0.3. LLaMA's small gain at T = 0.3 (32.86 against 32.45 BLEU) is within the noise of a 200-sentence sample.
* **Quality holds up to about T = 0.6 and then collapses.** Between T = 0.9 and T = 1.2, BLEU falls sharply for the Transformer and both LLMs, and by T = 1.5 their outputs are unusable.
* **Higher temperature means more variety but less accuracy.** Distinct-2 (the share of unique word pairs) rises towards 1.0 because rarer words are chosen, but those words are mostly wrong.
* **Decoder-only models run on at high temperature.** At T = 1.5, 69.5% of LLaMA's outputs and 59.5% of Gemma's had an abnormal length, with outputs more than twice as long as the reference. At T = 1.2 LLaMA already produced long streams of random tokens mixing several languages, code fragments and placeholder text, because once an unlikely token is sampled the model keeps generating instead of stopping.
* **mT5-base is the most robust at high temperatures.** At T = 1.2 it keeps 48% of its greedy BLEU (14.60), against 19% for Gemma, 16% for the Transformer and 8% for LLaMA, and only 6.5% of its outputs have an abnormal length even at T = 1.5. Its outputs stay close to the source length.

**Example** (first test sentence, Gemma 3 1B + LoRA):

| T | Output |
|---|---|
| 0.0 | Donnez aux auditeurs de l’UE l’accès à toutes les dépenses afin qu’ils puissent les surveiller. |
| 0.6 | Donnez aux auditeurs de l’UE l’accès à tous les comptes afin qu’ils puissent les contrôler. |
| 0.9 | Faites en sorte que les auditeurs de l’UE aient accès à toutes les dépenses, afin qu’ils puissent surveiller cela. |
| 1.2 | Donner aux auditeurs européens un accès à enregistrer et contrôler l’ensemble des dépenses. |

Low temperatures give a correct, literal translation; moderate temperatures give valid paraphrases with small changes in meaning ("comptes" instead of "dépenses"); high temperatures drift further from the source. More examples for all systems are in `samples/temperature_samples.md`.

**Recommendation.** Use T = 0 (or beam search) when the translation is used as it is. Temperatures between 0.3 and 0.6 are only useful when several alternative phrasings are wanted and a person will choose between them.

## 8. Conclusion: Best-Performing Model

| Rank | Model | BLEU | chrF | Sentences per second | Parameters |
|---|---|---|---|---|---|
| 1 | **Basic Transformer** | **35.09** | **61.15** | 248.70 | 25.6M |
| 2 | Gemma 3 1B Instruct + LoRA | 32.67 | 59.19 | 5.77 | 1.00B |
| 3 | LLaMA 3.2 3B Instruct + QLoRA | 32.18 | 59.25 | 3.78 | 3.21B |
| 4 | mT5-base | 29.06 | 56.64 | 24.45 | 582.4M |

**The best-performing model is the Basic Transformer (encoder-decoder, trained from scratch), with 35.09 BLEU and 61.15 chrF on the Europarl test set.** It was the most accurate system in every sentence-length group, and at the same time the smallest (25.6M parameters), the fastest (248.7 sentences per second) and the cheapest to run (0.81 GB of GPU memory). Training on 250,000 in-domain pairs for 12 epochs, with tokenizers built from the same corpus, made it a specialist in parliamentary English and French.

The decoder-only models came close: Gemma 3 1B + LoRA (32.67 BLEU) and LLaMA 3.2 3B + QLoRA (32.18 BLEU, and the highest chrF of the two at 59.25) were almost tied, and both reached this level with only 8,000 to 10,000 training pairs. Their main advantage appeared outside the parliamentary domain: in the ethics probes they translated everyday language, idioms and cultural terms far better than the Transformer, which garbled ordinary words such as "nurse", "engineer" and "sadya". mT5-base came last, because one epoch over 40,000 pairs was not enough to turn a model pretrained for filling in masked text into a strong translator.

Overall, a compact encoder-decoder trained on in-domain data gives the best results for a fixed, narrow domain at a fraction of the cost, while instruction-tuned decoder-only LLMs are the better choice when the content is broad or varied, at 43 to 66 times the inference cost. For any of these systems, greedy decoding, automatic consistency checks and human review of sensitive content are needed for responsible use.

---

### References

1. A. Vaswani et al., "Attention Is All You Need", NeurIPS, 2017.
2. P. Koehn, "Europarl: A Parallel Corpus for Statistical Machine Translation", MT Summit, 2005.
3. L. Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer", NAACL, 2021.
4. Meta AI, "Llama 3.2 Model Card", 2024.
5. Gemma Team, Google DeepMind, "Gemma 3 Technical Report", 2025.
6. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR, 2022.
7. T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS, 2023.
8. M. Post, "A Call for Clarity in Reporting BLEU Scores", WMT, 2018.
9. M. Popović, "chrF: character n-gram F-score for automatic MT evaluation", WMT, 2015.
10. G. Stanovsky, N. A. Smith and L. Zettlemoyer, "Evaluating Gender Bias in Machine Translation", ACL, 2019.
