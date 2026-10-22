# Prompting Small LLMs for Bangla Text Classification

This project tests how well small, locally run LLMs can label Bangla text using three prompting styles:

- **Zero-shot**: instructions only
- **Few-shot**: instructions plus a few labeled examples
- **Chain-of-thought (CoT)**: "think step by step" instructions

The models run through [Ollama](https://ollama.com). Each row of a dataset is sent to the model, the predicted label is saved to a CSV, and the predictions are scored against the human labels (accuracy, weighted precision/recall/F1, confusion matrix and classification report).

The four datasets are split into two parts, and each part is run in its own way:

| | Part 1: Python scripts | Part 2: Jupyter notebook |
|---|---|---|
| Datasets | RevBangla, BanFakeNews | SentNoB, BanglaFoodReview |
| Code | `Codes/*.py` (one script per experiment) | `run_prompt.ipynb` (one configurable notebook) |
| Prompts | Built into each script, with copies in `Dataset prompting/RevBangla` and `Dataset prompting/Banfake` | Pasted in from `Dataset prompting/SentNoB` and `Dataset prompting/BanglaFoodReview` |
| Models | Llama 3.2 1B, Llama 3.2 3B, Gemma 2 2B | Any Ollama model (default `gemma2:2b`) |
| Run from | `Codes/` | Repository root |

The full write-up is in [NLP_Project.pdf](NLP_Project.pdf).

## Repository layout

```
.
├── Codes/                     # Part 1: one script per (dataset, prompting style, model)
├── Dataset prompting/         # Prompts (system messages) for each dataset/style
│   ├── RevBangla/  Banfake/   #   Part 1
│   └── SentNoB/  BanglaFoodReview/   #   Part 2
├── Datasets/                  # The four Bangla datasets
├── run_prompt.ipynb           # Part 2 notebook (Codes/run_prompt.ipynb is an identical copy)
└── NLP_Project.pdf            # Project report
```

## Datasets

| File | Part | Task | Text column | Label column | Labels | Rows |
|---|---|---|---|---|---|---|
| `Datasets/RevBangla.csv` | 1 | Product review sentiment | `comment` | `sentiment` | `positive`, `negative` | 5,015 |
| `Datasets/BanFakeNews.csv` | 1 | Fake news detection | `headline` / `Data` | `Label` | `0` = fake, `1` = authentic | 1,299 (all labeled `0`) |
| `Datasets/SentNoB.csv` | 2 | Sentiment | `Data` | `Label` | `0` neg, `1` neutral, `2` pos | 1,584 |
| `Datasets/BanglaFoodReview.csv` | 2 | Food review sentiment | `Data` | `Label` | `0` neg, `1` pos | 1,038 |

## Common setup (both parts)

### 1. Python dependencies

Python 3.10 or newer is recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install ollama pandas scikit-learn jupyter
```

### 2. Ollama and the base models

Install Ollama from https://ollama.com/download and make sure it is running (`ollama serve`, or open the Ollama app). Then pull the base models:

```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull gemma2:2b
```

Part 2 only needs `gemma2:2b`, or whichever model you choose in the notebook.

---

## Part 1: RevBangla and BanFakeNews (Python scripts)

Each script in `Codes/` runs one experiment from start to finish: it labels the whole dataset, then evaluates the labels. Script names follow the pattern `<dataset>_<prompting>_<model>.py`.

Every script starts with a header comment that lists its dataset, prompting style, model, input file and output files. Read the top of a script to see what it does.

### Step 1: Create the custom model names

The scripts call custom Ollama models: the base models with `temperature 0.25`, to keep the output close to deterministic.

| Name used in code | Base model |
|---|---|
| `1bmyllama3.2` | `llama3.2:1b` |
| `tmp3myllama3.2` | `llama3.2:3b` |
| `mygemma2` | `gemma2:2b` |

```bash
printf 'FROM llama3.2:1b\nPARAMETER temperature 0.25\n' > Modelfile.1b
printf 'FROM llama3.2:3b\nPARAMETER temperature 0.25\n' > Modelfile.3b
printf 'FROM gemma2:2b\nPARAMETER temperature 0.25\n'   > Modelfile.gemma

ollama create 1bmyllama3.2   -f Modelfile.1b
ollama create tmp3myllama3.2 -f Modelfile.3b
ollama create mygemma2       -f Modelfile.gemma

ollama list   # check that all three are listed
```

### Step 2: Run the experiments

The scripts read the datasets directly from `../Datasets/`, so no data preparation is needed. Run them from inside `Codes/`:

```bash
cd Codes
python Revbangla_zeroshot_1B.py   # for example
```

**RevBangla (sentiment):** reads `Datasets/RevBangla.csv` and classifies the `comment` column.

| Prompting | Llama 3.2 1B | Llama 3.2 3B | Gemma 2 2B |
|---|---|---|---|
| Zero-shot | `python Revbangla_zeroshot_1B.py` | `python Revbangla_zeroshot_3B.py` | – |
| Few-shot | `python Revbangla_fewshot_1B.py` | `python Revbangla_fewshot_3B.py` | `python Revbangla_fewshot_gemma2b.py` |
| CoT | `python Revbangla_COT_1B.py` | `python Revbangla_COT_3B.py` | `python Revbangla_COT_gemma2b.py` |

**BanFakeNews (fake news detection):** reads `Datasets/BanFakeNews.csv` and classifies the `headline`. The exception is `banfakenews_COT_3B.py`, which classifies the article body (`Data`, first 1,000 characters).

| Prompting | Llama 3.2 1B | Llama 3.2 3B | Gemma 2 2B |
|---|---|---|---|
| Zero-shot | `python banfake_zeroshot_1B.py` | `python banfake_zeroshot_old.py` | `python banfake_zeroshot_gemma2B.py` |
| Few-shot | `python banfake_fewshot_1B.py` | `python banfake_fewshot.py` | – |
| CoT | `python banfakenews_COT_1B.py` | `python banfakenews_COT_3B.py` | – |

### Outputs

Each script writes two files into `Codes/`:

- `annotated_*.csv`: the model's label for every row. Model errors are saved as `-1`. Replies that aren't a valid label are saved as `-1` (BanFakeNews), or as `negative` or `unknown` (RevBangla).
- `metrics_analysis_*.txt`: accuracy, precision, recall, F1, the confusion matrix and the classification report. These are also printed to the console.

The output file names for each script are listed in its header comment.

> **Tip:** to try a quick run first, uncomment the `df = df.head(N)` and `human_df = human_df.head(N)` lines in a script.

### Notes

- **`BanFakeNews.csv` contains only fake (`0`) articles.** Accuracy here measures how many fake articles the model flags as fake, and precision/recall for class `1` are not meaningful.
- **RevBangla scoring differs between models.** The Llama 1B/3B scripts compute accuracy over valid predictions only, while the Gemma scripts score every row.

---

## Part 2: SentNoB and BanglaFoodReview (Jupyter notebook)

`run_prompt.ipynb` is a single notebook. You configure the dataset, model and prompt in a few cells, and then run all cells. The notebook has its own instructions in markdown cells, and every cell you need to change is marked **✏️ EDIT**. Run it from the **repository root** so that the `Datasets/...` paths resolve:

```bash
jupyter notebook run_prompt.ipynb
```

### Step 1: Configure the run

Edit these cells:

1. **Config cell**: these values only affect the output file names.
   ```python
   dataset_name = "SentNoB"        # or "BanglaFoodReview"
   model_name   = "gemma_2B"
   prompting    = "zero_shot"      # "few_shot" or "chain_of_thought"
   ```
2. **Dataset path**
   ```python
   dataset_path = "Datasets/SentNoB.csv"   # or "Datasets/BanglaFoodReview.csv"
   ```
3. **`messages` cell**: replace the whole cell with the matching prompt file. Each file is already a ready-to-paste `messages = [...]` block.

   | Prompting | SentNoB | BanglaFoodReview |
   |---|---|---|
   | Zero-shot | `Dataset prompting/SentNoB/zero_shot_prompting.txt` | `Dataset prompting/BanglaFoodReview/zero_shot.txt` |
   | Few-shot | `Dataset prompting/SentNoB/few_shot_prompting.txt` | `Dataset prompting/BanglaFoodReview/few_shot.txt` |
   | CoT | `Dataset prompting/SentNoB/chain_of_thought_prompting.txt` | `Dataset prompting/BanglaFoodReview/chain_of_thought.txt` |

4. **Model**: in the `main` function, set `ollama.chat(model='gemma2:2b', ...)` to any model shown by `ollama list` (for example `llama3.2:1b` or `llama3.2:3b`), and update `model_name` to match.
5. **Text column**: leave `main(column_to_annotate="Data")` as it is. Both datasets use `Data` for the text and `Label` for the label.

### Step 2: Run all cells

**Kernel → Restart & Run All.** Restarting matters: the `messages` list grows on every run, so a restart clears out the previous conversation.

### Outputs

These are written to the repository root:

- `annotated_results_<dataset>_<model>_<prompting>.csv`: the model's label for every row
- `metrics_analysis_<dataset>_<model>_<prompting>.txt`: accuracy, precision, recall, F1, the confusion matrix and the classification report

### Part 2 notes

- The notebook only strips a leading `Label: ` from the model's reply. If the model answers with something like `Sentiment: 1` or adds an explanation, the evaluation cell can fail or give poor scores. Check the annotated CSV, and clean up the labels if needed, before relying on the metrics.
- The notebook as committed is configured for a BanFakeNews few-shot test run. Change the config, dataset path and prompt cells as described above before running SentNoB or BanglaFoodReview.

---

## Runtime

Each row is a separate model call, and both approaches send up to 1,500 previous messages as context. Full runs can take a long time on a laptop. Test on a small subset first.
