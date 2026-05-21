# Beyond Built-in Multiclass: Binary VQC Decomposition with Two-Layer Decision Making on Iris — a Qiskit Supervised Learning Experiment

A quantum machine learning project that extends Qiskit's Variational Quantum Classifier (VQC) to improve multiclass accuracy on the Iris dataset through a two-layer decision architecture: One-vs-Rest (OvR) binary classifiers as the first layer, with One-vs-One (OvO) classifiers resolving ambiguous predictions as the second layer.

Classical SVC baselines are included throughout for comparison.

> Designed to run on **Google Colab**. Falls back to a local path automatically when run outside Colab.

---

## What is a VQC?

A Variational Quantum Classifier is a hybrid quantum-classical algorithm. A parameterised quantum circuit encodes input features into a quantum state, measures the output, and a classical optimiser adjusts the circuit parameters to minimise a loss function — similar in spirit to a small neural network, but running on a quantum (or simulated quantum) backend.

Qiskit's built-in `VQC` supports multiclass classification natively via a one-vs-rest strategy internally. However, its output probabilities across classes are often close together, leading to uncertain or incorrect decisions — especially on the harder Iris class boundaries (versicolor vs virginica). This project addresses that directly.

---

## The core idea — two-layer decision making

Rather than relying on Qiskit's built-in multiclass handling, this project trains the classifiers explicitly and controls the decision process.

### Workflow 1 — training pipeline

1. Load the full Iris dataset and scale features to `[0, π]`
2. Perform a single train/test split on the original multiclass data
3. For each class, create a balanced binary training set:
   - Positive samples: all examples of that class
   - Negative samples: equal numbers drawn from each other class (undersampling)
   - The test set is **never resampled** — only training data is balanced
4. Train one binary VQC per class (3 total)
5. Persist every model and its training metrics to TinyDB via `dill` + `base64` serialisation

### Workflow 2 — two-layer decoding

At evaluation time, each binary VQC votes on a test sample:

- **Clean prediction** `[1, 0, 0]` → one winner → final class directly
- **Tie** `[1, 1, 0]` or no winner `[0, 0, 0]` → escalate to Layer 2

Layer 2 uses One-vs-One classifiers trained on pairwise class combinations (0v1, 0v2, 1v2). When a tie occurs between classes A and B, the (A, B) OvO classifier makes the final call. This replaces random tie-breaking with a structurally informed second decision, which is where the accuracy improvement comes from.

---

## Architecture

### Workflow 1 — training pipeline

```
 Iris dataset ──► balanced OvR split ──► VQC training (×3) ──► TinyDB
                  (train only,            one binary VQC        persist models
                   test kept clean)       per class             + metrics
```

### Workflow 2 — two-layer decoding

```
                         Test sample
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   Classifier 0        Classifier 1        Classifier 2
   class 0 vs rest     class 1 vs rest     class 2 vs rest
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    Prediction matrix
                e.g. [1, 0, 0]  [1, 1, 0]  [0, 0, 0]
                       │              │           │
                    1 winner       2 winners   0 winners
                       │              └─────┬─────┘
                       ▼                    ▼
                 Clean prediction      Ambiguous → Layer 2
                 winner = class        escalate to OvO
                       │
                       │          ┌──────────────────────────────┐
                       │          │  OvO classifiers             │
                       │          │  0v1  ·  0v2  ·  1v2        │
                       │          │  only relevant pair called   │
                       │          └──────────────┬───────────────┘
                       │                         │
                       └─────────────────────────┘
                                       ▼
                               Final prediction
                               class 0 / 1 / 2
```

### Example — longest path: `[1, 1, 0]`

```
  Classifier 0 → 1  ─┐
  Classifier 1 → 1  ─┼──► [1, 1, 0]  two winners → tie between class 0 and class 1
  Classifier 2 → 0  ─┘
                              │
                              ▼
                     escalate to Layer 2
                              │
                    ┌─────────┴──────────────────────┐
                    │         │                      │
                  0v1 ✓     0v2 ✗                 1v2 ✗
               (called)  (not called)          (not called)
                    │
                    ▼
             OvO 0v1 classifier decides
                    │
                    ▼
           Final class → 0 or 1
```

Only the `0v1` classifier is invoked — the pair that matches the two tied classes.
`0v2` and `1v2` are never called for this sample.

---

## Experiments

The notebook runs five experiments in order, each building on the previous:

| # | Experiment | Description |
|---|---|---|
| 1 | Single multiclass VQC | Baseline — Qiskit's built-in multiclass VQC on all 3 classes |
| 2 | OvR VQC (unbalanced) | 3 binary VQCs, naive train/test split, random tie-breaking |
| 3 | OvR VQC (balanced) | 3 binary VQCs, balanced training data, shared held-out test set |
| 4 | SVC baselines | Classical SVC — multiclass, OvR naive, OvR + OvO tiebreaker |
| 5 | VQC OvR + OvO | Full two-layer quantum architecture |

---

## Requirements

- Google Colab (recommended) or Python 3.10+ locally
- A Google Drive folder at `MyDrive/TinyDb/` (created automatically on first run)

All Python packages are installed in the first notebook cell:

```bash
pip install qiskit pylatexenc qiskit_algorithms qiskit_machine_learning tinydb dill
```

### Key packages

| Package | Purpose |
|---|---|
| `qiskit` | Quantum circuit construction |
| `qiskit_machine_learning` | VQC implementation, optimisers, samplers |
| `qiskit_algorithms` | COBYLA, SLSQP classical optimisers |
| `scikit-learn` | Iris dataset, SVC baseline, train/test split, metrics |
| `tinydb` | Lightweight JSON-based experiment database |
| `dill` + `base64` | Model serialisation and persistence |

---

## Project structure

```
QML_IRIS_EXTENSION.ipynb
│
├── 1. Installation
├── 2. Imports
├── 3. Google Drive & DB setup         ← Colab + local fallback
├── 4. Data classes
│     ModelConfig                      ← all hyperparameters
│     ModelResult                      ← training metrics & curves
│     ModelRecord                      ← config + result + MD5 hash id
│
├── 5. Database helpers
│     save_record()                    ← insert, skip duplicates by hash
│     update_record()                  ← merge result into existing record
│     load_record()                    ← retrieve by id
│     query_records()                  ← filter with a lambda predicate
│
├── 6. Component registry & builders
│     load_dataset()                   ← scale + train/test split
│     build_components()               ← feature map, ansatz, optimizer, sampler
│
├── 7. Training
│     train_and_save()                 ← build VQC, fit, persist to TinyDB
│
├── 8. Binary dataset helpers
│     load_dataset_binary()            ← naive unbalanced OvR split
│     load_data_set_binary_balanced()  ← undersample negatives per class
│     load_dataset_binary_balanced_with_test() ← balanced train, clean test
│
├── 9. Decode helpers
│     decode_ovr_naive()               ← majority vote + random tie-breaking
│     decode_ovr_with_ovo_tiebreaker() ← two-layer decoder with stats printout
│     train_ovo_tiebreakers()          ← train pairwise OvO classifiers
│
├── Experiment 1 — single multiclass VQC
├── Experiment 2 — OvR VQC (unbalanced)
├── Experiment 3 — OvR VQC (balanced)
├── Experiment 4 — SVC baselines
└── Experiment 5 — VQC OvR + OvO (full two-layer)
```

---

## Experiment tracking

Every trained model is automatically saved to a TinyDB JSON file with:

- A **deterministic MD5 hash** of the full `ModelConfig` — duplicate experiments are never saved twice
- Training loss curve, per-iteration accuracy (train + test)
- Final train/test scores and elapsed time
- The full serialised model object (reloadable with `dill`)

To reload a previously trained model:

```python
loaded_record = load_record("your_record_id_here")
model_bytes   = base64.b64decode(loaded_record.result.model_dill.encode("utf-8"))
model         = dill.loads(model_bytes)
print(model.score(test_features, test_labels))
```

To query across experiments:

```python
# all records with test accuracy above 0.85
results = query_records(lambda r: r.result.test_score is not None and r.result.test_score > 0.85)

# all binary models using COBYLA with 150 iterations
results = query_records(lambda r: r.config.binary == True and r.config.optimizer_type == "cobyla" and r.config.optimizer_maxiter >= 150)
```

---

## Notes

- VQC training is **slow** — each binary classifier can take several minutes on Colab's CPU runtime. Use GPU runtime or reduce `optimizer_maxiter` for faster iteration.
- The OvO tiebreaker layer uses SVC by default in Experiment 5. Uncomment the QVC tiebreaker line to run a fully quantum two-layer architecture (significantly slower).
- The `algorithm_globals.random_seed` and `sampler_seed` in `ModelConfig` control quantum randomness for reproducibility.

---

## License

See [LICENSE](LICENSE) for details.
