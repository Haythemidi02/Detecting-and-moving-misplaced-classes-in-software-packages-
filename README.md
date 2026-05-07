# ClassMoveExplorer

ClassMoveExplorer is a sophisticated Java class placement analyzer that identifies misplaced classes in Java software packages and suggests optimal relocations based on structural, semantic, and architectural patterns.

## 🚀 Key Features

- **AST-Based Analysis**: Precise Java source code parsing using `javalang` to extract method signatures, field types, and annotations.
- **Structural Dependency Graph**: Deep analysis of class relationships including inheritance, field types, and method parameters.
- **Cohesion & Coupling Metrics**: Calculates Package Cohesion, Afferent/Efferent coupling, and LCOM to identify architectural violations.
- **Semantic reasoning**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) to compare class intent with package definitions.
- **LLM Reasoning Engine**: Leverages `distilgpt2` to generate human-readable reasoning for every relocation suggestion.
- **Evaluation Framework**: Built-in ground-truth evaluator to measure Precision, Recall, and F1-score via intentional project "corruption" tests.

## 📁 Project Structure

```text
root/
├── src/
│   └── class_move_explorer/        # Core library package
│       ├── analyzers/              # Java, Dependency, Embedding, and LLM analyzers
│       ├── core/                   # Orchestration (MoveClassAssistant)
│       ├── utils/                  # Metrics and formatting
│       └── evaluation/             # Ground-truth testing framework
├── scripts/                        # Executable CLI and verification scripts
├── examples/                       # Usage demonstrations
└── test_project/                   # Synthetic test data
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Haythemidi02/Detecting-and-moving-misplaced-classes-in-software-packages-
cd Detecting-and-moving-misplaced-classes-in-software-packages-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Run Analysis via CLI
The fastest way to analyze a project is using the provided CLI script:

```bash
python scripts/run_cli.py /path/to/your/java/project --output results.csv --verbose
```

### 1b. Run the Streamlit UI
To use a modern web UI (Dashboard) that lets you select a project, run analysis, download the CSV, and view results:

```bash
python -m streamlit run streamlit_app.py
```

- **Inputs**:
  - **Local folder selection** (via a folder picker when running locally)
  - **ZIP upload** (`.zip`) containing your Java project
- **Outputs**:
  - **Downloadable CSV** (`class_placement_analysis.csv`)
  - **Dashboard** with KPIs, confidence distribution, top recommendations, full results table, and optional evaluation

#### Evaluation (what it means in this repo)
This project does **not** train a custom model on a fixed dataset. Instead, it combines:
- Heuristics/rules + dependency signals
- A **pretrained** sentence-transformer (semantic embeddings)
- A lightweight **pretrained** LLM for explanation text

Because of that, evaluation can be done in two ways:
- **Synthetic benchmark (recommended)**: repeatable, input-independent score for the current algorithm.
- **Project corruption test**: input-dependent score (varies by the project you upload/select).

##### 1) Synthetic benchmark (repeatable, recommended)
The benchmark runs a loop over \(N\) generated “toy projects”:

1. **Generate** a small, well-placed Java-like project *in memory* (no file parsing) with packages such as:
   - `com.example.controller`, `com.example.service`, `com.example.model`, `com.example.repository`, `com.example.util`, `com.example.config`, `com.example.exception`
2. **Corrupt** it by intentionally changing the package of a subset of classes:
   - \(k = \max(1, \lfloor \text{total\_classes} \cdot \text{misplace\_ratio} \rfloor)\)
   - This produces a **ground truth** set \(GT\) = “classes we intentionally misplaced”.
3. Run the normal pipeline (dependency analysis / embeddings if enabled) to **detect** misplaced classes \(DET\) and **suggest** target packages.
4. Compute metrics:
   - \(TP = |GT \cap DET|\)
   - \(FP = |DET \setminus GT|\)
   - \(FN = |GT \setminus DET|\)
   - Precision \(= TP/(TP+FP)\), Recall \(= TP/(TP+FN)\), F1 is the harmonic mean.
5. Aggregate (mean/std) metrics across the \(N\) projects.

In the Streamlit app: **Evaluation → Synthetic benchmark (recommended)**.

##### 2) Project corruption test (depends on your input)
This is a “self-check” on the project you uploaded/selected:

1. Parse the selected project to collect class metadata.
2. Randomly pick a subset of classes to misplace and record their original packages (ground truth).
3. Run the pipeline on the corrupted metadata and score Precision/Recall/F1 as above.

In the Streamlit app: enable **Evaluation** before clicking **Run**, then open **Evaluation → Project corruption**.

> Note: If you upload/select a very small project (e.g., only 1–2 `.java` files), project-based metrics like Precision/Recall may show 0% and won’t be representative.

### 2. Run Formal Evaluation
To evaluate the tool's accuracy against a test project:

```bash
python scripts/verify_rebuild.py
```

### 3. Programmatic Usage
```python
import sys, os
sys.path.append(os.path.abspath("src"))
from class_move_explorer.core.assistant import MoveClassAssistant

assistant = MoveClassAssistant()
results = assistant.analyze_and_recommend(
    project_path="/path/to/project",
    output_csv="analysis.csv"
)
```

## 📊 Requirements

- Python 3.8+
- Java (for project analysis)
- Dependencies: `javalang`, `transformers`, `sentence-transformers`, `torch`, `pandas`, `scikit-learn`

## 📝 License
MIT License
