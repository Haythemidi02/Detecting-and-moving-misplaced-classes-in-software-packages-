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
