# 🧭 ClassMoveExplorer

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java Analysis](https://img.shields.io/badge/Analysis-Java-orange)](https://www.java.com/)

**ClassMoveExplorer** is a research-driven tool designed to automate the detection of misplaced classes in Java software architectures. It combines static analysis, semantic embeddings, and Large Language Models (LLMs) to identify architectural violations and suggest optimal package relocations with human-readable reasoning.

---

## 🌟 Key Features

-   **🔍 Multi-Layered Analysis**:
    -   **Structural**: AST-based parsing (via `javalang`) to extract dependencies, inheritance, and field types.
    -   **Semantic**: Intent-based comparison using Sentence Transformers (`all-MiniLM-L6-v2`) to match class purpose with package context.
    -   **Reasoning**: LLM-powered explanation engine (supporting Phi-2, Qwen, TinyLlama) to justify every relocation.
-   **📈 Intelligent Metrics**: Calculates Package Cohesion, Afferent/Efferent coupling, and LCOM to identify architectural bottlenecks.
-   **🖥️ Modern Dashboard**: A Streamlit-based UI for project visualization, KPI tracking, and interactive analysis results.
-   **🧪 Robust Evaluation**:
    -   **Synthetic Benchmark**: Automated testing on $N$ generated "toy projects" to measure baseline performance.
    -   **Project Corruption Test**: Self-check mechanism that intentionally misplaces classes in your own project to verify detection accuracy.

---

## 🏗️ Architecture: How it Works

ClassMoveExplorer operates on a "Three Pillars" methodology:

1.  **Structural Mapping**: We build a directed dependency graph of the entire project. Classes with high external coupling and low internal cohesion are flagged.
2.  **Semantic Alignment**: Using NLP, we compare the class's docstrings, method names, and fields against the "semantic signature" of its current and potential packages.
3.  **LLM Synthesis**: An LLM reviews the findings from the first two pillars to provide a final recommendation and a plain-English explanation for developers.

---

## 📁 Project Structure

```text
root/
├── src/
│   └── class_move_explorer/        # Core library
│       ├── analyzers/              # Java, Dependency, Embedding, and LLM analyzers
│       ├── core/                   # Orchestration (MoveClassAssistant)
│       ├── evaluation/             # Synthetic & Project evaluation logic
│       └── utils/                  # Metrics and formatting utilities
├── streamlit_app.py                # Main Dashboard application
├── scripts/                        # CLI tools and verification scripts
├── examples/                       # Python usage demonstrations
├── test_project/                   # Synthetic Java test data
└── requirements.txt                # Dependency list
```

---

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.8+**
- **Java Development Kit (JDK)** (Required for parsing `.java` files)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/Haythemidi02/Detecting-and-moving-misplaced-classes-in-software-packages-
cd Detecting-and-moving-misplaced-classes-in-software-packages-

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory to enable HuggingFace LLM support:
```env
HF_API_TOKEN=your_huggingface_token_here
```

---

## 📖 Usage

### 🚀 Streamlit Dashboard (Recommended)
The most user-friendly way to interact with the tool:
```bash
streamlit run streamlit_app.py
```
- **Upload**: Drop a `.zip` of your Java project.
- **Analyze**: Visualize KPIs, confidence distributions, and top recommendations.
- **Export**: Download results as a structured `.csv`.

### 💻 Command Line Interface (CLI)
For batch processing or integration into CI/CD:
```bash
python scripts/run_cli.py /path/to/java/project --output results.csv --verbose
```

### 🐍 Programmatic Usage
```python
from class_move_explorer.core.assistant import MoveClassAssistant

# Initialize the assistant
assistant = MoveClassAssistant(use_huggingface=True)

# Run analysis
results = assistant.analyze_and_recommend(
    project_path="./my_java_project",
    output_csv="analysis_results.csv"
)

# Access calculated metrics
print(assistant.metrics.metrics_calculated)
```

---

## 🧪 Evaluation Methodology

### 1. Synthetic Benchmark
Repeatable, input-independent score. It generates $N$ small projects in memory, corrupts them, and measures detection accuracy.
*   **Metric Goal**: Evaluate the raw algorithmic performance.

### 2. Project Corruption Test
Input-dependent. It takes the project you uploaded, randomly misplaces a subset of classes, and tests if the tool can find them.
*   **Metric Goal**: Evaluate how well the tool adapts to your specific coding style.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with ❤️ for Architectural Excellence in Software Engineering.*
