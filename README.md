# Detecting-and-moving-misplaced-classes-in-software-packages-
#Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```
Main dependencies:
- `pandas` - Data processing
- `sentence-transformers` - Semantic embeddings  
- `transformers` - Language models
- `scikit-learn` - Machine learning utilities
- `torch` - Deep learning framework
#Basic Usage
```in usage_example.py file
from move_class_assistant import MoveClassAssistant
# Initialize the tool
assistant = MoveClassAssistant()
# Analyze a Java project
results = assistant.analyze_and_recommend(
    project_path="/path/to/your/java/project",
    output_csv="class_placement_analysis.csv"
)
# View results
print(f"Found {len(results[results['is_misplaced'] == True])} misplaced classes")
```
# Command Line Usage
```bash
python usage_example.py
```
