import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from class_move_explorer.core.assistant import MoveClassAssistant
from class_move_explorer.evaluation.evaluator import Evaluator

def run_test():
    # Adjusted path for test_project relative to root
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_project'))
    
    assistant = MoveClassAssistant()
    evaluator = Evaluator(assistant)
    
    print("\n" + "="*50)
    print(" RUNNING EVALUATION ON TEST PROJECT ".center(50, "="))
    print("="*50)
    
    # We misplace 25% of classes
    results = evaluator.run_evaluation(project_path, misplace_ratio=0.25)
    
    # Display results
    assistant.metrics.display_evaluation_summary(results)

if __name__ == "__main__":
    run_test()
