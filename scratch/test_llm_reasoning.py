import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_analyzer import LLMAnalyzer

def test_llm():
    analyzer = LLMAnalyzer()
    
    # Mock class info
    class_info = {
        'class_name': 'UserController',
        'annotations': ['RestController', 'RequestMapping'],
        'package': 'com.example.model'
    }
    
    print("\n" + "="*50)
    print(" TESTING LLM REASONING GENERATION ".center(50, "="))
    print("="*50)
    
    reasoning = analyzer._generate_reasoning(class_info, 'controller', 0.95, {})
    print(f"Input Class: {class_info['class_name']}")
    print(f"Target Package Type: controller")
    print(f"Generated Reasoning:\n{reasoning}")
    print("="*50)

if __name__ == "__main__":
    test_llm()
