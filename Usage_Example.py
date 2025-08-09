"""
Example usage of the MoveClassAssistant tool
"""
from move_class_assistant import MoveClassAssistant
import os

def main():
    # Initialize the assistant
    assistant = MoveClassAssistant()
    
    # Example 1: Analyze a Java project
    project_path = r"C:\Users\Haythem\Downloads\MC_Assistant_2\spring-petclinic"  # Replace with actual path
    
    if os.path.exists(project_path):
        print("Analyzing Java project...")
        results_df = assistant.analyze_and_recommend(
            project_path=project_path,
            output_csv="misplaced_classes_analysis.csv"
        )
        
        # Display summary
        print(f"\nAnalysis complete! Found {len(results_df[results_df['is_misplaced'] == True])} potentially misplaced classes.")
        
        # Show some examples
        misplaced = results_df[results_df['is_misplaced'] == True]
        if len(misplaced) > 0:
            print("\nTop 5 misplaced classes:")
            for _, row in misplaced.head().iterrows():
                print(f"  {row['class_name']} ({row['current_package']} → {row['suggested_package']})")
                print(f"    Confidence: {row['confidence']:.3f}")
                print(f"    Reason: {row['reasoning'][:100]}...")
                print()
    
    else:
        # Example with sample data for demonstration
        print("Project path not found. Running demonstration with sample data...")
        demo_analysis()

def demo_analysis():
    """Demonstrate the tool with sample Java class data"""
    # This would normally come from actual Java files
    sample_classes = [
        {
            'class_name': 'UserController',
            'package': 'com.example.model',  # Misplaced - should be in controller
            'file_path': '/demo/UserController.java',
            'relative_path': 'src/main/java/com/example/model/UserController.java',
            'imports': ['org.springframework.web.bind.annotation.RestController'],
            'methods': ['getUser', 'createUser', 'updateUser', 'deleteUser'],
            'fields': ['userService'],
            'annotations': ['RestController', 'RequestMapping'],
            'extends': '',
            'implements': [],
            'class_type': 'class',
            'content_preview': 'public class UserController { @Autowired UserService userService; }'
        },
        {
            'class_name': 'ReportGenerator',
            'package': 'com.example.model',  # Misplaced - should be in service/util
            'file_path': '/demo/ReportGenerator.java',
            'relative_path': 'src/main/java/com/example/model/ReportGenerator.java',
            'imports': ['java.util.List', 'com.example.model.Report'],
            'methods': ['generateReport', 'exportToCSV', 'exportToPDF', 'formatData'],
            'fields': ['dateFormatter', 'csvWriter'],
            'annotations': ['Component'],
            'extends': '',
            'implements': [],
            'class_type': 'class',
            'content_preview': 'public class ReportGenerator { public Report generateReport() { } }'
        },
        {
            'class_name': 'User',
            'package': 'com.example.model',  # Correctly placed
            'file_path': '/demo/User.java',
            'relative_path': 'src/main/java/com/example/model/User.java',
            'imports': ['javax.persistence.Entity', 'javax.persistence.Table'],
            'methods': ['getId', 'setId', 'getName', 'setName', 'equals', 'hashCode'],
            'fields': ['id', 'name', 'email', 'createdDate'],
            'annotations': ['Entity', 'Table'],
            'extends': '',
            'implements': [],
            'class_type': 'class',
            'content_preview': '@Entity @Table(name = "users") public class User { private Long id; }'
        }
    ]
    
    # Simulate the analysis process
    print("Demo Analysis Results:")
    print("=" * 50)
    
    for i, class_info in enumerate(sample_classes, 1):
        class_name = class_info['class_name']
        current_package = class_info['package']
        
        # Simple rule-based classification for demo
        if 'Controller' in class_name and 'RestController' in class_info['annotations']:
            is_misplaced = 'model' in current_package
            suggested = 'com.example.controller'
            confidence = 0.92
            reasoning = "Class name and annotations indicate REST controller functionality"
        elif 'Generator' in class_name or 'export' in str(class_info['methods']):
            is_misplaced = 'model' in current_package
            suggested = 'com.example.service'
            confidence = 0.87
            reasoning = "Contains data processing and export functionality"
        elif 'Entity' in class_info['annotations']:
            is_misplaced = False
            suggested = current_package
            confidence = 0.95
            reasoning = "Entity annotation indicates correct model placement"
        else:
            is_misplaced = False
            suggested = current_package
            confidence = 0.80
            reasoning = "No clear misplacement indicators"
        
        print(f"{i}. {class_name}")
        print(f"   Current package: {current_package}")
        print(f"   Is misplaced: {'Yes' if is_misplaced else 'No'}")
        if is_misplaced:
            print(f"   Suggested package: {suggested}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Reasoning: {reasoning}")
        print()
    
    print("Demo complete! In a real analysis, results would be saved to CSV.")

if __name__ == "__main__":
    main()