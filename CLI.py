#!/usr/bin/env python3
"""
Command Line Interface for Java Class Placement Analysis Tool
"""
import argparse
import sys
import os
from pathlib import Path
from move_class_assistant import MoveClassAssistant


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Java projects for misplaced classes and suggest better package locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/java/project
  %(prog)s /path/to/project --output results.csv --confidence 0.8
  %(prog)s /path/to/project --no-embeddings --fast
        """
    )
    
    parser.add_argument(
        "project_path",
        help="Path to the Java project directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="class_placement_analysis.csv",
        help="Output CSV file path (default: class_placement_analysis.csv)"
    )
    
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for misplacement detection (default: 0.6)"
    )
    
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip semantic embedding analysis (faster but less accurate)"
    )
    
    parser.add_argument(
        "--no-dependencies",
        action="store_true", 
        help="Skip dependency analysis"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster analysis (equivalent to --no-embeddings)"
    )
    
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Limit analysis to first N classes (for testing large projects)"
    )
    
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only display metrics, don't save detailed CSV"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Java Class Placement Analyzer v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Validate project path
    if not os.path.exists(args.project_path):
        print(f"Error: Project path '{args.project_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.project_path):
        print(f"Error: '{args.project_path}' is not a directory.")
        sys.exit(1)
    
    # Check for Java files
    java_files = list(Path(args.project_path).rglob("*.java"))
    if not java_files:
        print(f"Warning: No Java files found in '{args.project_path}'")
        print("Make sure you're pointing to the root of a Java project.")
    
    if args.verbose:
        print(f"Found {len(java_files)} Java files in the project")
    
    try:
        # Initialize the assistant
        if args.verbose:
            print("Initializing Java Class Placement Analyzer...")
        
        assistant = MoveClassAssistant()
        
        # Configure analysis options
        if args.fast or args.no_embeddings:
            assistant.embedding_analyzer = None
            if args.verbose:
                print("Skipping embedding analysis for faster processing")
        
        if args.no_dependencies:
            assistant.dependency_analyzer = None
            if args.verbose:
                print("Skipping dependency analysis")
        
        # Run analysis
        print(f"Analyzing Java project: {args.project_path}")
        print("-" * 50)
        
        results_df = assistant.analyze_and_recommend(
            project_path=args.project_path,
            output_csv=None if args.metrics_only else args.output
        )
        
        # Filter by confidence if specified
        if args.confidence > 0.6:
            original_count = len(results_df[results_df['is_misplaced'] == True])
            results_df.loc[results_df['confidence'] < args.confidence, 'is_misplaced'] = False
            filtered_count = len(results_df[results_df['is_misplaced'] == True])
            
            if args.verbose and original_count != filtered_count:
                print(f"Filtered {original_count - filtered_count} low-confidence suggestions")
        
        # Display summary
        misplaced_count = len(results_df[results_df['is_misplaced'] == True])
        total_count = len(results_df)
        
        print(f"\n✅ Analysis Complete!")
        print(f"   Total classes: {total_count}")
        print(f"   Misplaced classes: {misplaced_count}")
        print(f"   Misplacement rate: {misplaced_count/total_count:.1%}")
        
        if misplaced_count > 0:
            print(f"\n🔍 Top Recommendations:")
            misplaced_df = results_df[results_df['is_misplaced'] == True].sort_values('confidence', ascending=False)
            
            for i, (_, row) in enumerate(misplaced_df.head(5).iterrows(), 1):
                print(f"   {i}. {row['class_name']}")
                print(f"      {row['current_package']} → {row['suggested_package']}")
                print(f"      Confidence: {row['confidence']:.3f}")
                if args.verbose:
                    print(f"      Reason: {row['reasoning'][:80]}...")
                print()
        
        if not args.metrics_only:
            print(f"📄 Detailed results saved to: {args.output}")
        
        # Display metrics if verbose
        if args.verbose:
            assistant.metrics.display_metrics(assistant.metrics.metrics_calculated)
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()