#!/usr/bin/env python3
"""
Complete ML Pipeline Runner
This script runs the entire machine learning pipeline for the assignment
"""

import os
import sys
import subprocess

def run_script(script_name, description):
    """
    Run a Python script and handle errors
    """
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def create_directories():
    """
    Create necessary directories
    """
    directories = ['data', 'data/images', 'data/audio', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("📁 Created necessary directories")

def main():
    """
    Run the complete ML pipeline
    """
    print("🎯 MULTIMODAL ML PIPELINE - ASSIGNMENT RUNNER")
    print("="*60)
    print("This script will run the complete machine learning pipeline")
    print("for the multimodal data preprocessing assignment.")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Pipeline steps
    steps = [
        ("scripts/data_merger.py", "Data Merging and Feature Engineering"),
        ("scripts/image_processor.py", "Image Data Processing"),
        ("scripts/audio_processor.py", "Audio Data Processing"),
        ("scripts/model_trainer.py", "Model Training and Evaluation")
    ]
    
    # Run each step
    success_count = 0
    for script, description in steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n⚠️ Pipeline stopped due to error in: {description}")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed steps: {success_count}/{len(steps)}")
    
if success_count == len(steps):
        print("🎉 All pipeline steps completed successfully!")
        print("\nYou can now run the demonstration system:")
        print("python app.py")
        
        print("\nGenerated files:")
        print("📄 data/merged_dataset.csv - Merged and engineered dataset")
        print("📄 data/image_features.csv - Extracted image features")
        print("📄 data/audio_features.csv - Extracted audio features")
        print("🤖 models/ - Trained ML models")
        print("📊 models/training_summary.csv - Model performance summary")
        
