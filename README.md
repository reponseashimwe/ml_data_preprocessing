# Multimodal Data Preprocessing Assignment

## Overview

This project implements a comprehensive multimodal authentication system that combines facial recognition, voice verification, and product recommendation capabilities. The system demonstrates advanced data preprocessing techniques, feature engineering, and machine learning model integration.

## System Architecture

```
[User Input]
     ↓
[Face Recognition ✅] → [Voice Verification ✅] → [Product Recommendation ✅]
            ↘                      ↘
        [Denied ❌]            [Denied ❌]
```

## Recent Updates

### Version 1.1 (Latest)
- **Fixed Data Processing**: Resolved regex warnings and data type encoding issues
- **Improved Error Handling**: Better handling of missing values and mixed data types
- **Enhanced Compatibility**: Tested with Python 3.12 and latest scikit-learn
- **Robust Pipeline**: More reliable execution with better error messages

## Features

### 1. Data Merging and Feature Engineering
- Merges customer social profiles and transaction data
- Creates engineered features for better model performance
- Handles categorical encoding and feature scaling

### 2. Image Processing Pipeline
- Facial image collection and processing
- Multiple augmentation techniques (rotation, flipping, grayscale, brightness)
- Feature extraction using computer vision techniques
- PCA dimensionality reduction

### 3. Audio Processing Pipeline
- Voice sample collection and processing
- Audio augmentations (pitch shift, time stretch, noise addition)
- MFCC and spectral feature extraction
- Comprehensive audio analysis

### 4. Machine Learning Models
- **Facial Recognition Model**: Random Forest classifier for user identification
- **Voice Verification Model**: Logistic Regression for voice authentication
- **Product Recommendation Model**: Random Forest for personalized recommendations

### 5. Security Features
- Unauthorized access detection and blocking
- Multi-modal authentication requirements
- Confidence threshold-based decisions

## Installation and Setup

### Prerequisites
\`\`\`bash
pip install pandas numpy scikit-learn opencv-python librosa soundfile matplotlib seaborn joblib pillow
\`\`\`

```
Quick Start

1. Clone/Download the project files

2. Create virtual environment (recommended):
   bash
   python -m venv ml_pipeline_env
   source ml_pipeline_env/bin/activate  # On Windows: ml_pipeline_env\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the complete pipeline:
   python run_pipeline.py

5. Launch the demonstration system:
   python app.py
```


### Troubleshooting
If you encounter issues during pipeline execution:

**Common Issues and Solutions:**
- **Regex Warning**: Fixed in latest version - uses raw strings for regex patterns
- **Data Type Encoding Errors**: Automatically handles mixed data types in categorical columns
- **Missing Dependencies**: Ensure all packages from requirements.txt are installed
- **Python Version**: Requires Python 3.8+ (tested with Python 3.12)

**If pipeline fails:**
\`\`\`bash
# Check Python version
python --version

# Reinstall dependencies
pip install -r requirements.txt

# Run individual scripts for debugging
python scripts/data_merger.py
python scripts/image_processor.py
python scripts/audio_processor.py
python scripts/model_trainer.py
\`\`\`

```
├── data/
│   ├── audio/
│   ├── image/
│   ├── X_test.npy
│   ├── X_train.npy
│   ├── audio_analysis.png
│   ├── audio_features.csv
│   ├── customer_social_profiles.csv
│   ├── customer_transactions.csv
│   ├── image_features.csv
│   ├── merged_dataset.csv
│   ├── sample_image_display.png
│   ├── y_test.npy
│   └── y_train.npy
│
├── models/
│   ├── audio_scaler.pkl
│   ├── face_label_encoder.pkl
│   ├── face_recognition_model.pkl
│   ├── image_pca.pkl
│   ├── image_scaler.pkl
│   ├── product_recommendation_model.pkl
│   ├── training_summary.csv
│   ├── voice_label_encoder.pkl
│   └── voice_verification_model.pkl
│
├── scripts/
│   ├── audio_processor.py
│   ├── data_merge.py
│   ├── image_processor.py
│   └── model_trainer.py
│
├── .gitignore
├├── analysis.ipynb
├── app.py
├── README.md
└── run_pipeline.py
```




## Usage

### Running the Complete Pipeline
\`\`\`bash
python run_pipeline.py
\`\`\`

This will execute all pipeline steps:
1. Data merging and feature engineering
2. Image data processing
3. Audio data processing
4. Model training and evaluation

### Using the Demonstration System
\`\`\`bash
python app.py
\`\`\`

Available options:
1. **Full Transaction Simulation**: Complete authentication flow
2. **Face Authentication Only**: Test facial recognition
3. **Voice Verification Only**: Test voice authentication
4. **Unauthorized Attempt**: Simulate security breach attempt

### Jupyter Notebook Analysis
Open `analysis.ipynb` for comprehensive data exploration and model analysis.

## 📊 Key Results

| Model                    | Accuracy | F1 Score |
|--------------------------|----------|----------|
| Facial Recognition       | 0.89     | 0.89     |
| Voice Verification       | 0.75     | 0.74     |
| Product Recommendation   | 0.87     | 0.87     |

✅ Authorized Access Success Rate: 85%  
🔒 Unauthorized Blocking Effectiveness: 97%


## Data Augmentation Techniques

### Image Augmentations
- Rotation (15 degrees)
- Horizontal flipping
- Grayscale conversion
- Brightness adjustment
- Original image preservation

### Audio Augmentations
- Pitch shifting
- Time stretching
- Background noise addition
- Speed modification
- Original audio preservation

## Feature Extraction

### Image Features
- Pixel intensity values
- Histogram features
- Texture patterns (LBP approximation)
- PCA-reduced dimensionality

### Audio Features
- MFCCs (Mel-frequency cepstral coefficients)
- Spectral centroid, rolloff, bandwidth
- Zero crossing rate
- RMS energy
- Chroma features

## Security Measures

1. **Multi-modal Authentication**: Requires both face and voice verification
2. **Confidence Thresholds**: Minimum confidence levels for authentication
3. **User Matching**: Ensures face and voice belong to same user
4. **Unauthorized Detection**: Blocks access attempts with low confidence scores

## System Demonstration

The system includes comprehensive demonstration capabilities:

### Authorized User Flow
1. Face image input → Recognition success
2. Voice sample input → Verification success
3. User matching confirmation
4. Product recommendation generation
5. Transaction completion

### Unauthorized User Flow
1. Unknown face image → Recognition failure
2. Unknown voice sample → Verification failure
3. Access denied with security alert

## Future Improvements

1. **Deep Learning Integration**: CNN for images, RNN for audio
2. **Real-time Processing**: Optimize for live authentication
3. **Advanced Security**: Liveness detection, anti-spoofing
4. **Scalability**: Database integration, cloud deployment
5. **User Interface**: Web-based dashboard

## Assignment Requirements Compliance

✅ **Data Merge**: Customer profiles and transactions merged with feature engineering  
✅ **Image Processing**: 3+ facial images per member with augmentations  
✅ **Audio Processing**: 2+ voice samples per member with augmentations  
✅ **Feature Extraction**: Comprehensive feature extraction to CSV files  
✅ **Model Creation**: Three models (face, voice, product) with evaluation  
✅ **System Demonstration**: Full transaction simulation with unauthorized attempts  
✅ **Deliverables**: Complete codebase, analysis notebook, and documentation  

## Team Contributions

```
Team Contributions

```
Team Contributions

👨‍💻 John – Lead Developer & Image Processing
- Designed and implemented the facial recognition model using Random Forest
- Developed the image augmentation pipeline (grayscale, rotation, brightness)
- Led feature extraction and PCA reduction for facial images
- Coordinated the main execution script (run_pipeline.py)

👩‍🔬 Diana – Audio Processing & Voice Verification
- Engineered the audio processing pipeline using librosa and soundfile
- Applied voice augmentations (pitch shift, time stretch, background noise)
- Extracted MFCCs and spectral features for voice verification
- Built and evaluated the logistic regression voice model

🧠 Christian – Data Integration & Product Recommendation
- Merged and cleaned customer social profile and transaction datasets
- Engineered behavioral features for recommendation model
- Trained and evaluated the Random Forest product recommendation system
- Led system evaluation and performance reporting

💡 Joan – System Simulation & Report Documentation
- Developed the CLI-based demonstration system (app.py)
- Implemented unauthorized attempt detection logic
- Created the analysis notebook with performance summaries and visualizations
- Compiled final documentation, including README and presentation assets

🚀 Response Ashimwe – Project Manager & Multimodal Integration
- Led project planning, architecture design, and task coordination
- Integrated face, voice, and product modules into a unified pipeline
- Developed multimodal decision logic with confidence thresholds and blocking
- Ensured rubric alignment, final deliverables, and codebase quality
```


This project demonstrates:
- Advanced data preprocessing techniques
- Multimodal machine learning implementation
- Security-focused system design
- Comprehensive evaluation and analysis
- Production-ready code structure


