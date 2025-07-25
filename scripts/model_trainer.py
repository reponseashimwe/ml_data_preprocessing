import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.decomposition import PCA 
import joblib
import os


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        # Initialize scalers and PCA here, will be fitted in the pipeline
        self.image_scaler = StandardScaler()
        self.image_pca = PCA(n_components=5) # Ensure this matches image_processor.py
        self.audio_scaler = StandardScaler()
        
    def load_data(self):
        """
        Load all processed datasets
        """
        # Load merged dataset
        merged_data = pd.read_csv('data/merged_dataset.csv')
        
        # Load image features
        image_features = pd.read_csv('data/image_features.csv')
        
        # Load audio features
        audio_features = pd.read_csv('data/audio_features.csv')
        
        # Load preprocessed ML data
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        return merged_data, image_features, audio_features, X_train, X_test, y_train, y_test


    def train_facial_recognition_model(self, image_features):
        """
        Train facial recognition model
        """
        print("Training Facial Recognition Model...")
        
        # Prepare data
        feature_cols = [col for col in image_features.columns if col.startswith('feature_')]
        X = image_features[feature_cols]
        y = image_features['member']
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders['face'] = le
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Facial Recognition - Accuracy: {accuracy:.3f}, F1-Score: {f1:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Save model
        self.models['face_recognition'] = model
        joblib.dump(model, 'models/face_recognition_model.pkl')
        joblib.dump(le, 'models/face_label_encoder.pkl')
        
        return model, accuracy, f1
    
    def train_voice_verification_model(self, audio_features):
        """
        Train voice verification model
        """
        print("\nTraining Voice Verification Model...")
        
        # Prepare data
        feature_cols = [col for col in audio_features.columns if col.startswith('audio_feature_')]
        X = audio_features[feature_cols]
        y = audio_features['member']
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders['voice'] = le
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Voice Verification - Accuracy: {accuracy:.3f}, F1-Score: {f1:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Save model
        self.models['voice_verification'] = model
        joblib.dump(model, 'models/voice_verification_model.pkl')
        joblib.dump(le, 'models/voice_label_encoder.pkl')
        
        return model, accuracy, f1
    
    def train_product_recommendation_model(self, X_train, X_test, y_train, y_test):
        """
        Train product recommendation model
        """
        print("\nTraining Product Recommendation Model...")
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Product Recommendation - Accuracy: {accuracy:.3f}, F1-Score: {f1:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Save model
        self.models['product_recommendation'] = model
        joblib.dump(model, 'models/product_recommendation_model.pkl')
        
        return model, accuracy, f1
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}:")
            print(f"Model type: {type(model).__name__}")
            
            if hasattr(model, 'feature_importances_'):
                print(f"Number of features: {len(model.feature_importances_)}")
                print(f"Top 3 feature importances: {sorted(model.feature_importances_, reverse=True)[:3]}")
        
        return results

def main():
    """
    Main function to train all models and save preprocessors
    """
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    trainer = ModelTrainer()
    
    print("Loading data...")
    merged_data, image_features, audio_features, X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Train all models
    face_model, face_acc, face_f1 = trainer.train_facial_recognition_model(image_features)
    voice_model, voice_acc, voice_f1 = trainer.train_voice_verification_model(audio_features)
    product_model, product_acc, product_f1 = trainer.train_product_recommendation_model(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    trainer.evaluate_models()
    
    # Save training summary
    summary = {
        'facial_recognition': {'accuracy': face_acc, 'f1_score': face_f1},
        'voice_verification': {'accuracy': voice_acc, 'f1_score': voice_f1},
        'product_recommendation': {'accuracy': product_acc, 'f1_score': product_f1}
    }
    
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv('models/training_summary.csv')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(summary_df)

if __name__ == "__main__":
    main()
