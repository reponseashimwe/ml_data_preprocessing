import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import joblib 

class AudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        
    def create_sample_audio(self):
        """
        Create sample audio files for demonstration
        """
        os.makedirs('data/audio', exist_ok=True)
        
        # Define 5 members
        members = ['member1', 'member2', 'member3', 'member4', 'member5']
        phrases = ['yes_approve', 'confirm_transaction']
        
        # Clear existing synthetic audio to avoid mixing
        for member in members:
            member_dir = f'data/audio/{member}'
            os.makedirs(member_dir, exist_ok=True)
            
            # Check if the directory already contains .wav files
            existing_audio = [f for f in os.listdir(member_dir) if f.endswith('.wav')]
            
            if existing_audio:
                print(f"Skipping synthetic audio generation for {member}: Real audio already found.")
                continue # Skip to the next member if real audio is present

            # If no real audio, generate synthetic ones
            for phrase in phrases:
                # Generate synthetic audio (in real scenario, these would be recordings)
                duration = 2.0  # 2 seconds
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                
                # Create different frequency patterns for different members/phrases
                # Vary base frequency slightly for more distinct members
                member_idx = int(member.replace('member', '')) - 1
                base_freq = 150 + (member_idx * 10) # e.g., 150, 160, 170, ...
                
                if phrase == 'yes_approve':
                    freq_mod = 1.2
                else:
                    freq_mod = 0.8
                
                # Generate synthetic voice-like signal
                frequency = base_freq * freq_mod
                audio = np.sin(2 * np.pi * frequency * t)
                
                # Add harmonics
                audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
                audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
                
                # Add some noise
                noise = np.random.normal(0, 0.1, len(audio))
                audio = audio + noise
                
                # Normalize
                audio = audio / np.max(np.abs(audio))
                
                # Save audio file
                sf.write(f'{member_dir}/{phrase}.wav', audio, self.sample_rate)
        
        print("Sample audio files created successfully for 6 members (or existing real audio preserved)!")
    
    def load_and_display_audio(self):
        """
        Load and display audio samples as waveforms and spectrograms
        """
        # Dynamically get members from data/audio subdirectories
        members = [d for d in os.listdir('data/audio') if os.path.isdir(os.path.join('data/audio', d))]
        phrases = ['yes_approve', 'confirm_transaction'] # Assuming these are the phrases
        
        if not members:
            print("No audio directories found to display.")
            return

        fig, axes = plt.subplots(len(members) * 2, len(phrases), figsize=(15, len(members) * 4))
        axes = np.atleast_2d(axes) # Ensure axes is always 2D
        
        for i, member in enumerate(members):
            for j, phrase in enumerate(phrases):
                audio_path = f'data/audio/{member}/{phrase}.wav'
                # Try to find any .wav file in the directory for display
                found_audio_file = None
                member_dir = f'data/audio/{member}'
                for f in os.listdir(member_dir):
                    if f.startswith(phrase) and f.endswith('.wav'):
                        found_audio_file = os.path.join(member_dir, f)
                        break

                if found_audio_file and os.path.exists(found_audio_file):
                    # Load audio
                    audio, sr = librosa.load(found_audio_file, sr=self.sample_rate)
                    
                    # Plot waveform
                    axes[i*2, j].plot(audio)
                    axes[i*2, j].set_title(f'{member} - {phrase} (Waveform)')
                    axes[i*2, j].set_xlabel('Sample')
                    axes[i*2, j].set_ylabel('Amplitude')
                    
                    # Plot spectrogram
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                    axes[i*2+1, j].imshow(D, aspect='auto', origin='lower')
                    axes[i*2+1, j].set_title(f'{member} - {phrase} (Spectrogram)')
                    axes[i*2+1, j].set_xlabel('Time')
                    axes[i*2+1, j].set_ylabel('Frequency')
                else:
                    axes[i*2, j].set_title(f'{member} - {phrase} (N/A)')
                    axes[i*2, j].axis('off')
                    axes[i*2+1, j].set_title(f'{member} - {phrase} (N/A)')
                    axes[i*2+1, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('data/audio_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def apply_augmentations(self, audio):
        """
        Apply various augmentations to audio
        """
        augmented_audio = []
        
        # Original audio
        augmented_audio.append(('original', audio))
        
        # Pitch shift
        pitched = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=2)
        augmented_audio.append(('pitch_shift', pitched))
        
        # Time stretch
        stretched = librosa.effects.time_stretch(audio, rate=1.2)
        augmented_audio.append(('time_stretch', stretched))
        
        # Add background noise
        noise = np.random.normal(0, 0.05, len(audio))
        noisy = audio + noise
        augmented_audio.append(('noisy', noisy))
        
        # Speed change
        speed_changed = librosa.effects.time_stretch(audio, rate=0.9)
        augmented_audio.append(('speed_change', speed_changed))
        
        return augmented_audio
    
    def extract_features(self, audio):
        """
        Extract features from audio
        """
        features = []
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features.extend(np.mean(chroma, axis=1))
        
        return np.array(features)
    
    def process_all_audio(self):
        """
        Process all audio files and extract features
        """
        # Dynamically get members from data/audio subdirectories
        members = [d for d in os.listdir('data/audio') if os.path.isdir(os.path.join('data/audio', d))]
        phrases = ['yes_approve', 'confirm_transaction'] # Assuming these are the phrases
        
        all_features = []
        labels = []
        
        for member in members:
            for phrase in phrases:
                audio_path_prefix = f'data/audio/{member}/{phrase}' # Use prefix to find any file matching phrase
                
                # Find any .wav file that starts with the phrase name
                found_audio_file = None
                for f in os.listdir(f'data/audio/{member}'):
                    if f.startswith(phrase) and f.endswith('.wav'):
                        found_audio_file = os.path.join(f'data/audio/{member}', f)
                        break

                if found_audio_file and os.path.exists(found_audio_file):
                    # Load audio
                    audio, sr = librosa.load(found_audio_file, sr=self.sample_rate)
                    
                    # Apply augmentations
                    augmented_audio = self.apply_augmentations(audio)
                    
                    # Extract features from each augmented audio
                    for aug_name, aug_audio in augmented_audio:
                        features = self.extract_features(aug_audio)
                        all_features.append(features)
                        
                        # Create a more robust label based on member, original filename part, and augmentation
                        original_name_part = os.path.basename(found_audio_file).split('.')[0] # e.g., 'yes_approve' or 'John_yes_approve'
                        label = f"{member}_{original_name_part}_{aug_name}"
                        labels.append(label)
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Normalize features
        self.scaler.fit(features_array) # Fit scaler here
        features_normalized = self.scaler.transform(features_array)
        
        # Save the fitted audio scaler object
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/audio_scaler.pkl')
        print("✓ Fitted audio scaler saved to models/audio_scaler.pkl")
        
        # Create DataFrame
        feature_columns = [f'audio_feature_{i}' for i in range(features_normalized.shape[1])]
        df = pd.DataFrame(features_normalized, columns=feature_columns)
        df['label'] = labels
        df['member'] = [label.split('_')[0] for label in labels]
        # Extract phrase from the new label format (e.g., 'yes_approve' or 'John_yes_approve')
        df['phrase'] = ['_'.join(label.split('_')[1:-1]) for label in labels] 
        df['augmentation'] = [label.split('_')[-1] for label in labels]
        
        # Save features
        df.to_csv('data/audio_features.csv', index=False)
        
        print(f"Audio features extracted and saved!")
        print(f"Features shape: {features_normalized.shape}")
        print(f"Number of samples: {len(labels)}")
        
        return df

def main():
    """
    Main function to process audio
    """
    processor = AudioProcessor()
    
    print("Creating sample audio files...")
    processor.create_sample_audio()
    
    print("Loading and displaying audio...")
    processor.load_and_display_audio()
    
    print("Processing audio and extracting features...")
    features_df = processor.process_all_audio()
    
    print("Audio processing complete!")

if __name__ == "__main__":
    main()
