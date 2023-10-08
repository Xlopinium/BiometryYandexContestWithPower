import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

def load_labels(tsv_file):
    labels_dict = {}
    with open(tsv_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.strip().split('\t')
            labels_dict[file_name] = int(label)
    return labels_dict

labels_dict = load_labels("C:/Users/frien/source/Analitics/train/targets.tsv")

def get_label(file_path):
    file_name = os.path.basename(file_path).replace('.wav', '')
    label = labels_dict.get(file_name, None)
    if label is None:
        print(f"No label found for {file_name}")
    return label

def trim_silence(audio_file, top_db=20):
    y, sr = librosa.load(audio_file, sr=None)
    y_trimmed, index = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed, sr


def extract_features(file_name, n_mels=40, max_length=60):
    audio, sample_rate = trim_silence(file_name)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mels)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)
    
    # Fourier Transform
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    
    # Combine features
    combined_features = np.hstack([mfccs_scaled, spectral_contrast_scaled, chroma])
    
    # Padding if necessary
    if len(combined_features) < max_length:
        pad_width = max_length - len(combined_features)
        combined_features = np.pad(combined_features, pad_width=((0, pad_width),), mode='constant')
    
    return combined_features

base_path = 'C:/Users/frien/source/Analitics/train'

features = []
labels = []

all_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.wav')]

for file_path in all_files:
    data = extract_features(file_path)
    features.append(data)
    label = get_label(file_path)
    if label is not None:  # ��������� ����� ������ ���� ��� �������
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

model = Sequential()

# Adding LSTM layers with more neurons
model.add(LSTM(256, input_shape=(features.shape[1], 1), return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(LSTM(256, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

labels = to_categorical(labels)

# Reshaping features for LSTM layers
features = features.reshape(features.shape[0], features.shape[1], 1)

# Adding early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

print("Starting training...")
model.fit(features, labels, batch_size=32, epochs=50, validation_split=0.3, callbacks=[early_stopping])
print("Training finished.")
model.save("gender_prediction_model.h5")