import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical

def load_labels(tsv_file):
    labels_dict = {}
    with open(tsv_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.strip().split('\t')
            labels_dict[file_name] = int(label)
    return labels_dict

labels_dict = load_labels("C:/Users/frien/source/Analitics/train/targets.tsv")
#
def get_label(file_path):
    file_name = os.path.basename(file_path).replace('.wav', '')
    label = labels_dict.get(file_name, None)
    if label is None:
        print(f"No label found for {file_name}")
    return label


#
def extract_features(file_name, n_mels=40, max_length=170):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mels)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # ќбрезать или дополнить нул€ми до фиксированной длины
    if len(mfccs_scaled) < max_length:
        pad_width = max_length - len(mfccs_scaled)
        mfccs_scaled = np.pad(mfccs_scaled, pad_width=((0, pad_width),), mode='constant')
    else:
        mfccs_scaled = mfccs_scaled[:max_length]
    
    return mfccs_scaled
#
base_path = 'C:/Users/frien/source/Analitics/train2properti'



features = []
labels = []

all_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.wav')]

for file_path in all_files:
    for n_mels in [20, 30, 40]:
        data = extract_features(file_path, n_mels)
        features.append(data)
        label = get_label(file_path)
        if label is not None:  # ƒобавл€ем метку только если она найдена
            labels.append(label)

features = np.array(features)
labels = np.array(labels)
#
model = Sequential()

model.add(Dense(256, input_shape=(170,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))
print(data.shape)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print(labels)
labels = to_categorical(labels)
print("Starting training...")
model.fit(features, labels, batch_size=32, epochs=100, validation_split=0.3)
print("Training finished.")
