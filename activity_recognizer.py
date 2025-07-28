import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamp = parts[0] + ' ' + parts[1]
                sensor_id = parts[2]
                sensor_state = parts[3]
                activity = parts[4] if len(parts) > 4 else None
                data.append([timestamp, sensor_id, sensor_state, activity])
    
    df = pd.DataFrame(data, columns=['timestamp', 'sensor_id', 'sensor_state', 'activity'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def preprocess_data(df):
    df['activity'] = df['activity'].ffill()
    df.dropna(subset=['activity'], inplace=True)

    target_sensors = ['M', 'LS']
    target_activities = ['Sleeping', 'Cooking', 'Bathing', 'Dining']
    
    df = df[df['sensor_id'].str.startswith(tuple(target_sensors))]
    df = df[df['activity'].isin(target_activities)]

    if df.empty:
        raise ValueError("No data left after filtering. Check sensor IDs and activity names.")

    sensor_encoder = LabelEncoder()
    df['sensor_id_encoded'] = sensor_encoder.fit_transform(df['sensor_id'])
    
    state_encoder = OneHotEncoder(sparse_output=False)
    sensor_states_encoded = state_encoder.fit_transform(df[['sensor_state']])
    
    features = np.hstack([
        df[['sensor_id_encoded']].values,
        sensor_states_encoded
    ])
    
    activity_encoder = LabelEncoder()
    labels = activity_encoder.fit_transform(df['activity'])
    
    sequence_length = 20
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(labels[i + sequence_length - 1])
        
    X = np.array(X)
    y = np.array(y)
    
    y_categorical = to_categorical(y, num_classes=len(target_activities))
    
    return X, y_categorical, activity_encoder

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    data_path = './hh101/hh101.ann.txt'
    
    try:
        raw_df = load_data(data_path)
        X, y, activity_encoder = preprocess_data(raw_df.copy())
        
        if X.shape[0] == 0:
            print("Not enough data to create sequences. Exiting.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            num_classes = y_train.shape[1]
            
            model = build_lstm_model(input_shape, num_classes)
            
            history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)
            
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
            
            if accuracy >= 0.70:
                print("Model accuracy is above 70%. Task successful!")
            else:
                print("Model accuracy is below 70%. You might need to tune hyperparameters, use more data, or try a different architecture.")

            if len(X_test) > 0:
                sample_prediction = model.predict(X_test[0:1])
                predicted_activity_index = np.argmax(sample_prediction)
                predicted_activity = activity_encoder.inverse_transform([predicted_activity_index])
                print(f"\nExample prediction for a test sample: {predicted_activity[0]}")

    except FileNotFoundError:
        print(f"Error: The file at '{data_path}' was not found.")
        print("Please make sure the dataset is in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
