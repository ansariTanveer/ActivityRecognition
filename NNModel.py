# Improved Activity Recognition using a 1D Convolutional Neural Network
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
# --- CHANGE: Import layers for a 1D-CNN model ---
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# --- Load Data (Unchanged) ---
df = pd.read_csv("./hh101/hh101.ann.txt", sep="\t", header=None, engine='python')
df.columns = ["DateTime", "Sensor", "Translate01", "Translate02", "Message", "SensorType", "Activity"]
df["DateTime"] = pd.to_datetime(df["DateTime"], format='mixed', errors='coerce')
df.dropna(subset=["DateTime"], inplace=True)
df = df[df["Sensor"].str.startswith(("M", "LS"))].copy()
df = df.sort_values("DateTime")

# --- Identify Valid Activity Segments (>=5 mins) (Unchanged) ---
df['Activity'] = df['Activity'].ffill()
df.dropna(subset=["Activity"], inplace=True)
df['activity_group'] = (df['Activity'] != df['Activity'].shift()).cumsum()
activity_durations = df.groupby('activity_group')['DateTime'].agg(lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds())
valid_groups = activity_durations[activity_durations >= 300].index
filtered_df = df[df['activity_group'].isin(valid_groups)].copy()

print(f"Original event count: {len(df)}, Filtered event count: {len(filtered_df)}")
if len(filtered_df) == 0:
    raise ValueError("No activity segments longer than 5 minutes were found. Try reducing the duration filter.")

# --- Feature Engineering (Unchanged) ---
filtered_df["motion_on"] = ((filtered_df["Sensor"].str.startswith("M")) & (filtered_df["Message"] == "ON")).astype(int)
filtered_df["light_event"] = filtered_df["Sensor"].str.startswith("LS").astype(int)
filtered_df['hour_sin'] = np.sin(2 * np.pi * filtered_df['DateTime'].dt.hour / 24.0)
filtered_df['hour_cos'] = np.cos(2 * np.pi * filtered_df['DateTime'].dt.hour / 24.0)
filtered_df['room'] = filtered_df['Translate01'].where(filtered_df['Translate01'] != 'Ignore',
                                                       filtered_df['Translate02'])
filtered_df['light_value'] = pd.to_numeric(filtered_df['Message'], errors='coerce').fillna(0)
filtered_df.loc[filtered_df['Sensor'].str.startswith('M'), 'light_value'] = 0
filtered_df['time_since_last_event'] = filtered_df['DateTime'].diff().dt.total_seconds().fillna(0)
filtered_df['time_since_last_event'] = filtered_df['time_since_last_event'].clip(upper=3600)
filtered_df['day_of_week'] = filtered_df['DateTime'].dt.dayofweek

# --- Preprocessing for Model (Unchanged) ---
activity_encoder = LabelEncoder()
filtered_df["ActivityCode"] = activity_encoder.fit_transform(filtered_df["Activity"])
numeric_features = ['light_value', 'hour_sin', 'hour_cos', 'time_since_last_event']
categorical_features = ['room', 'day_of_week']
passthrough_features = ['motion_on', 'light_event']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('pass', 'passthrough', passthrough_features)
    ],
    remainder='drop'
)
processed_features = preprocessor.fit_transform(filtered_df)

# --- Sequence Building (Unchanged) ---
def create_sequences(features, labels, sequence_length=20):
    X_seq, y_seq = [], []
    for i in range(len(features) - sequence_length + 1):
        seq_features = features[i:i + sequence_length]
        seq_labels = labels.iloc[i:i + sequence_length]
        modal_label = seq_labels.mode()[0]
        X_seq.append(seq_features)
        y_seq.append(modal_label)
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(processed_features, filtered_df["ActivityCode"], sequence_length=20)
if X_seq.shape[0] == 0:
    raise ValueError("Not enough data to create sequences after processing. Check filters.")
y_seq_cat = to_categorical(y_seq, num_classes=len(activity_encoder.classes_))

# --- Split and Balance (Unchanged) ---
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_cat, test_size=0.25, random_state=42, stratify=y_seq)
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))


# --- NEW: Model changed to a 1D Convolutional Neural Network ---
# This architecture uses convolutional filters to find local patterns in the time-series data.
model = Sequential([
    # The input shape is (sequence_length, num_features)
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Dropout(0.4),

    Conv1D(filters=128, kernel_size=3, activation='relu'),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Dropout(0.4),

    # After finding patterns, we flatten the output to feed it into a standard classifier
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    # The final output layer has one neuron per activity class
    Dense(len(activity_encoder.classes_), activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- Train (Unchanged) ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
                    class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr])

# --- Evaluate and Report (Unchanged) ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n1D-CNN Test Accuracy: {acc * 100:.2f}%")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=activity_encoder.classes_,
            yticklabels=activity_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (1D-CNN Model)")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=activity_encoder.classes_, zero_division=0))

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()