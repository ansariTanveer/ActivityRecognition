# --------------------  Working code with accuracy 60% starts here -------------------------
#<editor-fold desc="Working code with accuracy 60% starts here">
# Improved Activity Recognition with Enhanced Features
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.compose import ColumnTransformer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import to_categorical
#
# # --- Load Data ---
# # Using engine='python' to avoid potential parsing errors with the separator
# df = pd.read_csv("./hh101/hh101.ann.txt", sep="\t", header=None, engine='python')
# df.columns = ["DateTime", "Sensor", "Translate01", "Translate02", "Message", "SensorType", "Activity"]
# df["DateTime"] = pd.to_datetime(df["DateTime"], format='mixed', errors='coerce')
# df.dropna(subset=["DateTime"], inplace=True)
# df = df[df["Sensor"].str.startswith(("M", "LS"))].copy()
# df = df.sort_values("DateTime")
#
# # --- Identify Valid Activity Segments (>=5 mins) ---
# # Forward-fill activities to label sensor events between activity changes
# df['Activity'] = df['Activity'].ffill()
# df.dropna(subset=["Activity"], inplace=True)
#
# # Group by contiguous activity blocks to calculate duration
# df['activity_group'] = (df['Activity'] != df['Activity'].shift()).cumsum()
# activity_durations = df.groupby('activity_group')['DateTime'].agg(lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds())
# valid_groups = activity_durations[activity_durations >= 300].index
# filtered_df = df[df['activity_group'].isin(valid_groups)].copy()
#
# print(f"Original event count: {len(df)}, Filtered event count: {len(filtered_df)}")
# if len(filtered_df) == 0:
#     raise ValueError("No activity segments longer than 5 minutes were found. Try reducing the duration filter.")
#
# # --- Feature Engineering ---
# # 1. Basic motion and light features
# filtered_df["motion_on"] = ((filtered_df["Sensor"].str.startswith("M")) & (filtered_df["Message"] == "ON")).astype(int)
# filtered_df["light_event"] = filtered_df["Sensor"].str.startswith("LS").astype(int)
#
# # 2. Time-based features
# filtered_df["hour"] = filtered_df["DateTime"].dt.hour
#
# # 3. Location feature (crucial for performance)
# filtered_df['room'] = filtered_df['Translate01'].where(filtered_df['Translate01'] != 'Ignore', filtered_df['Translate02'])
#
# # 4. Light intensity feature
# filtered_df['light_value'] = pd.to_numeric(filtered_df['Message'], errors='coerce').fillna(0)
# # Set light_value to 0 for motion sensors
# filtered_df.loc[filtered_df['Sensor'].str.startswith('M'), 'light_value'] = 0
#
#
# # --- Preprocessing for Model ---
# # Encode Activities
# activity_encoder = LabelEncoder()
# filtered_df["ActivityCode"] = activity_encoder.fit_transform(filtered_df["Activity"])
#
# # Define columns for different transformations
# numeric_features = ['light_value', 'hour']
# categorical_features = ['room']
# passthrough_features = ['motion_on', 'light_event']
#
# # Create a preprocessor to scale numeric data and one-hot encode categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#         ('pass', 'passthrough', passthrough_features)
#     ],
#     remainder='drop'
# )
#
# # Fit and transform the features
# processed_features = preprocessor.fit_transform(filtered_df)
# feature_names = preprocessor.get_feature_names_out()
#
# # --- Sequence Building ---
# sequence_length = 20
# X_seq, y_seq = [], []
#
# for i in range(len(processed_features) - sequence_length):
#     seq = processed_features[i:i+sequence_length]
#     label = filtered_df["ActivityCode"].iloc[i+sequence_length-1]
#     X_seq.append(seq)
#     y_seq.append(label)
#
# if not X_seq:
#     raise ValueError("Not enough data to create sequences after processing. Check filters.")
#
# X_seq = np.array(X_seq)
# y_seq = np.array(y_seq)
# y_seq_cat = to_categorical(y_seq, num_classes=len(activity_encoder.classes_))
#
# # --- Split and Balance ---
# X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_cat, test_size=0.25, random_state=42, stratify=y_seq)
#
# # Class weights for imbalanced data
# y_train_labels = np.argmax(y_train, axis=1)
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
# class_weight_dict = dict(enumerate(class_weights))
#
# # --- Model ---
# # Increased model capacity slightly to handle richer features
# model = Sequential([
#     LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
#     Dropout(0.4),
#     BatchNormalization(),
#     LSTM(50),
#     Dropout(0.4),
#     Dense(50, activation='relu'),
#     Dense(len(activity_encoder.classes_), activation='softmax')
# ])
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()
#
# # --- Train ---
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
#           class_weight=class_weight_dict, callbacks=[early_stop])
#
# # --- Evaluate ---
# loss, acc = model.evaluate(X_test, y_test)
# print(f"\nImproved Test Accuracy: {acc*100:.2f}%")
#
# # --- Confusion Matrix ---
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)
# cm = confusion_matrix(y_true, y_pred)
#
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=activity_encoder.classes_, yticklabels=activity_encoder.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix (Filtered >=5 min Activities with Enhanced Features)")
# plt.show()
#
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=activity_encoder.classes_, zero_division=0))
#
# # Plot training history
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# --------------------  Working code with accuracy 60% ends here -------------------------
#</editor-fold>

#<editor-fold desc="Working code with accuracy 66.35% starts here">
# -------------------- Bidirectional LSTM starts here -------------------------

# Improved Activity Recognition with Enhanced Features
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.compose import ColumnTransformer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Sequential
# # --- CHANGE: Import Bidirectional and ReduceLROnPlateau ---
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
#
# # --- Load Data ---
# # Using engine='python' to avoid potential parsing errors with the separator
# df = pd.read_csv("./hh101/hh101.ann.txt", sep="\t", header=None, engine='python')
# df.columns = ["DateTime", "Sensor", "Translate01", "Translate02", "Message", "SensorType", "Activity"]
# df["DateTime"] = pd.to_datetime(df["DateTime"], format='mixed', errors='coerce')
# df.dropna(subset=["DateTime"], inplace=True)
# df = df[df["Sensor"].str.startswith(("M", "LS"))].copy()
# df = df.sort_values("DateTime")
#
# # --- Identify Valid Activity Segments (>=5 mins) ---
# # Forward-fill activities to label sensor events between activity changes
# df['Activity'] = df['Activity'].ffill()
# df.dropna(subset=["Activity"], inplace=True)
#
# # Group by contiguous activity blocks to calculate duration
# df['activity_group'] = (df['Activity'] != df['Activity'].shift()).cumsum()
# activity_durations = df.groupby('activity_group')['DateTime'].agg(lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds())
# valid_groups = activity_durations[activity_durations >= 300].index
# filtered_df = df[df['activity_group'].isin(valid_groups)].copy()
#
# print(f"Original event count: {len(df)}, Filtered event count: {len(filtered_df)}")
# if len(filtered_df) == 0:
#     raise ValueError("No activity segments longer than 5 minutes were found. Try reducing the duration filter.")
#
# # --- Feature Engineering ---
# # 1. Basic motion and light features
# filtered_df["motion_on"] = ((filtered_df["Sensor"].str.startswith("M")) & (filtered_df["Message"] == "ON")).astype(int)
# filtered_df["light_event"] = filtered_df["Sensor"].str.startswith("LS").astype(int)
#
# # --- CHANGE: Create cyclical time features instead of a single 'hour' ---
# # This helps the model understand the cyclical nature of a day (23:00 is close to 00:00)
# filtered_df['hour_sin'] = np.sin(2 * np.pi * filtered_df['DateTime'].dt.hour / 24.0)
# filtered_df['hour_cos'] = np.cos(2 * np.pi * filtered_df['DateTime'].dt.hour / 24.0)
#
# # 3. Location feature (crucial for performance)
# filtered_df['room'] = filtered_df['Translate01'].where(filtered_df['Translate01'] != 'Ignore', filtered_df['Translate02'])
#
# # 4. Light intensity feature
# filtered_df['light_value'] = pd.to_numeric(filtered_df['Message'], errors='coerce').fillna(0)
# # Set light_value to 0 for motion sensors
# filtered_df.loc[filtered_df['Sensor'].str.startswith('M'), 'light_value'] = 0
#
#
# # --- Preprocessing for Model ---
# # Encode Activities
# activity_encoder = LabelEncoder()
# filtered_df["ActivityCode"] = activity_encoder.fit_transform(filtered_df["Activity"])
#
# # --- CHANGE: Update numeric features to use new cyclical time features ---
# numeric_features = ['light_value', 'hour_sin', 'hour_cos']
# categorical_features = ['room']
# passthrough_features = ['motion_on', 'light_event']
#
# # Create a preprocessor to scale numeric data and one-hot encode categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#         ('pass', 'passthrough', passthrough_features)
#     ],
#     remainder='drop'
# )
#
# # Fit and transform the features
# processed_features = preprocessor.fit_transform(filtered_df)
# feature_names = preprocessor.get_feature_names_out()
#
# # --- NEW: Encapsulated Sequence Building into a function for clarity ---
# def create_sequences(features, labels, sequence_length=20):
#     X_seq, y_seq = [], []
#     for i in range(len(features) - sequence_length):
#         seq = features[i:i+sequence_length]
#         # Label of a sequence is the activity at the end of that sequence
#         label = labels.iloc[i+sequence_length-1]
#         X_seq.append(seq)
#         y_seq.append(label)
#     return np.array(X_seq), np.array(y_seq)
#
# X_seq, y_seq = create_sequences(processed_features, filtered_df["ActivityCode"], sequence_length=20)
#
# if X_seq.shape[0] == 0:
#     raise ValueError("Not enough data to create sequences after processing. Check filters.")
#
# y_seq_cat = to_categorical(y_seq, num_classes=len(activity_encoder.classes_))
#
# # --- Split and Balance ---
# X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_cat, test_size=0.25, random_state=42, stratify=y_seq)
#
# # Class weights for imbalanced data
# y_train_labels = np.argmax(y_train, axis=1)
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
# class_weight_dict = dict(enumerate(class_weights))
#
# # --- CHANGE: Updated Model Architecture ---
# # Using Bidirectional LSTM for better context understanding and slightly adjusted dropout
# model = Sequential([
#     Bidirectional(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)),
#     Dropout(0.3), # Slightly reduced dropout
#     BatchNormalization(),
#     Bidirectional(LSTM(50)), # A second bidirectional layer
#     Dropout(0.3),
#     Dense(50, activation='relu'),
#     Dense(len(activity_encoder.classes_), activation='softmax')
# ])
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()
#
# # --- CHANGE: Add ReduceLROnPlateau callback for smarter training ---
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
#
# history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
#           class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr]) # Add new callback
#
# # --- Evaluate ---
# loss, acc = model.evaluate(X_test, y_test)
# print(f"\nImproved Test Accuracy: {acc*100:.2f}%")
#
# # --- Confusion Matrix ---
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)
# cm = confusion_matrix(y_true, y_pred)
#
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=activity_encoder.classes_, yticklabels=activity_encoder.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix (Filtered >=5 min Activities with Enhanced Features)")
# plt.show()
#
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=activity_encoder.classes_, zero_division=0))
#
# # Plot training history
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
#
# plt.tight_layout()
# plt.show()


# -------------------- Bidirectional LSTM ends here -------------------------
#</editor-fold>

#<editor-fold desc="Working code with accuracy 83.50% starts here">
# --------------------  Enhanced features Bidirectional LSTM starts here -------------------------

# Improved Activity Recognition with Enhanced Features
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# --- Load Data ---
# Using engine='python' to avoid potential parsing errors with the separator
df = pd.read_csv("./hh101/hh101.ann.txt", sep="\t", header=None, engine='python')
df.columns = ["DateTime", "Sensor", "Translate01", "Translate02", "Message", "SensorType", "Activity"]
df["DateTime"] = pd.to_datetime(df["DateTime"], format='mixed', errors='coerce')
df.dropna(subset=["DateTime"], inplace=True)
df = df[df["Sensor"].str.startswith(("M", "LS"))].copy()
df = df.sort_values("DateTime")

# --- Identify Valid Activity Segments (>=5 mins) ---
df['Activity'] = df['Activity'].ffill()
df.dropna(subset=["Activity"], inplace=True)
df['activity_group'] = (df['Activity'] != df['Activity'].shift()).cumsum()
activity_durations = df.groupby('activity_group')['DateTime'].agg(lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds())
valid_groups = activity_durations[activity_durations >= 300].index
filtered_df = df[df['activity_group'].isin(valid_groups)].copy()

print(f"Original event count: {len(df)}, Filtered event count: {len(filtered_df)}")
if len(filtered_df) == 0:
    raise ValueError("No activity segments longer than 5 minutes were found. Try reducing the duration filter.")

# --- Feature Engineering ---
# 1. Basic motion and light features
filtered_df["motion_on"] = ((filtered_df["Sensor"].str.startswith("M")) & (filtered_df["Message"] == "ON")).astype(int)
filtered_df["light_event"] = filtered_df["Sensor"].str.startswith("LS").astype(int)

# 2. Cyclical time features
filtered_df['hour_sin'] = np.sin(2 * np.pi * filtered_df['DateTime'].dt.hour / 24.0)
filtered_df['hour_cos'] = np.cos(2 * np.pi * filtered_df['DateTime'].dt.hour / 24.0)

# 3. Location feature
filtered_df['room'] = filtered_df['Translate01'].where(filtered_df['Translate01'] != 'Ignore',
                                                       filtered_df['Translate02'])

# 4. Light intensity feature
filtered_df['light_value'] = pd.to_numeric(filtered_df['Message'], errors='coerce').fillna(0)
filtered_df.loc[filtered_df['Sensor'].str.startswith('M'), 'light_value'] = 0

# --- NEW: Add more sophisticated temporal features ---
# 5. Time since the last sensor event (captures rhythm)
filtered_df['time_since_last_event'] = filtered_df['DateTime'].diff().dt.total_seconds().fillna(0)
# Clip the value to prevent extreme outliers from overnight gaps from dominating the scaler
filtered_df['time_since_last_event'] = filtered_df['time_since_last_event'].clip(upper=3600)  # Cap at 1 hour

# 6. Day of the week (captures weekly routines)
filtered_df['day_of_week'] = filtered_df['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday

# --- Preprocessing for Model ---
activity_encoder = LabelEncoder()
filtered_df["ActivityCode"] = activity_encoder.fit_transform(filtered_df["Activity"])

# --- CHANGE: Update feature lists with new features ---
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


# --- CHANGE: Updated Sequence Building function for robust labeling ---
def create_sequences(features, labels, sequence_length=20):
    X_seq, y_seq = [], []
    # Iterate up to the last possible starting point for a full sequence
    for i in range(len(features) - sequence_length + 1):
        seq_features = features[i:i + sequence_length]
        seq_labels = labels.iloc[i:i + sequence_length]

        # Use the most frequent (modal) label in the sequence as the target
        # .mode()[0] handles cases with multiple modes by picking the first one.
        modal_label = seq_labels.mode()[0]

        X_seq.append(seq_features)
        y_seq.append(modal_label)
    return np.array(X_seq), np.array(y_seq)


X_seq, y_seq = create_sequences(processed_features, filtered_df["ActivityCode"], sequence_length=20)

if X_seq.shape[0] == 0:
    raise ValueError("Not enough data to create sequences after processing. Check filters.")

y_seq_cat = to_categorical(y_seq, num_classes=len(activity_encoder.classes_))

# --- Split and Balance ---
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_cat, test_size=0.25, random_state=42, stratify=y_seq)

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))

# --- Model (Unchanged) ---
# The model architecture is already powerful; the focus is on improving the data it receives.
model = Sequential([
    Bidirectional(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)),
    Dropout(0.3),
    BatchNormalization(),
    Bidirectional(LSTM(50)),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(len(activity_encoder.classes_), activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- Train (Unchanged) ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
                    class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr])

# --- Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\nImproved Test Accuracy: {acc * 100:.2f}%")

# --- Confusion Matrix and Reporting (Unchanged) ---
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=activity_encoder.classes_,
            yticklabels=activity_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Filtered >=5 min Activities with Enhanced Features)")
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

# --------------------  Enhanced features Bidirectional LSTM ends here -------------------------
#</editor-fold>