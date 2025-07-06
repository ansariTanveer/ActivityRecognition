import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_sensor_data(input_file, output_file):
    """
    Complete preprocessing pipeline for sensor data:
    1. Loads and filters motion/light sensors
    2. Handles timestamps and missing values
    3. Creates temporal features
    4. Encodes categorical variables
    5. Normalizes numerical features
    6. Splits into train/test sets
    """
    # 1. Load and filter data
    df = pd.read_csv(input_file, sep='\t', header=None,
                     names=['DateTime', 'Sensor', 'Translate01', 'Translate02',
                            'Message', 'SensorType', 'Activity'])

    # Filter for motion (M*) and light (LS*) sensors
    sensor_filter = df['Sensor'].str.match(r'^(M|LS)\d+', na=False)
    df = df[sensor_filter].copy()

    # 2. Convert and extract datetime features
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    df['Second'] = df['DateTime'].dt.second
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek  # Monday=0, Sunday=6

    # 3. Handle sensor messages
    # For motion sensors: Convert ON/OFF to binary
    motion_mask = df['Sensor'].str.startswith('M')
    df.loc[motion_mask, 'Message'] = (
        df.loc[motion_mask, 'Message']
        .map({'ON': 1, 'OFF': 0})
    )

    # For light sensors: Ensure numeric values
    light_mask = df['Sensor'].str.startswith('LS')
    df.loc[light_mask, 'Message'] = (
        pd.to_numeric(df.loc[light_mask, 'Message'], errors='coerce')
    )

    # 4. Feature engineering
    # Create sensor location features (from Translate01/Translate02)
    df['Room'] = np.where(df['Translate01'] != 'Ignore', df['Translate01'],
                          np.where(df['Translate02'] != 'Ignore', df['Translate02'], 'Unknown'))

    # 5. Handle categorical data
    categorical_cols = ['Sensor', 'Room', 'SensorType']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

    # 6. Handle missing values (with proper type conversion)
    df['Message'] = pd.to_numeric(df['Message'], errors='coerce')
    df['Message'] = df['Message'].fillna(df['Message'].median()).astype(float)

    # 7. Normalize numerical features
    numerical_cols = ['Message', 'Hour', 'Minute', 'Second']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 8. Create time-based aggregations INCLUDING ALL FEATURES
    df.sort_values('DateTime', inplace=True)
    window_size = '5min'

    aggregated = df.groupby(['Sensor', pd.Grouper(key='DateTime', freq=window_size)]).agg({
        'Message': ['mean', 'std', 'count'],
        'Hour': 'first',
        'Minute': 'first',
        'Second': 'first',
        'DayOfWeek': 'first',
        'Room_encoded': 'first',
        'Sensor_encoded': 'first',
        'SensorType_encoded': 'first'
    })

    # Flatten multi-index columns
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    aggregated = aggregated.reset_index()

    # 9. Select final features (now includes ALL engineered features)
    final_features = aggregated[[
        'Message_mean', 'Message_std', 'Message_count',
        'Hour_first', 'Minute_first', 'Second_first',
        'DayOfWeek_first', 'Room_encoded_first',
        'Sensor_encoded_first', 'SensorType_encoded_first'
    ]]

    # Rename columns for clarity
    final_features.columns = [
        'Message_mean', 'Message_std', 'Message_count',
        'Hour', 'Minute', 'Second', 'DayOfWeek',
        'Room_encoded', 'Sensor_encoded', 'SensorType_encoded'
    ]

    # 10. Save processed data
    final_features.to_csv(output_file, index=False)
    print(f"Data preprocessing complete. Saved to {output_file}")
    print(f"Final feature shape: {final_features.shape}")


# Run preprocessing
input_file = './hh101/hh101.rawdata.txt'
output_file = 'preprocessed_sensor_data.csv'
preprocess_sensor_data(input_file, output_file)