import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_sensor_data(input_file, output_file):
    """Process sensor data with proper type handling"""
    # 1. Load data with type specification
    try:
        df = pd.read_csv(
            input_file,
            sep='\t',
            header=None,
            dtype={
                0: 'str',  # DateTime
                1: 'str',  # Sensor
                2: 'str',  # Translate01
                3: 'str',  # Translate02
                4: 'str',  # Message
                5: 'str',  # SensorType
                6: 'str'  # Activity (if exists)
            }
        )

        # Handle column names
        if len(df.columns) == 7:
            df.columns = ['DateTime', 'Sensor', 'Translate01', 'Translate02',
                          'Message', 'SensorType', 'Activity']
        else:
            df.columns = ['DateTime', 'Sensor', 'Translate01', 'Translate02',
                          'Message', 'SensorType']
            df['Activity'] = np.nan

    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

    # 2. Convert all text columns to string
    text_cols = ['Sensor', 'Translate01', 'Translate02', 'Message', 'SensorType', 'Activity']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', np.nan)  # Handle pandas NA

    # 3. Filter motion/light sensors (with string conversion)
    sensor_mask = (
            df['Sensor'].str.match(r'^(M|LS)\d+$', na=False, flags=0) &
            df['Sensor'].notna()
    )
    df = df[sensor_mask].copy()

    # 4. Process sensor messages
    motion_mask = df['Sensor'].str.startswith('M', na=False)
    df.loc[motion_mask, 'Message'] = (
        df.loc[motion_mask, 'Message']
        .replace({'ON': '1', 'OFF': '0'})
        .astype(float)
    )

    light_mask = df['Sensor'].str.startswith('LS', na=False)
    df.loc[light_mask, 'Message'] = pd.to_numeric(
        df.loc[light_mask, 'Message'],
        errors='coerce'
    )

    # 5. Feature engineering
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df[df['DateTime'].notna()]  # Remove invalid timestamps

    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    # Room identification
    df['Room'] = np.where(
        df['Translate01'] != 'Ignore',
        df['Translate01'],
        np.where(
            df['Translate02'] != 'Ignore',
            df['Translate02'],
            'Unknown'
        )
    )

    # 6. Save processed data
    df.to_csv(output_file, index=False)
    print(f"Successfully processed {len(df)} records")
    return df


# Run processing
try:
    processed_data = preprocess_sensor_data(
        input_file='./hh101/hh101.rawdata.txt',
        output_file='processed_sensors.csv'
    )
    print(processed_data.head())
except Exception as e:
    print(f"Error: {str(e)}")