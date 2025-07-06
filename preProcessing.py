import pandas as pd


def process_sensor_data(input_file, output_file):
    """
    Process sensor data to extract only motion and light sensors,
    while ensuring proper timestamp formatting for Excel.

    Args:
        input_file (str): Path to input .txt file
        output_file (str): Path for output .csv file
    """
    # Load the data with correct column names
    df = pd.read_csv(
        input_file,
        sep='\t',
        header=None,
        names=['DateTime', 'Sensor', 'Translate01', 'Translate02',
               'Message', 'SensorType', 'Activity']
    )

    # Filter for motion sensors (M*) and light sensors (LS*) and MotionArea (MA*)
    filtered_df = df[df['Sensor'].str.match(r'^(M|LS|MA)\d+', na=False)]

    # Add empty space in beginning to display date-time:
    filtered_df.loc[:, 'DateTime'] = ' ' + filtered_df['DateTime'].astype(str)

    # Select and reorder columns for output
    output_columns = [
        'DateTime',
        'Sensor',
        'Translate01',
        'Translate02',
        'Message',
        'SensorType',
        'Activity'
    ]

    # Save to CSV
    filtered_df[output_columns].to_csv(output_file, index=False)

    print(f"Successfully processed and saved data to {output_file}")
    print(f"Original records: {len(df)} | Filtered records: {len(filtered_df)}")


# Run the processing
input_file = './hh101/hh101.rawdata.txt'
output_file = 'filtered_sensor_data.csv'
process_sensor_data(input_file, output_file)