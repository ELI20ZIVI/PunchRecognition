import os
import pandas as pd


def label_frames(csv_file, output_file_path, punch_frames):
    # Read the CSV file containing motion data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    headers = df.columns.tolist()
    # Read csv line by line and label the frames
    with open(output_file_path, 'w') as f:
        f.write(','.join(headers) + ',Label\n')  # Write headers with Label column
        for index, row in df.iterrows():
            # Skip header row
            if index == 0:
                continue

            frame = row['Frame']
            label = '0'  # Default label

            # Check if the frame falls within any punch frames
            for punch in punch_frames:
                start_frame = int(punch[0])
                end_frame = int(punch[1])
                punch_label = punch[2]
                if start_frame <= frame <= end_frame:
                    label = punch_label
                    break
            
            # Write the row with the label
            row_values = row.tolist()
            row_values.append(label)
            f.write(','.join(map(str, row_values)) + '\n')

    print(f"Labelled data saved to: {output_file_path}")


def main(label_folder="Data/Labelled"):
    # Read labelled txt files and process them
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_folder, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            print(f"Processing file: {label_file}")

            # Process each line and save the punch frames
            punch_frames = []
            label = '0'
            for line in lines:
                if ':' in line:
                    label = line.split(':')[1].strip()
                    continue  # Skip header or non-data lines

                # Check for valid frame range lines
                if '-' in line:
                    parts = line.split('-')
                    if len(parts) == 2:
                        try:
                            start_frame = int(parts[0].strip())
                            end_frame = int(parts[1].strip())

                            # end_frame - start_frame has to be > 0
                            if end_frame < start_frame:
                                print(f"Invalid frame range: Found punch frames: {start_frame} - {end_frame} with label {label}")
                                raise ValueError
                            
                            # Check if previous punch frame overlaps with current
                            if punch_frames and start_frame <= punch_frames[-1][1]:
                                print(f"Overlapping punch frames detected: {punch_frames[-1]} and {start_frame}-{end_frame}")
                                raise ValueError
                            
                            punch_frames.append([start_frame, end_frame, label])
                        except ValueError:
                            exit(1)
                    else:
                        print(f"Invalid line format in label file: {line.strip()}")
                        exit(1)

                # punch_frames.append([line.split('-')[0].strip(), line.split('-')[1].strip(), label])

            # Save processed labels to a new file
            csv_file = os.path.join("Data/ProcessedData", label_file.replace(".txt", "_accel.csv"))
            output_label_path = os.path.join(label_folder, label_file.replace(".txt", "_labelled.csv"))
            label_frames(csv_file, output_label_path, punch_frames)


if __name__ == "__main__":
    label_folder = "Data/Labelled"
    main(label_folder)