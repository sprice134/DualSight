
'''
import pandas as pd
import json

def get_after_first_space(s):
    parts = s.split(' ', 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return ''


f = open("yolo_nano_rmse.txt", "r")
for i in f:
    line = str(f.readline())
    data = {line.split(' ')[0]: json.loads(get_after_first_space(line).replace("'", '"'))}
    data_list = []

    # print(data)
    data_list = []

    for dataset, metrics in data.items():
        for metric, values in metrics.items():
            row = {
                'Dataset': dataset,
                'Metric': metric,
                'RMSE': values['RMSE'],
                'Mean of Predictions': values['Mean of Predictions'],
                'Mean of Ground Truth': values['Mean of Ground Truth'],
                'R2': values['R2']
            }
            data_list.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data_list)

    print(df)
'''

import pandas as pd
import json

def get_after_first_space(s):
    parts = s.split(' ', 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return ''

# Function to read and parse lines from a file
def read_and_parse_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    data_list = []
    for line in lines:
        line = line.strip()
        if line:
            data = {line.split(' ')[0]: json.loads(get_after_first_space(line).replace("'", '"'))}
            data_list.append(data)
    return data_list

# Read and parse data from both files
yolo_data = read_and_parse_file("yolo_nano_rmse.txt")
dualsight_data = read_and_parse_file("dualSight_rmse.txt")

# Combine the data into a single DataFrame
combined_data = []

for yolo, dual in zip(yolo_data, dualsight_data):
    yolo_dataset = list(yolo.keys())[0]
    dual_dataset = list(dual.keys())[0]
    print(yolo_dataset)
    for metric in yolo[yolo_dataset]:
        combined_row = {
            'image': yolo_dataset,
            'Metric': metric,
            'yolo_rmse': yolo[yolo_dataset][metric]['RMSE'],
            'dualsight_rmse': dual[dual_dataset][metric]['RMSE'],
            'yolo_pred_mean': yolo[yolo_dataset][metric]['Mean of Predictions'],
            'dual_pred_mean': dual[dual_dataset][metric]['Mean of Predictions'],
            'yolo_gt_mean': yolo[yolo_dataset][metric]['Mean of Ground Truth'],
            'dual_gt_mean': dual[dual_dataset][metric]['Mean of Ground Truth'],
            'yolo_r2_mean': yolo[yolo_dataset][metric]['R2'],
            'dual_r2_mean': dual[dual_dataset][metric]['R2']
        }
        combined_data.append(combined_row)

# Create the DataFrame
df = pd.DataFrame(combined_data)
df_filtered = df[~df['Metric'].isin(['extent', 'euler_number'])]


# for index, row in df.iterrows():
#     print(str(row))

df_filtered.to_csv('processed.csv')
