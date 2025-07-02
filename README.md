# Automated Outlier Detection Framework

## Description

This repository contains a Python-based automated outlier detection framework. It is designed to process datasets containing numerical time-series data and detect anomalies using various statistical and machine learning techniques.

The framework supports multiple outlier detection methods such as:
- Rolling Window
- IQR (Interquartile Range)
- Z-Score
- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- DBSCAN

The results are saved in **CSV** and/or **Excel** formats for easy analysis.

---

## Features

- **Multiple Outlier Detection Methods**: Choose from a range of statistical and machine learning approaches to detect outliers in your data.
- **Parallel Processing**: Utilizes parallel processing to speed up computation using `pandarallel`.
- **Configurable**: Easily configurable with a `config.json` file to specify dataset paths, key fields, and value fields.
- **Output Formats**: Results can be saved in both **CSV** and **Excel** formats.
- **Flexible**: Can be used with any time-series data by adjusting the configuration.

---

## Installation

### Requirements

- Python 3.8+ (Tested on Python 3.12.11)
- Required Python libraries listed in `requirements.txt`

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/bhupathiraju-renisha-hat/automated-outlier-detection.git
   
2. Navigate to the project directory:
    ```bash
    cd automated-outlier-detection

3. Install the dependencies using pip:
   ```bash
   pip install -r requirements.txt

## Configuration

1. Input Data: Place your CSV file(s) containing the dataset(s) inside the input/ directory.
The CSV should have the following fields:
Datetime field (e.g., captured_at)
Key fields (e.g., schemaname, tablename) used for grouping the data
Value fields (e.g., n_tup_ins, n_tup_upd) for which outliers need to be detected.

2. Edit config.json: Open the config.json file and modify the dataset information:
input_file: Path to your input CSV file(s).
datetime_field: Name of the datetime column in your dataset.
key_fields: List of fields used for grouping.
value_fields: List of numerical fields for which outliers will be detected.
Example:
   ```json
   {
   "datasets": [
    {
      "name": "table_ins_stats",
      "input_file": "input/table_ins_stats.csv",
      "datetime_field": "captured_at",
      "key_fields": ["schemaname", "tablename"],
      "value_fields": ["n_tup_ins", "n_tup_upd"]
    }
   ],
   "output_folder": "output/",
   "output_formats": ["excel", "csv"]
   }


3. Output: The results will be saved in the output/ directory in either CSV or Excel formats, depending on the configuration.

## Usage
# Running the Script
1. Run the detection process:
python outlier_detection.py

-Read the datasets specified in the config.json.
-Apply multiple outlier detection methods to each dataset.
-Save the output to the output/ folder in the specified formats (CSV and/or Excel).

2. Review the Output:
-After running the script, the output will be saved in the output/ folder. The output files will contain:
-Outlier detection results: A column for each method used.
-Total outliers detected: A summary of how many outliers were detected for each timestamp.

## Example Output
## Output Example

Each value field will have multiple new columns indicating outliers detected using various methods.

For example, for `n_tup_ins`, the following columns will be added:
- `outlier_rollingwindow_7d`
- `outlier_rollingwindow_30d`
- `outlier_zscore`
- `outlier_modified_zscore`
- `outlier_iqr`
- `outlier_isolationforest`
- `outlier_lof`
- `outlier_ocsvm`
- `total_outliers_detected`

Here's an example of what the output table might look like:

| timestamp           | schemaname | tablename | n_tup_ins | n_tup_upd | outlier_rollingwindow_7d | outlier_rollingwindow_30d | outlier_zscore | outlier_modified_zscore | outlier_iqr | outlier_isolationforest | outlier_lof | outlier_ocsvm | total_outliers_detected |
|---------------------|------------|-----------|-----------|-----------|---------------------------|----------------------------|----------------|--------------------------|--------------|---------------------------|--------------|----------------|---------------------------|
| 2024-01-01 00:00:00 | public     | mytable   | 500       | 200       | 0                         | 1                          | 0              | 0                        | 1            | 0                         | 1            | 0              | 3                         |
| 2024-01-02 00:00:00 | public     | mytable   | 520       | 210       | 0                         | 0                          | 0              | 0                        | 0            | 0                         | 0            | 0              | 0                         |
| 2024-01-03 00:00:00 | public     | mytable   | 8000      | 250       | 1                         | 1                          | 1              | 1                        | 1            | 1                         | 1            | 1              | 8                         |

## Outlier Detection Methods
1. Rolling Window (7d, 30d): A time-based moving average to detect sudden spikes or drops in data.
2. IQR (Interquartile Range): Detects outliers based on the spread of the data.
3. Z-Score: Detects outliers based on how far away a point is from the mean in terms of standard deviations.
4. Isolation Forest: A machine learning-based algorithm to isolate outliers from normal data points.
5. Local Outlier Factor (LOF): A density-based algorithm that finds anomalies in a local context.
6. One-Class SVM: A machine learning approach to classify data as "normal" or "outlier".
7. DBSCAN: A clustering-based algorithm to detect noise (outliers) in the data.




