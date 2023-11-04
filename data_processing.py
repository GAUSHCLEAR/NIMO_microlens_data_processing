import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.stats import sem, t
from scipy import mean
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline



# Function to read the Excel file and extract the optical power data
def read_data(file_path):
    # 如果是excel文件，则读取excel文件
    # 如果是csv文件，则读取csv文件
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unrecognized file format: {file_path}")

    # Extract the relevant columns
    return data[['Sphere profile distance', 'Sphere profile Power']]

# Function to identify local peaks in the data with a minimum threshold
def find_local_peaks(data, minimum_peak_threshold=5):
    # Identify the peaks in the 'Sphere profile Power' column
    peaks, _ = find_peaks(data['Sphere profile Power'], height=minimum_peak_threshold)
    return peaks.tolist()

# Function to split the data into two parts if two lenses are present in one file
def split_data_if_needed(data, peak_indices):
    # If there is more than one peak, we need to split the data
    if len(peak_indices) > 1:
        # Find the midpoint between the two peaks
        midpoint = data.iloc[(peak_indices[0] + peak_indices[1]) // 2]['Sphere profile distance']
        # Split the data into two parts
        data_1 = data[data['Sphere profile distance'] <= midpoint]
        data_2 = data[data['Sphere profile distance'] > midpoint]
        return [data_1, data_2]
    else:
        # If there's only one peak, we don't need to split the data
        return [data]

# Function to apply spline smoothing to the data
def smooth_data(data):
    # Fit a spline to the data
    spline = UnivariateSpline(data['Sphere profile distance'], data['Sphere profile Power'], s=0.5)
    # Evaluate the spline over the original distances
    data = data.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    data.loc[:, 'Smoothed Power'] = spline(data['Sphere profile distance'])
    return data


# Function to find the peak of the smoothed data representing the lens center
def find_peak_center(data):
    # Find the maximum value of the smoothed power, which corresponds to the lens center
    peak_center_index = data['Smoothed Power'].idxmax()
    # Return the corresponding distance value
    return data.at[peak_center_index, 'Sphere profile distance']

# Function to align the data so that the peak is at the horizontal axis' zero point
def align_data(data, peak_center_value):
    data['Aligned Distance'] = data['Sphere profile distance'] - peak_center_value
    return data

# Function to resample multiple measurements of the same lens and calculate the mean and 95% confidence interval
def resample_and_stat_analysis(data_frames):
    # Concatenate all data frames
    concatenated_data = pd.concat(data_frames)
    # Group by the aligned distance and calculate mean and standard error
    stat_data = concatenated_data.groupby('Aligned Distance').agg({'Smoothed Power': ['mean', sem]})
    stat_data.columns = ['Mean Power', 'SE']
    # Calculate the 95% confidence interval
    ci95_hi = []
    ci95_lo = []
    for i in stat_data.index:
        m, se = stat_data.loc[i]
        ci95_hi.append(m + 1.96*se)
        ci95_lo.append(m - 1.96*se)
    stat_data['CI95 Hi'] = ci95_hi
    stat_data['CI95 Lo'] = ci95_lo
    return stat_data.reset_index()

# Function to truncate the data to the range of [-1.5, +1.5] and pad with NaN if necessary
def truncate_and_pad_data(data, step=0.01):
    min_distance = data['Aligned Distance'].min()
    max_distance = data['Aligned Distance'].max()
    spline = UnivariateSpline(data['Aligned Distance'], data['Smoothed Power'], s=0.5)
    new_index = np.arange(-1, 1, step)
        # full index为new_index在min_distance，max_distance之间的值
    full_index = new_index[(new_index >= min_distance) & (new_index <= max_distance)]
    full_interpolated_values = spline(full_index)
    # 对full_index, full_interpolated_values进行截断和padding
    # 对于new_index中的值，如果在full_index中，则取full_interpolated_values中的值
    # 如果不在full_index中，则取NaN
    truncate_and_pad_data = pd.DataFrame({'Aligned Distance': new_index,
                                            'Smoothed Power': np.nan},
                                            index=new_index)               
    truncate_and_pad_data.loc[full_index, 'Smoothed Power'] = full_interpolated_values
    return truncate_and_pad_data

# Function to plot the optical power curve with mean and 95% confidence interval
def plot_data(data, lens_id):
    plt.figure(figsize=(10, 5))
    plt.fill_between(data['Aligned Distance'], data['CI95 Lo'], data['CI95 Hi'], color='skyblue', alpha=0.5)
    plt.plot(data['Aligned Distance'], data['Mean Power'], label=f"Lens {lens_id} Mean Power")
    plt.xlabel('Sphere profile distance')
    plt.ylabel('Sphere profile Power')
    plt.title(f'Optical Power Curve for Lens {lens_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

# Function to save the lens data into a CSV file
def save_to_csv(data, file_path):
    data.to_csv(file_path, index=False)

def process_single_file(file_path, id_list, measure_id):
    # base_file_name = os.path.basename(file_path).split('.')[0]
    # Process the file
    data = read_data(file_path)
    peaks = find_local_peaks(data)
    split_data_list = split_data_if_needed(data, peaks)
    
    # Dictionary to store stats data for each lens
    data_dict = {}
    # id_list, measure_id=process_file_name(base_file_name)

    for idx, lens_data_part in enumerate(split_data_list):
        lens_id = id_list[idx]
        smoothed_data = smooth_data(lens_data_part)
        peak_center = find_peak_center(smoothed_data)
        aligned_data = align_data(smoothed_data, peak_center)
        truncated_data = truncate_and_pad_data(aligned_data)
        data_dict[lens_id]={measure_id: truncated_data}

    return data_dict



# def process_all_files_in_directory(files_list):
#     # Get all the xlsx files in the directory
#     files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx') or f.endswith('.csv')]
#     data_dict_list=[
#         process_single_file(os.path.join(directory_path, file)) for file in files
#     ]
#     merged_data_dict = {}
#     for data_dict in data_dict_list:
#         for lens_id, measure_data_dict in data_dict.items():
#             if lens_id in merged_data_dict:
#                 merged_data_dict[lens_id].update(measure_data_dict)
#             else:
#                 merged_data_dict[lens_id] = measure_data_dict
    
#     # 对于merged_data_dict中的每个lens_id，中的每个measure_id对应的数据
#     # 进行resample_and_stat_analysis
#     stats_data_dict = {}
#     for lens_id, measure_data_dict in merged_data_dict.items():
#         stats_data = resample_and_stat_analysis(list(measure_data_dict.values()))
#         stats_data_dict[lens_id] = stats_data
#         plt = plot_data(stats_data, lens_id)
#         plot_file_name = f"{lens_id}_optical_power_curve.png"
#         plt.savefig(os.path.join(directory_path, plot_file_name))
#         plt.close()
#         # Save to CSV
#         csv_file_name = f"{lens_id}_data.csv"
#         save_to_csv(stats_data, os.path.join(directory_path, csv_file_name))


def process_one_peak_single_file(file_path,peak_threshold=3):
    # base_file_name = os.path.basename(file_path).split('.')[0]
    # Process the file
    data = read_data(file_path)
    peaks = find_local_peaks(data,minimum_peak_threshold=peak_threshold)
    split_data_list = split_data_if_needed(data, peaks)
    lens_data_part=split_data_list[0]

    smoothed_data = smooth_data(lens_data_part)
    peak_center = find_peak_center(smoothed_data)
    aligned_data = align_data(smoothed_data, peak_center)
    truncated_data = truncate_and_pad_data(aligned_data)

    return truncated_data

def process_one_peak_file_list(file_list, peak_threshold=3):

    truncated_data_list=[process_one_peak_single_file(file, peak_threshold) for file in file_list]

    stats_data = resample_and_stat_analysis(truncated_data_list)
    # plt = plot_data(stats_data, "")
    return stats_data

import streamlit as st
def save_uploaded_file(uploaded_file, path):
    try:
        with open(os.path.join(path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 
    except Exception as e:
        st.error(f'Error: {e}')
        return
    
def plot_2_data(data1,data2):
    plt.figure(figsize=(10, 5))
    plt.fill_between(data1['Aligned Distance'], data1['CI95 Lo'], data1['CI95 Hi'], color='skyblue', alpha=0.5)
    plt.plot(data1['Aligned Distance'], data1['Mean Power'], label=f"Lens 1 Mean Power")

    plt.fill_between(data2['Aligned Distance'], data2['CI95 Lo'], data2['CI95 Hi'], color='olive', alpha=0.5)
    plt.plot(data2['Aligned Distance'], data2['Mean Power'], label=f"Lens 2 Mean Power")
    plt.xlabel('Sphere profile distance')
    plt.ylabel('Sphere profile Power')
    plt.title(f'Compare Optical Power for Lenses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt