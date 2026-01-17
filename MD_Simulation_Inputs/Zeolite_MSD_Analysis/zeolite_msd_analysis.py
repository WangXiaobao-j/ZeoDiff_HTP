#============================================================
# Batch Processing of Zeolite Diffusion Coefficients
# Process MSD data from GROMACS simulations (msd.xvg)
# Each zeolite has 3 independent MD runs for averaging
#
# Author: Wang Xiaobao
#============================================================

import os
import numpy as np
import pandas as pd
from scipy import stats

#----------------------------
# Read .xvg file, skip comments
#----------------------------
def read_xvg(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.startswith(('#', '@')):
                data.append([float(x) for x in line.split()])
    return np.array(data)

#----------------------------
# Linear fit of MSD (1-3 ns) to obtain diffusion slope
#----------------------------
def plot_and_fit(data):
    time = data[:, 0]           # Time (ps)
    msd = data[:, 1] * 100      # MSD (nm^2 -> Å^2)

    # Select 1-3 ns range
    mask = (time >= 1000) & (time <= 3000)
    time_selected = time[mask]
    msd_selected = msd[mask]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_selected, msd_selected)
    r_squared = r_value ** 2

    return slope, intercept, r_squared

#----------------------------
# Process all zeolite folders
#----------------------------
def process_directory(base_path, output_file):
    result = []

    for zeolite_folder in os.listdir(base_path):
        zeolite_folder_path = os.path.join(base_path, zeolite_folder)
        if os.path.isdir(zeolite_folder_path):
            slopes = []

            # Assume 3 independent runs in subfolders: 1, 2, 3
            for i in range(1, 4):
                subfolder = str(i)
                msd_file = os.path.join(zeolite_folder_path, subfolder, 'msd.xvg')

                if os.path.exists(msd_file):
                    data = read_xvg(msd_file)
                    slope, intercept, r_squared = plot_and_fit(data)
                    slopes.append(slope)

            # Average slopes and calculate diffusion coefficient
            if len(slopes) == 3:
                average_slope = np.mean(slopes)
                Ds = average_slope / 2   #  For 1D channel
                result.append([zeolite_folder, slopes[0], slopes[1], slopes[2], average_slope, Ds])

    # Save results to Excel
    df = pd.DataFrame(result, columns=['Zeolites', 'slope1', 'slope2', 'slope3', 'average_slope', 'Ds'])
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

#----------------------------
# Main execution
#----------------------------
base_path = r'/path/to/zeolite/folders/'           # 根目录，包含各分子筛子文件夹
output_file = r'/path/to/output/diffusion.xlsx'    # 输出 Excel 文件

process_directory(base_path, output_file)
