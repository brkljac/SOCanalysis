"""
# SOC_inter_comparison.py

### Copyright (c) [2025] [Branko Brkljač, Faculty of Technical Sciences, University of Novi Sad]
                         [SONATA project soil mapping team, https://sonata-nbs.com/, BioSense Institute, University of Novi Sad]

### Licensed under the Lesser General Public License (LGPL)

You may obtain a copy of the License at: https://www.gnu.org/licenses/lgpl-3.0.txt

This software is provided "as is," without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.

If you find this code useful, consider citing the following publication:

#### SONATA: Monitoring of nature infrastructure - Skill acquisition fOr NATure-bAsed solutions. 
* DOI: https://doi.org/10.3030/101159546


### Description:

* The goal of comparisons presented in this script is to validate the historical SoilAtlas SOC data's ability to 
explain or predict contemporary SOC measurements provided by newly collected SONATA soil SOC indicators.

* Comparison follows methodology in digital soil mapping and environmental modeling literature, 
where historical datasets serve as explanatory variables to validate or adjust to more recent observations.

* The script contains detailed descriptions of performed data curration and statistical analyses of defined metrics.

* The validation methodology, including the computation of error metrics and confidence intervals, follows 
the guidance outlined in the GSOC Map Cookbook Manual (FAO 2017), which provides protocols for assessing 
the quality and uncertainty of soil organic carbon spatial datasets.

* For more details and supported functionalities, please check the corresponding functions and code

### functions:
def sort_by_name_key(), fix_encoding(), replace_words(), convert_to_range_or_nan(),
print_in_columns(), save_figure(), plot_soc_vs_soc_atlas(), partition_into_groups(), 
prepare_partitioned_groups(), plot_spatial_groups_with_ranges(), plot_spatial_groups_with_names(), 
partition_with_reference_quantiles(), plot_spatial_groups_with_custom_cmap(), 
plot_spatial_groups_with_names_and_custom_cmap(), plot_histogram_kde(), 
generate_plot_comparisons_with_SoilAtlas_SOC(), atlas_SOC_validation(), 
output_formatter_dict(), plot_diff_heatmap() 

### additional requirements:
data_indicators.xlsx, data_atlas.xlsx

"""

import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample



# ######################################
# Functions and data structures
# ######################################

def sort_by_name_key(
        t
):
    if isinstance(t, str):
        s = t
    else:
        s = t[0]  # assume sequence, use first element
    nums = re.findall(r'\d+', s)
    return [int(n) for n in nums]


def fix_encoding(
        text
):
    if isinstance(text, str):
        try:
            fixed = text.encode('latin1').decode('utf-8')
            return fixed
        except UnicodeEncodeError:
            pattern = r'([\x80-\xFF])[šŠžŽ]|[šŠžŽ]([\x80-\xFF])'
            def replacer(match):
                if match.group(1):
                    return match.group(1)[-1] if len(match.group(1)) > 1 else match.group(1)
                if match.group(2):
                    return match.group(2)[-1] if len(match.group(2)) > 1 else match.group(2)
                return match.group(0)
            cleaned = re.sub(pattern, lambda m: m.group(0)[-1] if m.group(1) or m.group(2) else m.group(0), text)
            try:
                fixed = cleaned.encode('latin1').decode('utf-8')
                return fixed
            except Exception:
                return cleaned
    
    return text


def replace_words(
        text, replacements
):
    if not isinstance(text, str):
        return text
    for wrong, correct in replacements:
        # Prepare regex pattern to find words ignoring case
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)    
        def repl(match):
            # Match original case - if capitalized, capitalize replacement
            word = match.group()
            if word[0].isupper():
                return correct.capitalize()
            else:
                return correct     
        text = pattern.sub(repl, text)
    return text


class FloatRange:
    def __init__(self, range_str):
        parts = range_str.split('-')
        if len(parts) == 2:
            self.start = float(parts[0])
            self.end = float(parts[1])
        else:
            raise ValueError(f"Invalid range string: {range_str}")

    def contains(self, value):
        return self.start <= value <= self.end


def convert_to_range_or_nan(
        x
):
    if pd.isna(x):
        return np.nan
    elif x == '-':
        return np.nan
    try:
        return FloatRange(x)
    except Exception:
        return np.nan


def print_in_columns(
    data_list, columns=3, decimal_places=2,
    column_labels=None, row_labels=None, 
    generate_default_labels=False,
    bool_symbols=('T', 'F'),
    return_matrix=False
):
    def format_item(item):
        if item is None:
            return ''
        # Handle all boolean types (Python bool, NumPy bool_, np.True_)
        elif bool(item) is True and bool(item) is False or np.issubdtype(type(item), np.bool_):
            return bool_symbols[1 if bool(item) else 0]
        elif isinstance(item, (int, float)):
            return f'{item:.{decimal_places}f}'
        else:
            return str(item)
        
    raw_values = []
    if return_matrix:
        raw_values.extend(data_list)
    
    max_width = max(len(format_item(item)) for item in data_list) if data_list else 0

    if generate_default_labels and column_labels is None:
        column_labels = [f'{i+1:02d}' for i in range(columns)]
    
    if column_labels:
        header_items = column_labels[:columns]
        column_label_space = ' '.rjust(max_width)
        header_line = '  '.join(f"{str(label).center(max_width)}" for label in header_items)
        print(f"\n{column_label_space}  {header_line}")
        print()
    
    num_rows = (len(data_list) + columns - 1) // columns
    
    if generate_default_labels and row_labels is None:
        row_labels = [f'{i+1:02d}' if (i+1) % 5 == 0 else '' for i in range(num_rows)]
    
    for i in range(0, len(data_list), columns):
        row_items = data_list[i:i + columns]
        row_index = i // columns
        
        row_label = ''
        if row_labels and row_index < len(row_labels):
            row_label = str(row_labels[row_index])
            print(f"{row_label.rjust(max_width)}  ", end='')
        
        print('  '.join(f"{format_item(item).rjust(max_width)}" for item in row_items))
    
    if return_matrix:
        num_rows = (len(data_list) + columns - 1) // columns
        target_size = num_rows * columns
        
        # convert None -> nan
        arr = np.array(data_list, dtype=object)
        arr[pd.isna(arr)] = np.nan
       
        # pad with NaN
        padded_values = np.pad(arr, (0, target_size - len(arr)), 
                            mode='constant', constant_values=np.nan)        
        matrix = padded_values.reshape((num_rows, columns)).astype(float)
        return matrix
    return None


def save_figure(
        fig, filename_base, suffix, formats=['pdf', 'png'], dpi=600
):
    for fmt in formats:
        filename = f"{filename_base}{suffix}.{fmt}"
        if fmt == 'tiff':
            fig.savefig(filename, format=fmt, dpi=dpi, pil_kwargs={'compression': 'tiff_lzw'})
        else:
            fig.savefig(filename, format=fmt, dpi=dpi)


def plot_soc_vs_soc_atlas(
        sample_data, no_SOC_list_matched=None, discard_matched=False, show_both=False
):
    names = [t[0] for t in sample_data]
    soc = np.array([t[1] for t in sample_data])
    soc_atlas = np.array([t[2] for t in sample_data], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot all points (excluding matched if discarded) with empty circles
    if discard_matched and no_SOC_list_matched is not None:
        # Mask to discard matched entries (these are kept points)
        mask = [name not in no_SOC_list_matched for name in names]
        names = [names[i] for i, m in enumerate(mask) if m]
        soc = soc[mask]
        soc_atlas = soc_atlas[mask]
        ax.scatter(soc_atlas, soc, facecolors='none', edgecolors='blue', label='Matched by soil subtype with SOC value of \ncorresponding polygon in SoilAtlas')
        discarded_mask = [name in no_SOC_list_matched for name in [t[0] for t in sample_data]]
        discarded_soc = np.array([t[1] for i, t in enumerate(sample_data) if discarded_mask[i]])
        discarded_soc_atlas = np.array([t[2] for i, t in enumerate(sample_data) if discarded_mask[i]], dtype=float)
        discarded_names = [t[0] for i, t in enumerate(sample_data) if discarded_mask[i]]
        ax.scatter(discarded_soc_atlas, discarded_soc, facecolors='none', edgecolors='red', marker = 'o',
                label='Polygons in SoilAtlas w/o original SOC value\n(unpaired samples)')
        for i, name in enumerate(discarded_names):
            label = name[3:] if name.startswith("ID_") else name
            ax.annotate(label, (discarded_soc_atlas[i], discarded_soc[i]), textcoords="offset points", xytext=(5,5),
                        ha='left', fontsize=9)
    else:
        mask = [name in no_SOC_list_matched for name in names] if no_SOC_list_matched else [False]*len(names)
        ax.scatter(soc_atlas, soc, facecolors='none', edgecolors='blue', label='Matched by soil subtype with SOC value of \ncorresponding polygon in SoilAtlas')
        # Plot matched points in red if not discarded
        if no_SOC_list_matched and not discard_matched and any(mask):
            matched_soc = [t[1] for i, t in enumerate(sample_data) if mask[i]]
            matched_soc_atlas = [t[2] for i, t in enumerate(sample_data) if mask[i]]
            matched_names = [t[0] for i, t in enumerate(sample_data) if mask[i]]
            ax.scatter(matched_soc_atlas, matched_soc, facecolors='none', edgecolors='red',
                    label='Polygons in SoilAtlas w/o original SOC value\n(unpaired samples)')
            for i, name in enumerate(matched_names):
                label = name[3:] if name.startswith("ID_") else name
                ax.annotate(label, (matched_soc_atlas[i], matched_soc[i]), textcoords="offset points", xytext=(5,5),
                            ha='left', fontsize=9)
    # Add labels removing 'ID_' prefix
    for i, name in enumerate(names):
        label = name[3:] if name.startswith("ID_") else name
        ax.annotate(label, (soc_atlas[i], soc[i]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=7)

    # Compute Pearson correlation coefficient on plotted data
    if not show_both:
        corr_coef, p_value = pearsonr(soc_atlas, soc)
        # Draw regression line
        m, b = np.polyfit(soc_atlas, soc, 1)
        x_vals = np.array(ax.get_xlim())
        y_vals = m * x_vals + b
        ax.plot(x_vals, y_vals, '--', color='green', label=f'Pearson r = {corr_coef:.2f}')
    else:
        corr_coef1, p_value1 = pearsonr(soc_atlas, soc)
        m, b = np.polyfit(soc_atlas, soc, 1)
        x_vals1 = np.array(ax.get_xlim())
        y_vals1 = m * x_vals1 + b
        soc = np.array([t[1] for t in sample_data])
        soc_atlas = np.array([t[2] for t in sample_data], dtype=float)
        corr_coef2, p_value2 = pearsonr(soc_atlas, soc)
        m, b = np.polyfit(soc_atlas, soc, 1)
        x_vals2 = np.array(ax.get_xlim())
        y_vals2 = m * x_vals2 + b
        ax.plot(x_vals2, y_vals2, ':', color='brown', label=f'All samples\n  Pearson r = {corr_coef2:.2f} (p = {p_value2:.2f})')
        ax.plot(x_vals1, y_vals1, '--', color='green', label=f'W/o unpaired samples\n  Pearson r = {corr_coef1:.2f} (p = {p_value1:.2f})')

    ax.set_ylabel('SOC SONATA soil indicators (new point measurements)', fontsize=14, labelpad=20)
    ax.set_xlabel('SOC SoilAtlas (historical data from atlas polygons)', fontsize=14, labelpad=20)
    ax.set_title('Scatter Plot of SOC SoilAtlas vs new SOC measurements', fontsize=16, pad=20)
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    if not show_both:
        return fig, corr_coef, p_value
    else:
        return fig, corr_coef1, p_value1


def partition_into_groups(
        data, names, num_groups=4
):
    quantiles = np.quantile(data, np.linspace(0, 1, num_groups + 1))
    groups = {}
    for i in range(num_groups):
        low, high = quantiles[i], quantiles[i+1]
        group_names = [names[j] for j, val in enumerate(data) if low <= val <= high]  # include boundary in higher bin
        groups[f"group_{i+1}"] = group_names
    return groups


def prepare_partitioned_groups(
    sample_data, num_groups=4
):
    names = [t[0] for t in sample_data]
    soc = np.array([t[1] for t in sample_data])
    soc_atlas = np.array([t[2] for t in sample_data], dtype=float)

    # Quantiles for soc
    soc_quantiles = np.quantile(soc, np.linspace(0, 1, num_groups + 1))
    soc_atlas_quantiles = np.quantile(soc_atlas, np.linspace(0, 1, num_groups + 1))

    def partition(data, names):
        groups = {}
        for i in range(num_groups):
            low, high = soc_quantiles[i], soc_quantiles[i + 1]
            group_names = [names[j] for j, val in enumerate(data) if low <= val <= high]
            groups[f"group_{i + 1}"] = group_names
        return groups

    soc_groups = partition(soc, names)
    soc_atlas_groups = partition(soc_atlas, names)
    return soc_groups, soc_atlas_groups, soc_quantiles, soc_atlas_quantiles


def plot_spatial_groups_with_ranges(
        df_ind, groups, quantiles, title
):
    base_cmap = plt.cm.Greens
    num_groups = len(groups)
    colors = [base_cmap(0.3 + 0.15 * i) for i in range(num_groups)]

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (group_label, names) in enumerate(groups.items()):
        xs = df_ind.loc[names, 'xcoord']
        ys = df_ind.loc[names, 'ycoord']
        ax.scatter(xs, ys, color=colors[i], marker='s', label=group_label)

    # Create legend patches with quantile ranges
    legend_patches = []
    for i in range(num_groups):
        label = f"{quantiles[i]:.2f} - {quantiles[i+1]:.2f}"
        patch = mpatches.Patch(color=colors[i], label=f"Q {i+1} ({label})")
        legend_patches.append(patch)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Longitude [°]', fontsize=14, labelpad=20)
    ax.set_ylabel('Latitude [°]', fontsize=14, labelpad=20)
    ax.legend(handles=legend_patches, title="SOC quantile ranges", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_spatial_groups_with_names(
        df_ind, groups, quantiles, sample_data, title, print_full_name=True
):
    names = [t[0] for t in sample_data]
    base_cmap = plt.cm.Greens
    num_groups = len(groups)
    colors = [base_cmap(0.3 + 0.15 * i) for i in range(num_groups)]

    fig, ax = plt.subplots(figsize=(10, 8))
    printed_labels = set()  # for tracking printed labels (to avoid overlapping)

    for i, (group_label, names) in enumerate(groups.items()):
        xs = df_ind.loc[names, 'xcoord']
        ys = df_ind.loc[names, 'ycoord']
        ax.scatter(xs, ys, color=colors[i], marker='s', label=group_label)
        for name in names:
            if print_full_name:
                label = name[3:] if name.startswith("ID_") else name
            else:
                if name.startswith("ID_"):
                    # Extracting substring after first underscore up to next underscore
                    parts = name.split('_')
                    label = parts[1] if len(parts) > 1 else name
                else:
                    label = name[3:] if name.startswith("ID_") else name
            if label in printed_labels:
                continue
            printed_labels.add(label)
            x = df_ind.loc[name, 'xcoord']
            y = df_ind.loc[name, 'ycoord']
            ax.annotate(label, (x, y),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha='left',
                        fontsize=9)
    legend_patches = []
    for i in range(num_groups):
        label = f"{quantiles[i]:.2f} - {quantiles[i+1]:.2f}"
        patch = mpatches.Patch(color=colors[i], label=f"Q {i+1} ({label})")
        legend_patches.append(patch)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Longitude [°]', fontsize=14, labelpad=20)
    ax.set_ylabel('Latitude [°]', fontsize=14, labelpad=20)
    ax.legend(handles=legend_patches, title="SOC quantile ranges", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def partition_with_reference_quantiles(
        data, names, reference_quantiles
):
    groups = {}
    num_groups = len(reference_quantiles) - 1
    for i in range(num_groups):
        low, high = reference_quantiles[i], reference_quantiles[i + 1]
        group_names = [names[j] for j, val in enumerate(data) if low <= val <= high]
        groups[f"group_{i+1}"] = group_names
    return groups


def plot_spatial_groups_with_custom_cmap(
        df_ind, groups, quantiles, base_cmap, title
):
    num_groups = len(groups)
    colors = [base_cmap(0.3 + 0.15 * i) for i in range(num_groups)]
    legend_patches = []
    for i in range(num_groups):
        label = f"{quantiles[i]:.2f} - {quantiles[i+1]:.2f}"
        patch = mpatches.Patch(color=colors[i], label=f"G {i+1} ({label})")
        legend_patches.append(patch)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (group_label, names) in enumerate(groups.items()):
        xs = df_ind.loc[names, 'xcoord']
        ys = df_ind.loc[names, 'ycoord']
        ax.scatter(xs, ys, color=colors[i], marker='s', label=group_label)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Longitude [°]', fontsize=14, labelpad=20)
    ax.set_ylabel('Latitude [°]', fontsize=14, labelpad=20)
    ax.legend(handles=legend_patches, title="SOC group ranges", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_spatial_groups_with_names_and_custom_cmap(
        df_ind, groups, quantiles, base_cmap, sample_data, title, print_full_name=True
):
    names = [t[0] for t in sample_data]
    num_groups = len(groups)
    colors = [base_cmap(0.3 + 0.15 * i) for i in range(num_groups)]
    printed_labels = set()

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (group_label, names) in enumerate(groups.items()):
        xs = df_ind.loc[names, 'xcoord']
        ys = df_ind.loc[names, 'ycoord']
        ax.scatter(xs, ys, color=colors[i], marker='s', label=group_label)

        for name in names:
            if print_full_name:
                label = name[3:] if name.startswith("ID_") else name
            else:
                if name.startswith("ID_"):
                    parts = name.split('_')
                    label = parts[1] if len(parts) > 1 else name
                else:
                    label = name
            if label in printed_labels:
                continue
            printed_labels.add(label)
            x = df_ind.loc[name, 'xcoord']
            y = df_ind.loc[name, 'ycoord']
            ax.annotate(label, (x, y),
                        textcoords="offset points", xytext=(5, 5),
                        ha='left', fontsize=9)
    legend_patches = []
    for i in range(num_groups):
        label = f"{quantiles[i]:.2f} - {quantiles[i+1]:.2f}"
        patch = mpatches.Patch(color=colors[i], label=f"G {i+1} ({label})")
        legend_patches.append(patch)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Longitude [°]', fontsize=14, labelpad=20)
    ax.set_ylabel('Latitude [°]', fontsize=14, labelpad=20)
    ax.legend(handles=legend_patches, title="SOC group ranges", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_histogram_kde(
        values, binrange=None, title=None
):
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=30, binrange=binrange, kde=True, color='forestgreen', edgecolor='saddlebrown', stat='percent')
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('SOC Value', fontsize=14, labelpad=20)
    plt.ylabel('Percentage of samples [%]', fontsize=14, labelpad=20)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def generate_plot_comparisons_with_SoilAtlas_SOC(
        sample_data, no_SOC_list_matched, df_ind, 
        output_path='', formats=['pdf', 'png'], dpi=600, fig_save=True,
        discard_matched=True, show_both=True, print_full_name=False
):
    fig, _, _ = plot_soc_vs_soc_atlas(sample_data, no_SOC_list_matched) 
    filename_base = "soc_vs_soc_atlas_plot"
    filename_base = os.path.join(output_path, filename_base)
    if fig_save:
        save_figure(fig, filename_base, '_all data', formats, dpi)

    fig, _, _ = plot_soc_vs_soc_atlas(sample_data, no_SOC_list_matched, discard_matched)
    if fig_save:
        save_figure(fig, filename_base, '_only_paired', formats, dpi)

    fig, _, _ = plot_soc_vs_soc_atlas(sample_data, no_SOC_list_matched, discard_matched, show_both)
    if fig_save:
        save_figure(fig, filename_base, '_both', formats, dpi=600)

    soc_groups, soc_atlas_groups, soc_quantiles, soc_atlas_quantiles = prepare_partitioned_groups(sample_data)
    soc_atlas_aligned_groups = partition_with_reference_quantiles(
        data=np.array([t[2] for t in sample_data], dtype=float),
        names=[t[0] for t in sample_data],
        reference_quantiles=soc_quantiles
    )

    fig = plot_spatial_groups_with_ranges(df_ind, soc_groups, soc_quantiles, 
                                    "Spatial plot of SOC values from SONATA soil indicators dataset, \n" \
                                    "with groups colored according to SOC quantiles")
    if fig_save:
        filename_base = "SOC_quantiles_spatial_plot"
        filename_base = os.path.join(output_path, filename_base)
        save_figure(fig, filename_base, '_SONATA_soil_indicators', formats, dpi)

    fig = plot_spatial_groups_with_names(df_ind, soc_groups, soc_quantiles, sample_data,
                                    "Spatial plot of SOC values from SONATA soil indicators dataset, \n" \
                                    "with groups colored according to SOC quantiles", 
                                    print_full_name=False)
    if fig_save:
        filename_base = "SOC_quantiles_with_names_spatial_plot"
        filename_base = os.path.join(output_path, filename_base)
        save_figure(fig, filename_base, '_SONATA_soil_indicators', formats, dpi)

    fig = plot_spatial_groups_with_ranges(df_ind, soc_atlas_groups, soc_atlas_quantiles, 
                                    "Spatial plot of SOC values from SoilAtlas dataset, \n" \
                                    "with groups colored according to SOC quantiles")
    if fig_save:
        filename_base = "SOC_quantiles_spatial_plot"
        filename_base = os.path.join(output_path, filename_base)
        save_figure(fig, filename_base, '_SoilAtlas', formats, dpi)

    fig = plot_spatial_groups_with_names(df_ind, soc_atlas_groups, soc_atlas_quantiles, sample_data,
                                    "Spatial plot of SOC values from SoilAtlas dataset, \n" \
                                    "with groups colored according to SOC quantiles", 
                                    print_full_name)
    if fig_save:
        filename_base = "SOC_quantiles_with_names_spatial_plot"
        filename_base = os.path.join(output_path, filename_base)
        save_figure(fig, filename_base, '_SoilAtlas', formats, dpi)

    fig = plot_spatial_groups_with_custom_cmap(df_ind, soc_atlas_aligned_groups, soc_quantiles, plt.cm.Greens,
                                        "Spatial plot of SOC values from SoilAtlas dataset " \
                                        "with group colors\n aligned to quantiles of SOC values "
                                        "in SONATA soil indicators dataset")
    if fig_save:
        filename_base = "SOC_groups_spatial_plot"
        filename_base = os.path.join(output_path, filename_base)
        save_figure(fig, filename_base, '_SoilAtlas_with_SONATA_colors', formats, dpi)

    fig = plot_spatial_groups_with_names_and_custom_cmap(df_ind, soc_atlas_aligned_groups, soc_quantiles, plt.cm.Greens, sample_data,
                                        "Spatial plot of SOC values from SoilAtlas dataset " \
                                        "with group colors\n aligned to quantiles of SOC values "
                                        "in SONATA soil indicators dataset", 
                                        print_full_name=False)
    if fig_save:
        filename_base = "SOC_groups_with_names_spatial_plot"
        filename_base = os.path.join(output_path, filename_base)
        save_figure(fig, filename_base, '_SoilAtlas_with_SONATA_colors', formats, dpi)

    return soc_groups, soc_atlas_groups, soc_quantiles, soc_atlas_quantiles, soc_atlas_aligned_groups


def atlas_SOC_validation(
        names, soc, soc_atlas, humus2SOC_scaling_factor_atlas, 
        df_ind, df_atlas_indexed,
        tolerance=0.3, mae_test_threshold=None, bootstrap_iterations=1000
):   
    validated_names_list = []
    range_validation_results = []
    mse_list = []
    mae_list = []
    errors = []  # soc_atlas - soc
    rel_err_list = []   # 100*(soc_atlas - soc)/soc

    for i, name in enumerate(names):
        # compute squared and absolute errors
        err = soc_atlas[i] - soc[i]
        mse_list.append(err ** 2)
        mae_list.append(abs(err))
        errors.append(err)

        # Podtip from df_ind
        subtype = df_ind.loc[name, 'Podtip']

        # humus range string from df_atlas_indexed using subtype
        humus_range_value = df_atlas_indexed.loc[subtype, 'Humus (%) range - Atlas 1972']

        # range string into FloatRange
        if not pd.isna(humus_range_value) and isinstance(humus_range_value, str):
            humus_range = FloatRange(humus_range_value)
        elif not pd.isna(humus_range_value):
            humus_range = humus_range_value
        else:
            continue
        validated_names_list.append(name) 
        
        # scale humus range to SOC range
        atlas_SOC_start = humus_range.start * humus2SOC_scaling_factor_atlas
        atlas_SOC_end = humus_range.end * humus2SOC_scaling_factor_atlas

        # check if SOC value is within atlas SOC range (atlas validation)
        in_range = atlas_SOC_start <= soc[i] <= atlas_SOC_end
        range_validation_results.append(in_range)
    
    # soc_array = np.array([soc[i] for i, name in enumerate(names) if name in validated_names_list])
    # soc_atlas_array = np.array([soc_atlas[i] for i, name in enumerate(names) if name in validated_names_list])
    soc_array = np.array(soc)
    soc_atlas_array = np.array(soc_atlas)

    # RMSE (Root Mean Squared Error), in same units as SOC, sensitive to large errors
    rmse = np.sqrt(np.mean(mse_list)) if mse_list else None

    # Mean Bias Error (MBE), mean of the differences (SOC_atlas - SOC), indicating systematic over- or underestimation
    mbe = np.mean(errors) if errors else None
    
    # Mean Absolute Error (MAE), the magnitude of errors
    mae = np.mean(mae_list) if mae_list else None

    # Linear regression for R^2
    if len(soc_atlas_array) > 1:
        model = LinearRegression().fit(soc_atlas_array.reshape(-1, 1), soc_array)
        # Coefficient of Determination (R^2), proportion of variance explained by SOC_atlas against observed SOC
        r_squared = model.score(soc_atlas_array.reshape(-1, 1), soc_array)
        residuals = soc_array - model.predict(soc_atlas_array.reshape(-1, 1))
    else:
        r_squared = None
        residuals = None

    # Percentage of samples within tolerance (fraction of soc_atlas within: tolerance * soc)
    tol_res_list = []
    for obs, historical in zip(soc_array, soc_atlas_array):
        rel_err = (historical - obs) / obs if obs != 0 else np.nan
        rel_err_list.append(100*rel_err)
        tol_res_list.append(abs(rel_err) <= tolerance)

    percent_within_tolerance = 100 * np.mean(tol_res_list) if tol_res_list else None

    # Statistical t-test: test if mean error significantly differs from zero
    t_results = {}
    if len(errors) > 1:
        # Test if mean error significantly differs from zero,
        # i.e. null hypothesis that it is 0
        t_stat, p_two_sided = stats.ttest_1samp(errors, 0.0)
        t_results['mbe_ttest'] = {
            'statistic': t_stat,
            'df': len(errors) - 1,
            'pvalue_two_sided': p_two_sided
        }
    if len(mae_list) > 1:
        # Test null hypothesis that MAE >= mae_test_threshold; alternative is that MAE is statistically 
        # significantly smaller than the threshold (e.g. smaller than the soc_atlas median or the first quantile)
        # One sided t-test against the theshold,
        # p-value < 0.05 indicates that null hypothesis should be rejected
        if mae_test_threshold is None:
            # default mae_test_threshold is SoilAtlas SOC median value
            mae_test_threshold = np.median(soc_atlas_array)
        t_stat, p_two_sided = stats.ttest_1samp(mae_list, popmean=mae_test_threshold)
        if t_stat < 0:
            p_value_one_sided = p_two_sided / 2
        else:
            p_value_one_sided = 1 - (p_two_sided / 2)
        t_results['mae_ttest'] = {
            'statistic': t_stat,
            'df': len(mae_list) - 1,
            'pvalue_one_sided': p_value_one_sided,
            'threshold': mae_test_threshold
        }
    if len(mse_list) > 1:
        # Test null hypothesis that MSE >= mse_test_threshold;
        mse_test_threshold = mae_test_threshold **2
        t_stat, p_two_sided = stats.ttest_1samp(mse_list, popmean=mse_test_threshold)
        if t_stat < 0:
            p_value_one_sided = p_two_sided / 2
        else:
            p_value_one_sided = 1 - (p_two_sided / 2)
        t_results['mse_ttest'] = {
            'statistic': t_stat,
            'df': len(mse_list) - 1,
            'pvalue_one_sided': p_value_one_sided,
            'threshold': mse_test_threshold
        }
 
    # Shapiro-Wilk, tests the null hypothesis that regression residuals come from a normally 
    # distributed population, a high p value should suggest no strong evidence against the null hypothesis.
    if residuals is not None and len(residuals) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        # Q-Q plot figure
        fig_qq = plt.figure(figsize=(10, 8))
        ax = fig_qq.add_subplot(1, 1, 1)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.xlabel('Normal distribution theoretical quantiles', fontsize=14, labelpad=20)
        plt.ylabel('Sample quantiles of SOC regression residuals between\n observed SOC and predictions based on SoilAtlas', fontsize=14, labelpad=20)
        plt.title("Q-Q plot of SOC regression residuals", fontsize=16, pad=20)
        ax.grid(True)
        ax.set_aspect('equal') 
        plt.show(block=False)
    else:
        shapiro_stat, shapiro_p = None, None
        fig_qq = None

    # Significance test for R^2, tests against null hypothesis that the explanatory
    # power of the regression model is zero (explains no variance in observed data). 
    # In the given case, SOC_atlas is regressed against observed SOC (dependent variable)
    # Approximated using F-test: F = (R²/(k)) / ((1-R²)/(n-k-1)), k=1 for simple linear regression
    # p-value < 0.05 indicates that the null hypothesis should be rejected 
    if r_squared is not None and len(soc_array) > 3:
        n = len(soc_array)
        k = 1
        F_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
        p_value_r2 = 1 - stats.f.cdf(F_stat, k, n - k - 1)
    else:
        F_stat, p_value_r2 = None, None

    # Bootstrap confidence intervals for RMSE and MBE
    # errors = soc_atlas - soc
    # For each bootstrap iteration, errors are randomly "sampled with replacement"
    # to create "bootstrap sample", from which statistics are "recomputed".
    # After some iterations (~1000) iterations, you have a

    def bootstrap_metric(data, metric_func, n=bootstrap_iterations, alpha=0.05):
        stats_ = []
        for _ in range(n):
            sample = resample(data)
            stats_.append(metric_func(sample))
        # set limits to 2.5th and 97.5th percentiles, for alpha=0.05
        lower = np.percentile(stats_, 100 * (alpha / 2))
        upper = np.percentile(stats_, 100 * (1 - alpha / 2))
        return lower, upper

    rmse_ci = bootstrap_metric(errors, lambda x: np.sqrt(np.mean(np.square(x)))) if errors else (None, None)
    mbe_ci = bootstrap_metric(errors, np.mean) if errors else (None, None)
    mae_ci = bootstrap_metric(mae_list, np.mean) if mae_list else (None, None)

    return {
        'range_validation_results': range_validation_results,
        'validated_names_list': validated_names_list,
        'mse_list': mse_list,
        'mae_list': mae_list,
        'err_list': errors,
        'rel_err_list': rel_err_list,
        'tol_res_list': tol_res_list,
        'rmse': rmse,
        'mae': mae,
        'mbe': mbe,
        'percent_within_tolerance': percent_within_tolerance,
        'r_squared': r_squared,
        'r_squared_F_stat': F_stat,
        'r_squared_p_value': p_value_r2,    
        't_test_results': t_results,
        'shapiro_stat': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'qq_plot_figure': fig_qq,
        'rmse_confidence_interval': rmse_ci,
        'mbe_confidence_interval': mbe_ci,
        'mae_confidence_interval': mae_ci,
    }


def output_formatter_dict(
        names: list,
        no_SOC_list_discarded: list,
        data_dict: dict,
        soc: list,
        soc_atlas: list,
        list_keys: list,
        sort_key_func,
        empty_entry_value=None
):
    # Merge all names to unify output
    combined_names = sorted(set(names).union(set(no_SOC_list_discarded)), key=sort_key_func)
    
    # Build output dict
    output = {'combined_names': combined_names}
    
    # Align all listed keys
    for key in list_keys:
        original_list = data_dict.get(key, [])
        # Map name -> value
        val_map = dict(zip(names, original_list))
        # Build aligned list with placeholders for missing names
        aligned_list = [val_map.get(name, empty_entry_value) for name in combined_names]
        output[f"aligned_{key}"] = aligned_list
    
    # Align SOC values that were part of analysis
    val_map = dict(zip(names, soc))
    aligned_list = [val_map.get(name, empty_entry_value) for name in combined_names]
    output[f"aligned_soc"] = aligned_list
    val_map = dict(zip(names, soc_atlas))
    aligned_list = [val_map.get(name, empty_entry_value) for name in combined_names]
    output[f"aligned_soc_atlas"] = aligned_list
    
    # Optionally align validated_names_list and range_validation_results specifically
    if 'validated_names_list' in data_dict and 'range_validation_results' in data_dict:
        val_names = data_dict['validated_names_list']
        range_vals = data_dict['range_validation_results']
        range_map = dict(zip(val_names, range_vals))
        output['aligned_range_validation_results'] = [range_map.get(n, empty_entry_value) for n in combined_names]
        val_names_map = {n: n for n in val_names}
        output['aligned_validated_names_list'] = [val_names_map.get(n, empty_entry_value) for n in combined_names]

    return output


def plot_diff_heatmap(
    diff_mat, title=None, cmap='RdBu_r', 
    figsize=(12, 8),
    scale_limit=None, scale_label='',
    legend_values=None, legend_title=''
):
    fig, ax = plt.subplots(figsize=figsize)
    diff_mat_clean = np.where(diff_mat == None, np.nan, diff_mat)
    
    if scale_limit is not None:
        data_range = scale_limit
    else:
        data_range = np.nanmax(np.abs(diff_mat_clean))
        if np.isnan(data_range):
            data_range = 1.0
    
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='lightgray')

    im = ax.imshow(diff_mat_clean, cmap=cmap_obj, aspect='auto', 
                   vmin=-data_range, vmax=data_range)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=12)

    # Update colorbar label to indicate clipping if scale_limit used
    if scale_limit is not None:
        cbar.set_label(f'Value (clipped at ±{scale_limit})', fontsize=12)
    else:
        cbar.set_label(scale_label, fontsize=12)
    
    if legend_values is not None:
        # Add custom legend and remove colorbar
        cbar.remove()
        handles = []
        for val in legend_values:
            norm_val = (val + data_range) / (2 * data_range)  # normalize to [0,1]
            color = cmap_obj(norm_val)
            patch = mpatches.Patch(color=color, label=f"{bool(val)}")
            handles.append(patch)
        
        nan_patch = mpatches.Patch(color='lightgray', label='n/a')
        handles.append(nan_patch)

        ax.legend(handles=handles, title=legend_title, 
                  bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=14)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('SOC sample column index', fontsize=14, labelpad=20)
    ax.set_ylabel('SOC sample row index', fontsize=14, labelpad=20)
    ax.set_aspect('equal')
    
    y_ticks = np.arange(4, diff_mat_clean.shape[0], 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks+1)

    x_ticks = np.arange(2, diff_mat_clean.shape[1], 3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks+1)

    plt.tight_layout()
    plt.show(block=False)
    return fig



if __name__ == "__main__":


    # ##################################################
    # Main script
    # ##################################################

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    data_atlas_filename = "data_atlas.xlsx"
    sheet_name_atlas = 'Podtipovi data set'

    data_indicators_filename = "data_indicators.xlsx"
    sheet_name_indicators = "SONATA soil indicators DATASET_"


    fig_save = False
    out_path = os.path.join(os.getcwd(), "Datasets comparison report")


    # Rule to resolve double labels (chosing one of two possible soil subtypes)
    no_match_atlas_pairs_resolve= {
        'Solončak - ili je černozem sa znacima ranijeg zabarivanja': 'černozem sa znacima ranijeg zabarivanja', 
        'Solonjec ili Černozem slabo ogajnjačeni': 'Černozem slabo ogajnjačeni'
    }

    # if present, add  spelling error pairs (original, correct)
    replacements = [
        ('cernozem', 'černozem'),
        ('ogajnjaceni', 'ogajnjačeni'),
        ('ogajnjacena', 'ogajnjačena'),
        ('gajnjaca', 'gajnjača'),
        ('mestimicno', 'mestimično'),
        ('mestimi?no', 'mestimično'),
        ('msetimi?no', 'mestimično'),
        ('smede', 'smeđe'),
        ('mocvarno', 'močvarno'),
        ('soloncak', 'solončak'),
        ('zeljište', 'zemljište'), 
        ('zemljiste', 'zemljište'), 
        ('zivi', 'živi')
    ]



    # ##################################################
    # DATA: SONATA soil indicators
    # ##################################################

    # Load entire sheet with default dtypes (no forced conversion)
    df_ind = pd.read_excel(data_indicators_filename, sheet_name=sheet_name_indicators)
    # Automatic conversion for all columns where possible
    df_ind = df_ind.convert_dtypes()

    # Custom cleaning and conversion for comma-decimal columns
    for col in ['SOC', 'SMz\n(g cm-3)', 'SOCstock [t ha-1]', 'xcoord', 'ycoord']:
        df_ind[col] = pd.to_numeric(df_ind[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Correct soil subtype text encoding
    columns_to_fix = ['Podtip']
    for col in columns_to_fix:
        if col in df_ind.columns:
            df_ind[col] = df_ind[col].apply(fix_encoding)
            df_ind[col] = df_ind[col].apply(lambda x: replace_words(x, replacements))

    # Set index column
    print(f'''\n\nSONATA soil indicators dataset:
    \nIDs of collected SOC samples denote unique point locations and appear in groups of 3, as indicated by numerical suffix.
    Samples in the same group correspond to SOC observations from three spatially close sampling locations.
    With each sample is also associated its soil subtype, which is used or matching with soil subtype information in SoilAtlas.
    ''')
    df_ind.set_index('Name', inplace=True, drop=False)
    print(df_ind['Podtip'].to_string())
    #df_ind.loc['ID_1_1']


    # ##################################################
    # DATA: SoilAtlas
    # ##################################################

    dataset_name = 'SoilAtlas'

    # Load entire sheet with default dtypes (no forced conversion)
    df_atlas = pd.read_excel(data_atlas_filename, sheet_name=sheet_name_atlas)
    # Automatic conversion for all columns where possible
    df_atlas = df_atlas.convert_dtypes()

    # Custom cleaning and conversion for comma-decimal columns
    for col in ['Bulk Density (g/cm³) - average value -  Atlas',
                'SOC (%) - Atlas', 'OBJECTID']:
        df_atlas[col] = pd.to_numeric(df_atlas[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Correct soil subtype text encoding
    columns_to_fix = ['Podtip_Var', 'RED']
    for col in columns_to_fix:
        if col in df_atlas.columns:
            df_atlas[col] = df_atlas[col].apply(fix_encoding)
            df_atlas[col] = df_atlas[col].apply(lambda x: replace_words(x, replacements))


    # Set index column
    df_atlas.set_index('OBJECTID', inplace=True, drop=False)
    print(f'''\n\n{dataset_name} dataset:
          \nEach polygon in the atlas has unique OBJECTID. However, several polygons can have the same soil subtype.
    Matching with observed SOC samples is based on soil subtype, not location. 
    \nIn the following list are OBJECTIDs of polygons that have been selected in {dataset_name} table, as 
    single representative examples of specific soil subtypes. This attribute is used for matching with newly collected SOC samples
    from SONATA soil indicators dataset and later analysis. Listed OBJECTIDs will be used for indexing available
    soil information for specific soil subtypes from the atlas.\n
          ''')
    print(df_atlas['Podtip_Var'].to_string())
    #df_atlas.loc[3048]


    # Convert range values from string to range data type
    for col in ['Bulk Density (g/cm³) - rang- Atlas',
                'Humus (%) range - Atlas 1972']:
        df_atlas[col] = df_atlas[col].apply(convert_to_range_or_nan)

    # Conversion test:
    # df_atlas['Bulk Density (g/cm³) - rang- Atlas'][151].start
    # df_atlas['Bulk Density (g/cm³) - rang- Atlas'][151].end
    # df_atlas['Bulk Density (g/cm³) - rang- Atlas'][151].contains(1.36)
    #
    # df_atlas['Humus (%) range - Atlas 1972'][151].start
    # df_atlas['Humus (%) range - Atlas 1972'][151].end
    # df_atlas['Humus (%) range - Atlas 1972'][151].contains(1.62)


    # ##################################################
    # Note about scaling factors that were used to convert
    # soil organic humus content to soil organic carbon (SOC)
    # ##################################################
    #
    # For the conversion of SoilAtlas Humus data from 1972 into SOC was used van Bemmelen factor of 0.58, 
    # which is based on the assumption that soil organic matter is, on average, 58% carbon by mass - as given by:
    #
    # humus2SOC_scaling_factor = df_atlas['SOC (%) - Atlas'][143] / df_atlas['Humus (%) - Atlas 1972'][143]
    # humus2SOC_scaling_factor * df_atlas['Humus (%) - Atlas 1972'][143]
    #
    # which can also be expressed by 1.724, which approximately is  1/0.58, and 
    # represents a scaling factor to convert SOC to soil organic matter/humus.
    #
    # However, new laboratory samples from SONATA soil indicators dataset have been obtained
    # by using a specific correction factor of 1.33 that is widely adopted in
    # Walkley & Black wet digestion method for determining SOC. 
    # This factor is applied to account for the fact that the Walkley & Black method typically
    # only oxidizes about 75-77% of the total soil organic carbon during the analysis.
    # Thus, the corresponding humus2SOC scaling factor in this case was 1/1.33, i.e. approximately 0.75

    humus2SOC_scaling_factor_atlas = 0.58
    humus2SOC_scaling_factor_indicators = 0.75


    # ##################################################
    # Comparison with SoilAtlas data
    # ##################################################

    df_atlas_indexed = df_atlas.set_index('Podtip_Var', drop=False)

    sample_data = []
    no_match_list = []
    no_SOC_list = []
    no_SOC_list_names = []
    no_SOC_list_matched = []
    no_SOC_list_discarded = []
    discarded_sample_data = []

    for idx, row in df_ind.iterrows():
        name = idx  # row['Name']
        subtype = row['Podtip']
        soc = row['SOC']

        if subtype in df_atlas_indexed.index:
            soc_atlas = df_atlas_indexed.loc[subtype, 'SOC (%) - Atlas']
            sample_data.append((name, soc, soc_atlas))
        else:
            print(f"Warning: No soil subtype match in SoilAtlas for sample ID: {name}: {subtype}")
            no_match_list.append((name, subtype))


    # Resolve entries from df_indicators that did not have 
    # direct subtype pair (match) in SoilAtlas due to double label
    print(f"\n\n*********\n*********\nEntries from SONATA soil indicators dataset that were not matched with SoilAtlas polygons due to originally provided double labels:\n")
    if len(no_match_list) != 0:
        for name, subtype in no_match_list:
            subtype_replacement = no_match_atlas_pairs_resolve.get(subtype)
            subtype_replacement = subtype_replacement[0].upper() + subtype_replacement[1:] 
            if subtype_replacement in df_atlas_indexed.index:
                soc_atlas = df_atlas_indexed.loc[subtype_replacement, 'SOC (%) - Atlas']
                soc = df_ind.loc[name, 'SOC'] if name in df_ind.index else None
                sample_data.append((name, soc, soc_atlas))
                print(f"{name}: {subtype} \n\t --> resolved as {name}: {subtype_replacement}")

                # Since this resolver will be used, also overwrite the original string in df_ind
                df_ind.loc[name, 'Podtip'] = subtype_replacement

    sample_data = sorted(sample_data, key=sort_by_name_key)


    # List matched entries in sample_data that do not have 
    # correpsonding SOC value in SoilAtlas, i.e. where soc_atlas is nan
    print(f'''\n\n*********\n*********\nMatched entries (samples) that are missing corresponding SOC values in {dataset_name}.
          Note that matching of IDs with OBJECTID is based on soil subtype of sample and polygons in SoilAtlas.
          However successful matching based on soil subtype does not guarantee that there are SOC information available for
          that type of polygons (such soil subtype) in {dataset_name}.\n''')
    for name, soc, soc_atlas in sample_data:
        if isinstance(soc_atlas, float) and np.isnan(soc_atlas):
            subtype = df_ind.loc[name, 'Podtip']
            objectID = df_atlas_indexed.loc[subtype, 'OBJECTID']
            print(f"""{name}: {subtype} \n\t --> is missing SOC value in
                {dataset_name}: 'OBJECTID': {objectID}""")
            no_SOC_list.append((name, subtype, soc, soc_atlas, objectID))
            no_SOC_list_names.append(name)

    print(f'''\nAmong newly observed data (SOC measurements from 'SONATA soil indicators') in variable 'sample_data',
    there are {len(no_SOC_list)} entries out of {len(sample_data)} 
    for which original SOC values in 'SoilAtlas' (measurements from original 1972 study) do not exist. 
    Thus, these samples are paired corresponding soil subtypes in the atlas, but original SOC values for these soil subtypes are missing.
    These {len(no_SOC_list)} entries correspond to {int(np.ceil(len(no_SOC_list)/3)):d} unique sampling areas from which up to 3 samples were taken.\n''')

    print(f'''\nFor samples with missing SoilAtlas SOC values (no_SOC_list entries), the soc_atlas values are replaced by:
        a) midpoint of SOC range for specific soil subtype from original study (when available), or
        b) SOC data from other sources corresponding to the same soil subtype, which are also reported
           as external data in the corresponding SoilAtlas table 
          (column 'Soil Organic Carbon (SOC, %)' without 'Atlas' suffix in SoilAtlas .xlsx)
    ''')

    for name, _, _, _, objectID in no_SOC_list:
        soc_range = df_atlas.loc[objectID, 'Humus (%) range - Atlas 1972']
        if isinstance(soc_range, float) and np.isnan(soc_range):
            # replace missing SOC value from original 1972 study with alternative source provided in SoilAtlas
            soc_atlas = df_atlas.loc[objectID, 'Soil Organic Carbon (SOC, %)']
        elif isinstance(soc_range, FloatRange):
            # replace missing SOC value from original 1972 study with midpoint of orignal 1972 SOC range
            soc_atlas = (soc_range.start + soc_range.end)/2
        else:
            soc_atlas = np.nan
        if isinstance(soc_atlas, float) and not np.isnan(soc_atlas):
            index = next((i for i, t in enumerate(sample_data) if t[0] == name), None)
            sample_data[index] = (sample_data[index][0], sample_data[index][1], soc_atlas)
            no_SOC_list_matched.append(sample_data[index][0])
        else:
            # SOC entry in sample_data cannot be matched to any value in SoilAtlas - will be discarded
            index = next((i for i, t in enumerate(sample_data) if t[0] == name), None)
            no_SOC_list_discarded.append(sample_data[index][0])
            # remove entry from sample_data
            discarded_sample_data.append(sample_data[index])
            del sample_data[index]


    print(f'''\n\n*********\n*********\nFrom {len(no_SOC_list)} entries out of {len(sample_data)} 
    that did not have match with original SOC values in 'SoilAtlas' (measurements from original study),
    {len(no_SOC_list_matched)} have been resolved (matched to some alternative SOC value source in SoilAtlas),
    while {len(no_SOC_list_discarded)} entries could not be resolved and have been discarded.
    \nCorresponding lists of entry IDs are:
    1) IDs of {len(no_SOC_list)} entries without direct SOC match in SoilAtlas:\n''')
    print_in_columns(no_SOC_list_names, columns=3)
    print(f'''\n\n2) IDs of {len(no_SOC_list_matched)} entries without direct SOC match in SoilAtlas that have been resolved (matched to some alternative SOC value in SoilAtlas):\n''')
    print_in_columns(no_SOC_list_matched, columns=3)
    print(f'''\n\n3) IDs of {len(no_SOC_list_discarded)} entries without direct SOC match in SoilAtlas that could not been resolved (discarded from further comparison):\n''')
    print_in_columns(no_SOC_list_discarded, columns=3)
    
    print(f'''\n\n*********\n*********\nTotal number of SOC points that will be compared against SoilAtlas data is: {len(sample_data)}''')



    # ##################################################
    # Plot SOC values, compare their range values, spatial distributions,
    # compute SOC values correlation quantiles and parition them into groups
    # ##################################################

    fig_path = os.path.join(out_path, dataset_name, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path, exist_ok=True)
 
    (
        soc_groups, soc_atlas_groups, 
        soc_quantiles, soc_atlas_quantiles, 
        soc_atlas_aligned_groups
    ) = generate_plot_comparisons_with_SoilAtlas_SOC(
        sample_data, no_SOC_list_matched, df_ind, 
        output_path=fig_path, formats=['pdf', 'tiff'], dpi=600, fig_save=fig_save,
        discard_matched=True, show_both=True, print_full_name=False
        )



    ##################################################
    # SOC distributions: histogram and KDE estimates
    ##################################################

    names = [t[0] for t in sample_data]
    soc = np.array([t[1] for t in sample_data])
    soc_atlas = np.array([t[2] for t in sample_data], dtype=float)
    # Determine max range of SOC values
    min_val = min(np.min(soc), np.min(soc_atlas))
    max_val = max(np.max(soc), np.max(soc_atlas))
    binrange=(min_val, max_val)

    # SOC distribution, SONATA soil indicators
    fig = plot_histogram_kde(soc, binrange, 'SOC values from SONATA soil indicators dataset')
    if fig_save:
        filename_base = "SOC_distribution_estimation"
        save_figure(fig, filename_base, '_SONATA_soil_indicators', formats=['pdf', 'tiff'], dpi=600)
        
    # SOC distribution, SoilAtlas
    fig = plot_histogram_kde(soc_atlas, binrange, 'SOC values from SoilAtlas dataset')
    if fig_save:
        filename_base = "SOC_distribution_estimation"
        save_figure(fig, filename_base, '_SoilAtlas', formats=['pdf', 'tiff'], dpi=600)



    ##################################################
    # Validation of SOC values from SoilAtlas dataset 
    # based on newly observed SOC samples 
    ##################################################

    # Historical data are compared against newly collected SOC samples from SONATA soil indicators dataset
    # using multiple metrics (e.g. MSE, MAE, MBE, ...)
    # please check the return value of atlas_SOC_validation function

    # Threshold for t-test of MAE and MSE against specified value
    #mae_test_threshold = soc_quantiles[0]
    mae_test_threshold = np.median(soc_atlas)

    # Threshold for computing percentage of SoilAtlas SOC samples within relative error tolerance 
    rel_diff_tolerance = 0.3

    # Number of boostrap iterations for nonparametric prediciton 
    # of RMSE, MBE and MAE confidence intervals
    ci_bootstrap_iterations = 1000


    atlas_SOC_val_res = atlas_SOC_validation(
        names, soc, soc_atlas, humus2SOC_scaling_factor_atlas, 
        df_ind, df_atlas_indexed, 
        rel_diff_tolerance, mae_test_threshold, ci_bootstrap_iterations)

    list_keys_to_align = ['mse_list', 'mae_list', 'err_list', 'rel_err_list', 'tol_res_list']
    aligned_data = output_formatter_dict(
        names,
        no_SOC_list_discarded,
        atlas_SOC_val_res,
        soc,
        soc_atlas,
        list_keys_to_align,
        sort_by_name_key,
        empty_entry_value=None
    ) 



    # ##################################################
    # Results of SoilAtlas validation analyses - report
    # ##################################################

    filename_base = "SoilAtlas_validation_vs_SONATA_"
    filename_base = os.path.join(fig_path, filename_base)

    print(f'''\n\n*********\n*********\nNumerical comparisons of SoilAtlas historical SOC data with newly observed SOC measurements from SONATA soil indicators dataset.
    \nNote that the matching between new SOC observations and SoilAtlas historical data is not based on location, but on the value of the 'soil subtype' attribute,
    which is associated with each polygon in the atlas, as well each observed point sample in the SONATA soil indicators dataset.
    SoilAtlas does not contain specific SOC values for each polygon, rather single SOC value or SOC range for all atlas polygons of the same soil subtype.\n''')
    
    print(f'''\nInter-comparison is carried out by using several metrics and analyses, as presented in the following:
    \n1) SOC values range validation, by checking whether observed SOC value is within SoilAtlas's SOC range values 
       (when available for specific soil subtype)
    \n2) Analysis of differences in SOC between observed samples and soil subtype data from SoilAtlas: 
        a) difference: soc_atlas - soc
        b) squared difference: (soc_atlas - soc)^2
        c) absolute difference: |soc_atlas - soc|
        d) relative difference: (soc_atlas - soc)/soc, in [%]
    \n3) RMSE (Root Mean Squared Error), Mean Absolute Error (MAE) and Mean Bias Error (MBE) against SoilAtlas historical data
    \n4) Percentage of observed SOC samples within specified relative difference to SoilAtlas SOC data (in absolute terms)
    \n5) Coefficient of determination (R^2) based on simple linear regression:
       proportion of variance explained by soc_atlas against observed SOC values
    \n6) Q-Q plot analysis of regression residuals from 5)
    \n7) Tests of statistical significance (test statistics and corresponding p_values):
        a) MBE t-test, against the null hypothesis that mean error is not significantly different from zero 
        b) MAE t-test, against the null hypothesis that MAE >= mae_test_threshold
        c) MSE t-test, against the null hypothesis that MSE >= mse_test_threshold
        d) Regression residuals Shapiro-Wilk test, against the null hypothesis that regression residuals
           do not come from a normally distributed population
        e) R^2 test, against the null hypothesis that the power of the simple regression model is zero, explains no variance in observed data
    \n8) Bootstrap confidence intervals for: a) RMSE, b) MBE and c) MAE
    ''')
    print(f'''\n\n*********\nNote: 
          Printed tabular results are arranged such that in each row are 9 values (samples) corresponding to 
          3 groups of 3 different SOC measurements (samples) from SONATA soil indicators dataset.
          Each group of 3 corresponds to SOC measurements from three spatially close locations, 
          e.g. ID_1_1, ID_1_2, ID_1_3, ID_2_1, ID_2_2, etc.''')
    

    # ##################################################
    # Print SOC sample IDs and their spatial arrangement 
    # that will be consistently used for results visualization
    # ##################################################
    
    print(f"\n\n*********\n*********\nNames (IDs) of SOC samples in SONATA soil indicators dataset.")
    print_in_columns(aligned_data['combined_names'], columns=9, generate_default_labels=True)


    # ##################################################
    # Print values of SOC samples that are part of analysis, discarded 
    # and unavailable data points from now on will be printed as empty cells
    # ##################################################

    print(f"\n\n*********\n*********\nSOC observations, SONATA soil indicators dataset:")
    print_in_columns(aligned_data['aligned_soc'], columns=9, 
                     decimal_places=2, generate_default_labels=True)

    print(f"\n\n*********\n*********\nSOC SoilAtlas, historical SOC data determined by matching new SOC\nobservations soil subtype with soil subtype of the polygons in the atlas:")
    print_in_columns(aligned_data['aligned_soc_atlas'], columns=9, 
                     decimal_places=2, generate_default_labels=True)
    

    # ##################################################
    # 1) SOC values range validation
    # ##################################################

    # '  ✔  ' observed SOC is in SoilAtlas SOC range (computed by scaling available Humus range)
    # '  🞨  ' observed SOC is not in SoilAtlas SOC range (computed by scaling available Humus range)
    # '     ' empty cells, SoilAtlas SOC range for specific soil subtype is not available (no available Humus range)
   
    print(f"\n\n*********\n*********\n1) SOC values range validation, testing whether observed SOC values are within SoilAtlas SOC range values for specific soil subtype")
    
    range_val_tested_samples = np.array([1 if x is not None else 0 for x in aligned_data['aligned_range_validation_results']])
    range_val_total_tested = np.sum(range_val_tested_samples)
    range_val_in_samples = np.array([1 if x is not None and bool(x) else 0 for x in aligned_data['aligned_range_validation_results']])
    range_val_total_true = np.sum(range_val_in_samples)
    range_val_out_samples = np.array(range_val_tested_samples) & ~np.array(range_val_in_samples)
    range_val_total_false = np.sum(range_val_out_samples)

    print(f'''\n\n*********\n\t1.a) Range validation results, percent of observed SOC samples inside SoilAtlas SOC range for the same soil subtype: 
          \n\tNo. tested samples: {range_val_total_tested}
          \n\tNo. SOC values inside SoilAtlas range: {range_val_total_true} ({range_val_total_true/range_val_total_tested*100:.2f} %)
          \n\tNo. SOC values outside SoilAtlas range: {range_val_total_false} ({range_val_total_false/range_val_total_tested*100:.2f} %)
    ''')   
    
    print(f'''\n\n*********\n\t1.b) Range validation results for individual samples (SOC values inside soc_atlas_range[low, high]): 
          \n\tTest: soc_atlas_range.low <= soc <=  soc_atlas_range.high''')   
    range_validation_mat = print_in_columns(aligned_data['aligned_range_validation_results'], columns=9, 
        decimal_places=2, generate_default_labels=True, bool_symbols=('  ✔  ', '  🞨  '), return_matrix=True)

    print(f'''\n\n*********\nNote: 
          '  ✔  ' : observed SOC is in SoilAtlas SOC range
          '  🞨  ' : observed SOC is not in SoilAtlas SOC range
          ' empty' : SoilAtlas SOC range for specific soil subtype is not available
          \n
          Please also note that SoilAtlas SOC range values were computed by conversion of original 
          SoilAtlas Humus range data to SOC using van Bemmelen humus2SOC scaling factor of 0.58, 
          which is based on the assumption that soil organic matter is, on average, 58% carbon by mass.
          
          This factor can be also expressed as 1.724, which is approximately 1/0.58, and represents an 
          opposite scaling factor, to convert SOC to soil organic matter/humus. 

          Samples from SONATA soil indicators dataset have been obtained by using a specific 
          correction factor of 1.33 that is widely adopted in Walkley & Black wet digestion method
          for determining SOC.

          This factor is applied to account for the fact that the Walkley & Black method typically
          only oxidizes about 75-77% of the total soil organic carbon during the analysis.
          
          Thus, the corresponding humus2SOC scaling factor for observed SOC was 1/1.33, i.e. approximately 0.75.
          ''')
    
    fig = plot_diff_heatmap(
        range_validation_mat, 
        title="${SOC_{atlas}^{low}} \\leq SOC \\leq {SOC_{atlas}^{high}}$",
        cmap='ocean_r',
        figsize=(5.3, 8),
        legend_values=[0.0, 1.0],
        legend_title="SOC in range:"
    )
    save_figure(fig, filename_base, '_SOC_rangeValidation', formats=['pdf', 'tiff'], dpi=600)


    # ##################################################
    # 2) Differences in SOC values between observed
    #  samples and soil subtype data from SoilAtlas
    # ##################################################

    print(f"\n\n*********\n*********\n2) Differences in SOC values between observed samples and data from SoilAtlas (matched based on soil subtype)")

    # SOC difference
    print(f"\n\n*********\n\t2.a) SOC difference: soc_atlas - soc:")
    diff_mat = print_in_columns(aligned_data['aligned_err_list'], columns=9, 
        decimal_places=2, generate_default_labels=True, return_matrix=True)
    fig = plot_diff_heatmap(
        diff_mat, 
        title='$SOC_\\Delta = SOC_{atlas} - SOC_{obs}$',
        figsize=(5, 8)
    )
    save_figure(fig, filename_base, '_SOC_Diff', formats=['pdf', 'tiff'], dpi=600)


    # SOC squared difference
    print(f"\n\n*********\n\t2.b) SOC squared difference: (soc_atlas - soc)^2")
    mse_mat = print_in_columns(aligned_data['aligned_mse_list'], columns=9, 
        decimal_places=2, generate_default_labels=True, return_matrix=True)
    fig = plot_diff_heatmap(
        mse_mat, 
        title='$SOC^2_\\Delta = (SOC_{atlas} - SOC_{obs})^2$',
        figsize=(5, 8)
    )
    save_figure(fig, filename_base, '_SOC_SquaredDiff', formats=['pdf', 'tiff'], dpi=600)


    # SOC absolute difference
    print(f"\n\n*********\n\t2.c) SOC absolute difference: |soc_atlas - soc|")
    mae_mat = print_in_columns(aligned_data['aligned_mae_list'], columns=9, 
        decimal_places=2, generate_default_labels=True, return_matrix=True)
    fig = plot_diff_heatmap(
        mae_mat, 
        title='$|SOC_\\Delta| = |SOC_{atlas} - SOC_{obs}|$',
        figsize=(5, 8)
    )
    save_figure(fig, filename_base, '_SOC_AbsDiff', formats=['pdf', 'tiff'], dpi=600)


    # SOC relative difference
    print(f"\n\n*********\n\t2.d) SOC relative difference in [%]: 100 * (soc_atlas - soc)/soc")
    rel_err_mat = print_in_columns(aligned_data['aligned_rel_err_list'], columns=9, 
        decimal_places=1, generate_default_labels=True, return_matrix=True)
    fig = plot_diff_heatmap(
        rel_err_mat, 
        title='$\\widetilde{SOC}_{\\Delta} = (SOC_{atlas} - SOC_{obs})/SOC_{obs} \\cdot 100$',
        figsize=(5, 8),
        scale_limit = 100
    )
    save_figure(fig, filename_base, '_SOC_RelDiff', formats=['pdf', 'tiff'], dpi=600)


    # ##################################################
    # 3) RMSE (Root Mean Squared Error) 
    #    MAE (Mean Absolute Error) and 
    #    MBE (Mean Bias Error)
    # ##################################################
    
    print(f"\n\n*********\n3) SOC RMSE, MAE and MBE")          
    print(f'''\n\n*********\n\t3.a) SOC RMSE (Root Mean Squared Error): sqrt( (1/n) * sum( (soc_atlas_i - soc_i)^2 ) )
    \n\t     RMSE = {atlas_SOC_val_res['rmse']:.2f}
    \n\t     RMSE confidence interval: [ {atlas_SOC_val_res['rmse_confidence_interval'][0]:.2f}, {atlas_SOC_val_res['rmse_confidence_interval'][1]:.2f} ]
    ''')

    print(f'''\n\n*********\n\t3.b) SOC MAE (Mean Absolute Error): (1/n) * sum( |soc_atlas_i - soc_i| ) 
    \n\t     MAE = {atlas_SOC_val_res['mae']:.2f}
    \n\t     MAE confidence interval: [ {atlas_SOC_val_res['mae_confidence_interval'][0]:.2f}, {atlas_SOC_val_res['mae_confidence_interval'][1]:.2f} ]
    ''')

    print(f'''\n\n*********\n\t3.c) SOC MBE (Mean Bias Error): (1/n) * sum( (soc_atlas_i - soc_i) ) 
    \n\t     MBE = {atlas_SOC_val_res['mbe']:.2f}
    \n\t     MBE confidence interval: [ {atlas_SOC_val_res['mbe_confidence_interval'][0]:.2f}, {atlas_SOC_val_res['mbe_confidence_interval'][1]:.2f} ]
    ''')
          

    # ##################################################
    # 4) Percentage of observed SOC samples with absolute
    #    relative difference smaller than given tolerance
    # ##################################################

    print(f'''\n\n*********\n*********\n4) Percentage of observed SOC samples that are within specified absolute relative difference, when compared to historical SoilAtlas data.
          
        Test threshold value is set to absolute relative difference of {rel_diff_tolerance*100:.0f} % between
        newly observed SOC measurements and historical SoilAtlas SOC data. 
    
        Note that the matching between observed SOC samples and SoilAtlas polygons was not based on spatial location, 
        but on the value of soil subtype attribute that is associated with each newly observed SOC point measurement.
        Polygons in the SoilAtlas with same soil subtype share the same SOC value (when available).
    ''')  

    print(f'''\n\n*********\n\t4.a) Percentage of observed SOC samples with absolute relative difference smaller than specified threshold: 
    \n\t     Percentage of samples within tolerance = {atlas_SOC_val_res['percent_within_tolerance']:.2f} [%]
    \n\t     rel_diff_tolerance =  {rel_diff_tolerance*100:.2f} [%]
    ''')


    print(f'''\n\n*********\n\t4.b) Results of abs.rel. diff. comparisons for individual SOC samples (abs. rel. diff < threshold): 
          \n\tTest: 100 * |(soc_atlas - soc)/soc| < rel_diff_tolerance''')
    tol_res_mat = print_in_columns(aligned_data['aligned_tol_res_list'], columns=9, 
        decimal_places=2, generate_default_labels=True, bool_symbols=('  ✔  ', '  🞨  '), return_matrix=True)
    print(f'''\n\n*********\nNote: 
          '  ✔  ' : absolute relative difference < specfied threshold
          '  🞨  ' : absolute relative difference > specfied threshold
          ' empty' : relative difference to SoilAtlas SOC is not available
          \n
    ''')
    fig = plot_diff_heatmap(
        tol_res_mat, 
        title="$100 \\cdot|(SOC_{{atlas}} - SOC_{{obs}})/SOC_{{obs}}|  < {:.0f} \\%$".format(rel_diff_tolerance*100),
        cmap='ocean_r',
        figsize=(5.3, 8),
        legend_values=[0.0, 1.0],
        legend_title="$\\widetilde{{SOC}}_{{\\Delta}} < {:.0f} \\%$:".format(rel_diff_tolerance*100)
    )
    save_figure(fig, filename_base, '_SOC_RelDiffValidation', formats=['pdf', 'tiff'], dpi=600)



    # ##################################################
    # 5) Coefficient of determination (R^2) based on simple linear regression
    # ##################################################

    print(f'''\n\n*********\n*********\n5) Coefficient of determination (R^2) for simple linear regression of observed SOC based on SoilAtals data. 
       Describes proportion of SOC values variance explained by soc_atlas.
    \n\tR^2 = {atlas_SOC_val_res['r_squared']:.2f}
    ''')


    # ##################################################
    # 6) Q-Q plot analysis of regression residuals from 5)
    # ##################################################

    print(f'''\n\n*********\n*********\n6) Q-Q plot analysis of regression residuals. 
       Sample quantiles of SOC regression residuals compared to normal distribution theoretical quantiles.''')
    fig = atlas_SOC_val_res['qq_plot_figure']
    if fig_save:
        save_figure(fig, filename_base, '_q-q_plot_SOC_regression_residuals', formats=['pdf', 'tiff'], dpi=600)


    # ##################################################
    # 7) Tests of statistical significance,
    #    test statistics and corresponding p_values
    # ##################################################

    print(f'''\n\n*********\n*********\n7) Tests of statistical significance.''')

    mbe_test_result = bool(atlas_SOC_val_res['t_test_results']['mbe_ttest']['pvalue_two_sided']<0.05)
    if mbe_test_result:
        test_m = "p_value < 0.05, null hypothesis can be rejected, MBE is probably significantly different from zero" 
    else:
        test_m = "p_value > 0.05, null hypothesis cannot not be rejected, MBE is probably not significantly different from zero"
    print(f'''\n\n*********\n\t7.a) MBE t-test, against the null hypothesis that mean error is not significantly different from zero: 
    \n\t     test_statistic = {atlas_SOC_val_res['t_test_results']['mbe_ttest']['statistic']:.2f}
    \n\t     df =  {atlas_SOC_val_res['t_test_results']['mbe_ttest']['df']:.2f}
    \n\t     p_value =  {atlas_SOC_val_res['t_test_results']['mbe_ttest']['pvalue_two_sided']:.5f}
    \n\t     {test_m}
    ''')
    

    mae_test_result = bool(atlas_SOC_val_res['t_test_results']['mae_ttest']['pvalue_one_sided']<0.05)
    if mae_test_result:
        test_m = "p_value < 0.05, null hypothesis can be rejected, MAE is probably smaller than threshold" 
    else:
        test_m = "p_value > 0.05, null hypothesis cannot be rejected, MAE is probably larger than threshold"
    print(f'''\n\n*********\n\t7.b) MAE t-test, against the null hypothesis that MAE >= mae_test_threshold: 
    \n\t     (1/n) * sum( |soc_atlas_i - soc_i| ) >= mae_test_threshold
    \n\n\tSince mean absolute error indicates average bias between observed SOC and historical SoilAtlas 
    \t  values, test threshold is set to SOC value corresponding to 50th percentile of observed SOC values.        
    \n\t    mae_test_threshold = {mae_test_threshold:.2f}
    \n\t     test_statistic = {atlas_SOC_val_res['t_test_results']['mae_ttest']['statistic']:.2f}
    \n\t     df =  {atlas_SOC_val_res['t_test_results']['mae_ttest']['df']:.2f}
    \n\t     p_value =  {atlas_SOC_val_res['t_test_results']['mae_ttest']['pvalue_one_sided']:.5f}
    \n\t     {test_m}
    ''')


    mse_test_result = bool(atlas_SOC_val_res['t_test_results']['mse_ttest']['pvalue_one_sided']<0.05)
    if mse_test_result:
        test_m = "p_value < 0.05, null hypothesis can be rejected, MSE is probably smaller than threshold" 
    else:
        test_m = "p_value > 0.05, null hypothesis cannot be rejected, MSE is probably larger than threshold"
    print(f'''\n\n*********\n\t7.c) MSE t-test, against the null hypothesis that MSE >= mse_test_threshold: 
    \n\t     (1/n) * sum( (soc_atlas_i - soc_i)^2 ) >= mse_test_threshold
    \n\n\tSince mean squared error indicates magnitude of differences between observed SOC and historical SoilAtlas 
    \t  values, test threshold is set to squared value of MAE test threshold defined in 7.b).        
    \n\t    mse_test_threshold = {mae_test_threshold**2:.2f}
    \n\t     test_statistic = {atlas_SOC_val_res['t_test_results']['mse_ttest']['statistic']:.2f}
    \n\t     df =  {atlas_SOC_val_res['t_test_results']['mse_ttest']['df']:.2f}
    \n\t     p_value =  {atlas_SOC_val_res['t_test_results']['mse_ttest']['pvalue_one_sided']:.5f}
    \n\t     {test_m}
    ''')

    sw_test_result = bool(atlas_SOC_val_res['shapiro_pvalue']<0.05)
    if sw_test_result:
        test_m = "p_value < 0.05, null hypothesis can be rejected, regression residuals probably do not follow normal distribution" 
    else:
        test_m = "p_value > 0.05, null hypothesis cannot be rejected, regression residuals probably follow normal distribution"
    print(f'''\n\n*********\n\t7.d) Shapiro-Wilk test against the null hypothesis that SOC regression residuals do not come from a normally distributed population.
    \n\t     Residuals correspond to simple linear regression between historical SOC data extracted from SoilAtlas and observed SOC samples.
    \t     High p value should suggest no strong evidence against the null hypothesis. Test is also directly related to Q-Q plot in 6).       
    \n\t     test_statistic = {atlas_SOC_val_res['shapiro_stat']:.2f}
    \n\t     p_value =  {atlas_SOC_val_res['shapiro_pvalue']:.5f}
    \n\t     {test_m}
    ''')


    # ##################################################
    # 8) Tests of statistical significance,
    #    test statistics and corresponding p_values
    # ##################################################

    r2_test_result = bool(atlas_SOC_val_res['r_squared_p_value']<0.05)
    if r2_test_result:
        test_m = "\np_value < 0.05, null hypothesis can be rejected, simple regression probably explains variance in observed data" 
    else:
        test_m = "p_value > 0.05, null hypothesis cannot be rejected, simple regression probably does not explain variance in observed data"
    print(f'''\n\n*********\n\t8) R^2 test against the null hypothesis that simple linear regression model explains no variance in observed data.
    \n\t     Corresponds to simple linear regression between historical SOC data extracted from SoilAtlas and observed SOC samples.
    \t       Test statistic is approximated using F-test: F = (R²/(k)) / ((1-R²)/(n-k-1)), with k=1 
    \n\t     test_statistic = {atlas_SOC_val_res['r_squared_F_stat']:.2f}
    \n\t     p_value =  {atlas_SOC_val_res['r_squared_p_value']:.5f}
    \n\t     {test_m}
    ''')

