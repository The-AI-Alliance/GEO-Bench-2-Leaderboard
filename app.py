import os
import json
import re
import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import quote
from pathlib import Path
import re
import html
import pickle
from typing import Dict, Any, Tuple
from scipy.stats import sem
from utils.constants import (DATASETS, DIGITS_FOR_VALUES, DIGITS_FOR_ERRORS,
                               DATASET_INFO, DIMENSIONS, RESULTS_DIR,
                               DIMENSION_INFO)


def sanitize_model_name(model_name):
    # Only allow alphanumeric chars, hyphen, underscore
    if model_name.startswith('.'):
        raise ValueError("model name cannot start with a dot")
    
    if not re.match("^[a-zA-Z0-9-_][a-zA-Z0-9-_.]*$", model_name):
        raise ValueError("Invalid model name format")
    return model_name


def safe_path_join(*parts):
    # Ensure we stay within results directory
    base = Path("results").resolve()
    try:
        path = base.joinpath(*parts).resolve()
        if not str(path).startswith(str(base)):
            raise ValueError("Path traversal detected")
        return path
    except Exception:
        raise ValueError("Invalid path")


def sanitize_column_name(col: str) -> str:
    """Sanitize column names for HTML display"""
    col= str(col)
    is_result_column = [True if item in col else False for item in ["IQM", "Mean"]]    
    col = col.replace("_", " ") if any(is_result_column) else col.replace("_", " ").title()
    return html.escape(col)



def sanitize_cell_value(value: Any) -> str:
    """Sanitize cell values for HTML display"""
    if value == "nan ± nan":
        value = "NA"
    if isinstance(value, (int, float)):
        output = str(value)
    else:
        output = html.escape(str(value))
    return output


def heat_rgb01(t: float, max_val: int) -> Tuple[int, int, int]:
    """Map t in [0,1] to white → IBM blue (#0F62FE)."""
    min_val = 0
    t = max(min_val, min(1, (t/max_val)))
    # t = (t - min_val)/max_val
    # White RGB
    wr, wg, wb = 255, 255, 255
    # IBM blue RGB
    br, bg, bb = 15, 98, 254
    r = int(round(wr + (br - wr) * t))
    g = int(round(wg + (bg - wg) * t))
    b = int(round(wb + (bb - wb) * t))
    return r, g, b

def rgb_to_css(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"rgb({r}, {g}, {b})"

def readable_text_color(rgb: Tuple[int, int, int]) -> str:
    """Pick text color (#222 or white) based on perceived luminance."""
    r, g, b = rgb
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#222222" if y > 160 else "#ffffff"


def create_html_results_table(df, df_err, rank_table, display_rank=False):
    html = '''
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        th {
            font-weight: bold;
        }
        .table-container {
            padding-bottom: 20px;
        }
    </style>
    '''
    html += '<div class="table-container">'
    html += '<table>'
    html += '<thead><tr>'
    
    rank_index = list(range(1, df.shape[0]+1))
    df.insert(loc=0, column='rank', value=rank_index)

    columns = list(df.columns)
    columns = [column for column in df.columns if column != "index"]
    for column in columns:
        html += f'<th>{sanitize_column_name(column)}</th>'
    html += '</tr></thead>'
    html += '<tbody>'

    for (_, row), (_, row_err), (_, rank_row) in zip(df.iterrows(), df_err.iterrows(), rank_table.iterrows()):
        html += '<tr>'
        for col in columns:
            #if column == "index": continue
            if col == "Model":
                html += f'<td>{row[col]}</td>'
            else:
                if col in row_err:
                    if row[col] != row_err[col]:
                        rgb = heat_rgb01(rank_row[col], rank_index[-1])
                        bg = rgb_to_css(rgb)
                        fg = readable_text_color(rgb)
                        if display_rank:
                            display_val = " - " if np.isnan(row[col]) else int(rank_row[col])
                            new_val = f'<td style="background-color:{bg};color:{fg}">{sanitize_cell_value(display_val)}</td>'
                        else:
                            new_val = f'<td style="background-color:{bg};color:{fg}">{sanitize_cell_value(row[col])} ± {sanitize_cell_value(row_err[col])} </td>'
                        html += new_val
                    else:
                        html += f'<td>{sanitize_cell_value(row[col])}</td>'
                else:
                    html += f'<td>{sanitize_cell_value(row[col])}</td>'
                    
        html += '</tr>'
    html += '</tbody></table>'
    html += '</div>'
    return html

def create_html_table_info(df):
    #create html table
    html = '''
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        th {
            font-weight: bold;
        }
        .table-container {
            padding-bottom: 20px;
        }
    </style>
    '''
    html += '<div class="table-container">'
    html += '<table>'
    html += '<thead><tr>'
    for column in df.columns:
        html += f'<th>{sanitize_column_name(column)}</th>'
    html += '</tr></thead>'
    html += '<tbody>'
    
    for (_, row) in df.iterrows():
        html += '<tr>'
        for column in df.columns:
            if column == "Citation":
                html += f'<td>{row[column]}</td>'
            else:
                html += f'<td>{sanitize_cell_value(row[column])}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    html += '</div>'
    return html
    

def check_sanity(model_name):
    try:
        safe_model = sanitize_model_name(model_name)
        for benchmark in DATASETS:
            file_path = safe_path_join(safe_model, f"{benchmark.lower()}.json")
            if not file_path.is_file():
                continue
            original_count = 0
            with open(file_path) as f:
                results = json.load(f)
                for result in results:
                    if result["original_or_reproduced"] == "Original":
                        original_count += 1
            if original_count != 1:
                return False
        return True
    except ValueError:
        return False

                
def make_hyperlink_datasets(url: str ,
                            url_name: str,
                            root: str = "") -> str:
    try:
        if len(url) == 0:
            return url_name
        full_url = f"{root}{url}"
        return f'<a href="{html.escape(full_url)}" target="_blank">{html.escape(url_name)}</a>'
    except ValueError:
        return ""
                

        
def filter_with_user_selections(unique_key: str,
                                iqm_column_name: str,
                                table = pd.DataFrame,
                                table_err = pd.DataFrame
                                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    table.reset_index(inplace=True)
    table_err.reset_index(inplace=True)
    #filter best results per model if selected
    view_best_per_model = st.radio(
                                    "Select all results or best results",
                                    ["best results per model", "all results"],
                                    index=0,
                                    key=unique_key,
                                    horizontal=True
                                )
    if view_best_per_model == "best results per model":
        table[iqm_column_name] = pd.to_numeric(table[iqm_column_name])
        # table = table.loc[table.groupby('Model')[iqm_column_name].transform('idxmax'),:]
        imax = table.groupby('Model')[iqm_column_name].idxmax(skipna=True)
        imax =imax.to_frame().reset_index()
        imax = [int(table[table["Model"] == row["Model"]].index[0]) if np.isnan(row[iqm_column_name]) else row[iqm_column_name] for i, row in imax.iterrows()]
        table = table.loc[imax]
        table = table.drop_duplicates(['Model'])

    #filter by search bars
    col1, col2  = st.columns(2)
    with col1:
        search_models_query = st.text_input(f"Search by model", "", key=f"search_{unique_key}_models")
    with col2:
        search_submission_query = st.text_input(f"Search by submission", "", key=f"search_{unique_key}_submission")
    # with col3:
    #     search_settings_query = st.text_input(f"Search by settings", "", key=f"search_{unique_key}_settings")
    if search_models_query:
        table = table[table['Model'].str.contains(search_models_query, case=False)]
    if search_submission_query:
        table = table[table['submission'].str.contains(search_submission_query, case=False)]
    # if search_settings_query:
    #     table = table[table['Config Settings'].str.contains(search_settings_query, case=False)]

    # Sort values
    table = table.sort_values(by=iqm_column_name, ascending=False)
    table_err = table_err.loc[table.index]
    table = table.drop(["index"], errors='ignore')
    table_err = table_err.drop(["index"], errors='ignore')
    return table, table_err



def get_rank(df):
    columns_to_rank = DATASETS + list(DIMENSIONS.keys())
    df_rank = df.copy()
    for col in df.columns:
        if col in columns_to_rank:
            df_rank[col] = df[col].rank(method='max', na_option='bottom', ascending=False)
    return df_rank


def create_overall_performance_tab(
        overall_performance_tables, 
        iqm_column_name='Core'
        ):
    # Main Leaderboard tab   
    st.header("Capabilities Ranking")

    #show full finetuning or frozen results if selected
    view_frozen_or_full_ft = st.radio(
                                    "Select full finetuning or frozen values",
                                    ["fully finetuned backbone", "frozen backbone"],
                                    index=0,
                                    key="overall_ft_or_frozen",
                                    horizontal=True
                                )

    # overall_performance_tables = overall_performance_tables["full_ft"].copy()
    if view_frozen_or_full_ft == "fully finetuned backbone":
        overall_performance_tables = overall_performance_tables["full_ft"].copy()
    else:
        overall_performance_tables = overall_performance_tables["frozen"].copy()
    overall_table = overall_performance_tables["normalized"].copy()
    overall_table_err = overall_performance_tables["normalized_err"].copy()

    # filter with user selections
    overall_table, overall_table_err =  filter_with_user_selections(unique_key="overall_all_or_best",
                                                                    iqm_column_name = iqm_column_name,
                                                                    table = overall_table, 
                                                                    table_err = overall_table_err
                                                                    )
    if not overall_table.empty:
        overall_table["submission"] = overall_table["submission"].str.split('-', expand = True)[0]
        overall_table_err["submission"] = overall_table_err["submission"].str.split('-', expand = True)[0]


    #convert to rank
    rank_table = get_rank(overall_table)
    
    # Export the DataFrame to CSV
    if st.button("Export to CSV", key=f"overall_performance_export_main"):
        csv_data = overall_table.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"overall_performance_leaderboard.csv",
            key="download-csv",
            help="Click to download the CSV file",
        )
    st.markdown("Lower rank is better (1 = best). Colors are scaled per column. ")

    html_table = create_html_results_table(overall_table, overall_table_err, rank_table, display_rank=True)
    st.markdown(html_table, unsafe_allow_html=True)

   

def create_dimension_performance_tab(
                        performance_by_dimension_tables
                        ):
    # Dimension  tab   
    st.header("Performance By Capability")
    
    #show full finetuning or frozen results if selected
    view_frozen_or_full_ft = st.radio(
                                    "Select full finetuning or frozen values",
                                    ["fully finetuned backbone", "frozen backbone"],
                                    index=0,
                                    key="dimension_ft_or_frozen",
                                    horizontal=True
                                )
    if view_frozen_or_full_ft == "fully finetuned backbone":
        performance_by_dimension_tables = performance_by_dimension_tables["full_ft"].copy()
    else:
        performance_by_dimension_tables = performance_by_dimension_tables["frozen"].copy()

    #add drop down
    dimension_drop_down = st.selectbox('Select dimension to view',
                                        ([f"{key} - {value}" for key, value in DIMENSION_INFO.items()]))
    dimension_drop_down = dimension_drop_down.split(" - ")[0]

    dimension_table = performance_by_dimension_tables["normalized"][dimension_drop_down].copy()
    dimension_table_err = performance_by_dimension_tables["normalized_err"][f"{dimension_drop_down}_err"].copy()
    iqm_column_name = f'{dimension_drop_down}'

    # filter with search bars
    dimension_table, dimension_table_err  = filter_with_user_selections(unique_key = "dimension_all_or_best",
                                                    iqm_column_name = iqm_column_name,
                                                    table = dimension_table,
                                                    table_err = dimension_table_err)
    if not dimension_table.empty:
        dimension_table["submission"] = dimension_table["submission"].str.split('-', expand = True)[0]
        dimension_table_err["submission"] = dimension_table_err["submission"].str.split('-', expand = True)[0]

    #convert to rank
    rank_table = get_rank(dimension_table)
    
    #performance_by_dimension_tables[dimension_drop_down]['Model'] = performance_by_dimension_tables[dimension_drop_down]['Model'].apply(make_hyperlink)
    html_table = create_html_results_table(dimension_table, dimension_table_err, rank_table, display_rank=False)
    st.markdown(html_table, unsafe_allow_html=True)


def create_datasets_tabs(all_datasets_tables: dict
                            ):
    datasets_tabs = st.tabs([dataset.replace("_", " ") for dataset in DATASETS]) 
    for i, dataset in enumerate(DATASETS):
        with datasets_tabs[i]:   
            dataset_name = dataset.replace("_", " ").title()
            dataset_desc = DATASET_INFO["Description"][DATASET_INFO["Dataset"].index(dataset_name)]
            st.header(dataset.replace("_", " ").title())
            st.markdown(dataset_desc)

            #show full finetuning or frozen results if selected
            view_frozen_or_full_ft = st.radio(
                                            "Select full finetuning or frozen values",
                                            ["fully finetuned backbone", "frozen backbone"],
                                            index=0,
                                            key=f"{dataset_name}_ft_or_frozen",
                                            horizontal=True
                                        )
            if view_frozen_or_full_ft == "fully finetuned backbone":
                datasets_tables = all_datasets_tables["full_ft"].copy()
            else:
                datasets_tables = all_datasets_tables["frozen"].copy()

            #show raw or normalized results if selected
            view_raw_or_normalized_dataset = st.radio(
                                            "Select raw or normalized values",
                                            ["normalized values (with IQM)", "raw values (with Mean)"],
                                            index=0,
                                            key=f"{dataset}_raw_or_normalized",
                                            horizontal=True
                                        )
            dataset_table = datasets_tables["normalized"][dataset].copy()
            dataset_table_err = datasets_tables["normalized_err"][dataset].copy()
            # iqm_column_name = "IQM"
            if view_raw_or_normalized_dataset == "normalized values (with IQM)":
                dataset_table = datasets_tables["normalized"][dataset].copy()
                dataset_table_err = datasets_tables["normalized_err"][dataset].copy()
                iqm_column_name = "IQM"
            else:
                dataset_table = datasets_tables["raw"][dataset].copy()
                dataset_table_err = datasets_tables["raw_err"][dataset].copy()
                iqm_column_name = "Mean"
            
            # filter with search bars
            dataset_table, dataset_table_err = filter_with_user_selections(unique_key = dataset,
                                                        iqm_column_name = iqm_column_name,
                                                        table = dataset_table,
                                                        table_err = dataset_table_err
                                                        )
            if not dataset_table.empty:
                dataset_table["submission"] = dataset_table["submission"].str.split('-', expand = True)[0]
                dataset_table_err["submission"] = dataset_table_err["submission"].str.split('-', expand = True)[0]

            #convert to rank
            rank_table = get_rank(dataset_table)

            #create html table
            html_table = create_html_results_table(dataset_table, dataset_table_err, rank_table, display_rank=False)
            st.markdown(html_table, unsafe_allow_html=True)



def create_info_tab(submission_info_table):
    # tabs = st.tabs(["Dataset Info", "Capability Info", "Submission Info"]) 
    tabs = st.tabs([ "Capability Info"]) 

    # with tabs[0]:
    #     st.header("Dataset Info")
        
    #     dataset_table = pd.DataFrame(DATASET_INFO)
    #     citation_hyperlinks = [make_hyperlink_datasets(url = row.Hyperlinks,
    #                             url_name = row.Citation) for _, row in dataset_table.iterrows()]
    #     dataset_table.drop(columns=['Hyperlinks', 'Citation'], inplace = True)
    #     dataset_table["Citation"] = citation_hyperlinks
    #     dataset_table = create_html_table_info(dataset_table)
    #     st.markdown(dataset_table, unsafe_allow_html=True)

    with tabs[0]:
        st.header("Capability Info")
        dims = []
        datasets = []
        details = []
        for dimension, info in DIMENSION_INFO.items():
            dims.append(dimension)
            datasets.append(", ".join(DIMENSIONS[dimension]))
            details.append(info)
        dim_table = pd.DataFrame({
                            "Capability": dims,
                            "Details": details,
                            "Datasets": datasets,
                            })
        dim_table = create_html_table_info(dim_table)
        st.markdown(dim_table, unsafe_allow_html=True)

    # with tabs[2]:
    #     st.header("Submission Info")
    #     dim_table = create_html_table_info(submission_info_table)
    #     st.markdown(dim_table, unsafe_allow_html=True)

    



def main():
    st.set_page_config(page_title="GeoBench Leaderboard", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <head>
            <meta http-equiv="Content-Security-Policy" 
                content="default-src 'self' https://huggingface.co;
                        script-src 'self' 'unsafe-inline';
                        style-src 'self' 'unsafe-inline';
                        img-src 'self' data: https:;
                        frame-ancestors 'none';">
            <meta http-equiv="X-Frame-Options" content="DENY">
            <meta http-equiv="X-Content-Type-Options" content="nosniff">
            <meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
        </head>
    """, unsafe_allow_html=True)

    compiled_results = pd.read_pickle(f'{RESULTS_DIR}/compiled.pkl')
    overall_performance_tables = compiled_results["overall_performance_tables"] 
    performance_by_dimension_tables = compiled_results["performance_by_dimension_tables"] 
    datasets_tables = compiled_results["datasets_tables"] 
    submission_info_table = compiled_results["submission_info_table"] 
    del compiled_results
    
    #create header
    st.title("🏆 GEO-Bench Leaderboard")
    st.markdown("Benchmarking Geospatial Foundation Models")
    # content = create_yall()
    tabs = st.tabs(["🏆 Main Leaderboard", "Capabilities", "Datasets", "Info", "📝 How to Submit"])

    with tabs[0]:
        create_overall_performance_tab(overall_performance_tables=overall_performance_tables)

    with tabs[1]:
        create_dimension_performance_tab(performance_by_dimension_tables=performance_by_dimension_tables)
    
    with tabs[2]:
        # Datasets tabs               
        #create individual dataset pages
        create_datasets_tabs(all_datasets_tables=datasets_tables)

    with tabs[3]:
        # Info tab       
        create_info_tab(submission_info_table)

    with tabs[-1]:
        #About page
        st.header("How to Submit")
        with open("utils/about_page.txt") as f:
            about_page = f.read()
        st.markdown(about_page)
    comment = """ 

    with tabs[2]:
        # Models tab       
        st.markdown("Models used for benchmarking")
        model_tabs = st.tabs(all_model_names) 
        #create individual benchmark pages
        #create_models_tabs(all_submission_results=all_submission_results,
        #                    model_tabs=model_tabs,
        #                    all_model_names=all_model_names
        #                    )
    with tabs[3]:
        # Submissions tab       
        st.markdown("Experiments submitted to benchmark benchmarking")
        submissions_tabs = st.tabs(all_submissions) 
        #create individual benchmark pages
        #create_submissions_tabs(all_submission_results=all_submission_results,
        #                    model_tabs=submissions_tabs,
        #                    all_submissions=all_submissions
        #                    )
     
    """
                
        
if __name__ == "__main__":
    main()
