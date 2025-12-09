import json
import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict
from scipy.stats import sem
from utils.constants import (NORM_BASE_SUBMISSION, DATASETS, DIGITS_FOR_VALUES, DIGITS_FOR_ERRORS,
                             DIMENSIONS, COLUMN_ORDER, MODEL_INFO_FILE, RESULTS_DIR)
from utils import compute_tools



def load_results(folder: str = RESULTS_DIR,
                items_to_ignore: list = ["__pycache__", "compiled.pkl", ".DS_Store"]
                ):
    """
    loads results from results folder.

    Args:
        folder: folder containing results
        items_to_ignore: list of items in results folder to ignore
    """
    #read model info
    with open(MODEL_INFO_FILE) as f:
        model_info = json.load(f)
    model_size = model_info["MODEL_SIZE"]
    backbone_names = model_info["BACKBONE_NAMES"]

    #read submission info
    all_submissions = os.listdir(folder)
    for item in items_to_ignore:
        if item in all_submissions: 
            all_submissions.remove(item)

    all_submission_results = {}
    #TODO: add some info to json files and read here also
    all_full_ft_model_names = []
    all_frozen_model_names = []
    all_submission_results["frozen"] = {}
    all_submission_results["full_ft"] = {}
    for submission in all_submissions:        
        combined_results = pd.read_csv(f"{folder}/{submission}/results_and_parameters.csv")
        combined_results = combined_results.drop(["index"], errors='ignore')
        try:
            frozen_or_full_ft = combined_results["frozen_or_full_ft"][0]    
        except KeyError as e:
            raise KeyError(f"{e} {combined_results=}")
        all_submission_results[frozen_or_full_ft][submission] = {}

        combined_results["# params"]  = combined_results.apply(lambda row: model_size[row.backbone], axis=1)
        combined_results["Model"]  = combined_results.apply(lambda row: backbone_names[row.backbone], axis=1)
        combined_results["Config Settings"]  = combined_results.apply(lambda row: get_config_setting_string(row), axis=1)

        #TODO: read json info
        all_backbones = list(set(combined_results["backbone"].tolist()))
        all_submission_results[frozen_or_full_ft][submission]["results"] = combined_results
        all_submission_results[frozen_or_full_ft][submission]["all_backbones"] = all_backbones
        
        config_settings = combined_results[["early_stop_patience", "decoder", "n_trials", "data_percentages", "batch_size_selection"]].iloc[0]
        config_settings = config_settings.replace("early_stopping_50", "50").replace("n_trials_16", "16").replace("data_100_perc", "100") 
        all_submission_results[frozen_or_full_ft][submission]["config_info"] = config_settings
        
        #all_submission_results[submission]["json_info"] = json_info
        if frozen_or_full_ft =="frozen":
            all_frozen_model_names.extend(all_backbones)
        else:
            all_full_ft_model_names.extend(all_backbones)

    all_frozen_model_names = list(set(all_frozen_model_names))
    all_full_ft_model_names = list(set(all_full_ft_model_names))
    all_model_names = {"full_ft": all_full_ft_model_names, "frozen": all_frozen_model_names}
    return all_submission_results, all_model_names, all_submissions



def compute_all_iqms(
                all_submission_results: dict,
                benchmark_name: str,
                dataset_group_keys:list =["backbone", "dataset"],
                overall_group_keys:list = ["backbone"],
                metric:str ="test metric",
                ) -> Dict:
    """
    - reads combined results from repeated seeds for multiple models
    - computes the raw and normalized IQM by dataset for each model by task type
    - computes the raw and normalized overall IQM across multiple datasets in each each task type
    
    Args:
        all_submission_results: dict containing all results
        benchmark_name: name of normalizer file to be used
        dataset_group_keys: grouping for computing dataset IQM
        overall_group_keys: grouping for computing overall IQM
        metric: the column containing scores/values in the combined results tables
    """
    output = {}
    for submission in all_submission_results:
        output[submission] = {}
        print(f'\n\n\n{submission=}')
        submission_backbones = all_submission_results[submission]["all_backbones"]

        #TODO: remove
        partition_name =  "0.10x train" if "data_10_perc" in submission else "1.00x train" 
        submission_results = all_submission_results[submission]["results"]
        if "partition name" not in list(submission_results.columns):
            submission_results["partition name"] = partition_name 
        submission_results["partition name"] = partition_name 

        #get raw values per dataset
        series = submission_results.groupby(dataset_group_keys)[metric].apply(np.mean)
        raw_per_dataset = series.to_frame().reset_index()
        raw_per_dataset = raw_per_dataset.drop(columns=["partition name"], errors='ignore') 
        included_datsets = [d for d in DATASETS if d in set(raw_per_dataset["dataset"])]
        raw_per_dataset_final = pd.DataFrame(columns=["backbone"] + included_datsets)
        
        #get raw errors per dataset
        series = submission_results.groupby(dataset_group_keys)[metric].apply(sem)
        raw_per_dataset_err = series.to_frame().reset_index()
        raw_per_dataset_err = raw_per_dataset_err.drop(columns=["partition name"], errors='ignore') 
        raw_per_dataset_final_err = pd.DataFrame(columns=["backbone"] + included_datsets)
       
        #rearrange
        for backbone in submission_backbones:
            #get values
            data = raw_per_dataset.loc[raw_per_dataset["backbone"] == backbone]
            data = data.drop(columns=["backbone"]).rename(columns={metric: backbone, "dataset": "backbone"})
            data = data.set_index(['backbone']).T.reset_index()
            data = data.rename(columns={"index": "backbone"})
            try:
                data = data.loc[:, ["backbone"] + included_datsets]
            except KeyError as e:
                print(f'{backbone} {e=}')
                continue

            raw_per_dataset_final = data.copy() if len(raw_per_dataset_final.index)==0 else pd.concat([raw_per_dataset_final, data], ignore_index=True)

            #get errors
            data_err = raw_per_dataset_err.loc[raw_per_dataset_err["backbone"] == backbone]
            data_err = data_err.drop(columns=["backbone"]).rename(columns={metric: backbone, "dataset": "backbone"})
            data_err = data_err.set_index(['backbone']).T.reset_index()
            data_err = data_err.rename(columns={"index": "backbone"})
            data_err = data_err.loc[:, ["backbone"] + included_datsets]
            raw_per_dataset_final_err = data_err.copy() if len(raw_per_dataset_final_err.index)==0 else pd.concat([raw_per_dataset_final_err, data_err], ignore_index=True)

        raw_per_dataset_final = raw_per_dataset_final.reset_index(drop=True).rename_axis(mapper=None, axis='columns')
        raw_per_dataset_final_err = raw_per_dataset_final_err.reset_index(drop=True).rename_axis(mapper=None, axis='columns')
        raw_per_dataset_final = raw_per_dataset_final.reindex(columns=["backbone"]+DATASETS, fill_value=np.nan)
        raw_per_dataset_final_err = raw_per_dataset_final_err.reindex(columns=["backbone"]+DATASETS, fill_value=np.nan)

        #normalize results
        normalizer = compute_tools.load_normalizer(benchmark_name=benchmark_name)
        new_metric = normalizer.normalize_data_frame(df=submission_results, metric=metric)
        
        #get normalized values  per dataset
        series = submission_results.groupby(dataset_group_keys)[new_metric].apply(compute_tools.iqm)
        normalized_per_dataset = series.to_frame().reset_index()
        normalized_per_dataset = normalized_per_dataset.drop(columns=["partition name"], errors='ignore') 
        included_datsets = [d for d in DATASETS if d in set(normalized_per_dataset["dataset"])]
        normalized_per_dataset_final = pd.DataFrame(columns=["backbone"] + included_datsets)

        #get normalized errors per dataset
        series = submission_results.groupby(dataset_group_keys)[new_metric].apply(compute_tools.trimmed_sem)
        normalized_per_dataset_err = series.to_frame().reset_index()
        normalized_per_dataset_err = normalized_per_dataset_err.drop(columns=["partition name"], errors='ignore') 
        normalized_per_dataset_final_err = pd.DataFrame(columns=["backbone"] + included_datsets)

        #rearrange
        for backbone in submission_backbones:
            #get values
            data = normalized_per_dataset.loc[normalized_per_dataset["backbone"] == backbone]
            data = data.drop(columns=["backbone"]).rename(columns={new_metric: backbone, "dataset": "backbone"})
            data = data.set_index(['backbone']).T.reset_index()
            data = data.rename(columns={"index": "backbone"})
            try:
                data = data.loc[:, ["backbone"] + included_datsets]
            except KeyError as e:
                print(f'{backbone} {e=}')
                continue
            normalized_per_dataset_final = data.copy() if len(normalized_per_dataset_final.index)==0 else pd.concat([normalized_per_dataset_final, data], ignore_index=True)

            #get errors
            data_err = normalized_per_dataset_err.loc[normalized_per_dataset["backbone"] == backbone]
            data_err = data_err.drop(columns=["backbone"]).rename(columns={new_metric: backbone, "dataset": "backbone"})
            data_err = data_err.set_index(['backbone']).T.reset_index()
            data_err = data_err.rename(columns={"index": "backbone"})
            data_err = data_err.loc[:, ["backbone"] + included_datsets]
            normalized_per_dataset_final_err = data_err.copy() if len(normalized_per_dataset_final_err.index)==0 else pd.concat([normalized_per_dataset_final_err, data_err], ignore_index=True)

        normalized_per_dataset_final = normalized_per_dataset_final.reset_index(drop=True).rename_axis(mapper=None, axis='columns')
        normalized_per_dataset_final_err = normalized_per_dataset_final_err.reset_index(drop=True).rename_axis(mapper=None, axis='columns')
        normalized_per_dataset_final =normalized_per_dataset_final.reindex(columns=["backbone"]+DATASETS, fill_value=np.nan)
        normalized_per_dataset_final_err =normalized_per_dataset_final_err.reindex(columns=["backbone"]+DATASETS, fill_value=np.nan)

        #get normalized values by dimension
        normalized_overall = pd.DataFrame(columns=["backbone"])
        normalized_overall_std_err = pd.DataFrame(columns=["backbone"])
        submission_dimensions = []
        for dimension in DIMENSIONS:
            dimension_data = submission_results.loc[submission_results["dataset"].isin(DIMENSIONS[dimension])].copy()
            dimension_datasets = sorted(set(dimension_data["dataset"]))
            dimension_backbones = sorted(set(dimension_data["backbone"]))
            exclude_backbone = []
            for backbone in dimension_backbones:
                backbone_datasets = dimension_data.loc[dimension_data["backbone"] == backbone]["dataset"].tolist()
                if set(backbone_datasets) != set(dimension_datasets):
                    #if backbone is missing datasets, drop from table
                    exclude_backbone.append(backbone)
                    
            dimension_datasets = [True if d in dimension_datasets else False for d in DIMENSIONS[dimension]]
            # dimension_data = dimension_data[~dimension_data["backbone"].isin(exclude_backbone)]
            if all(dimension_datasets): 
                submission_dimensions.append(dimension)
                
                #get values
                normalized_iqms_dimension = compute_tools.bootstrap_iqm_aggregate(dimension_data, metric= new_metric)
                series = normalized_iqms_dimension.groupby(overall_group_keys)[new_metric].apply(np.mean)
                normalized_iqms_dimension = series.to_frame().reset_index()
                normalized_iqms_dimension = normalized_iqms_dimension.rename(columns={new_metric: dimension})
                normalized_iqms_dimension.loc[normalized_iqms_dimension["backbone"].isin(exclude_backbone), dimension, ] = np.nan

            
                #get errors
                normalized_dimension_std_err = compute_tools.bootstrap_iqm_aggregate(dimension_data, metric=new_metric)
                series = normalized_dimension_std_err.groupby(["backbone"])[new_metric].apply(sem)
                # series = submission_results.loc[submission_results["dataset"].isin(DIMENSIONS[dimension])].copy()
                # series = series[~series["dataset"].isin(exclude_backbone)]
                # series = series.groupby(overall_group_keys)[new_metric].apply(sem)
                normalized_dimension_std_err = series.to_frame().reset_index()  
                normalized_dimension_std_err = normalized_dimension_std_err.drop(columns=["partition name"], errors='ignore') 
                normalized_dimension_std_err = normalized_dimension_std_err.rename(columns={new_metric: dimension})
                normalized_dimension_std_err.loc[normalized_dimension_std_err["backbone"].isin(exclude_backbone), dimension] = np.nan
                # series = dimension_data.groupby(overall_group_keys)[new_metric].apply(sem)
                # normalized_dimension_std_err = series.to_frame().reset_index()
                # normalized_dimension_std_err = normalized_dimension_std_err.rename(columns={new_metric: dimension})
            else:
                normalized_iqms_dimension = pd.DataFrame({
                    "backbone": submission_backbones,
                    dimension: [np.nan] * len(submission_backbones),
                })
                normalized_dimension_std_err = pd.DataFrame({
                    "backbone": submission_backbones,
                    dimension: [np.nan] * len(submission_backbones),
                })
                
            normalized_iqms_dimension.sort_values(by=['backbone'], inplace=True)
            normalized_dimension_std_err.sort_values(by=['backbone'], inplace=True)
            normalized_overall = normalized_iqms_dimension.copy() if len(normalized_overall.index)==0 else normalized_overall.merge(normalized_iqms_dimension, how="left", on="backbone")
            normalized_overall_std_err = normalized_dimension_std_err.copy() if len(normalized_overall_std_err.index)==0 else normalized_overall_std_err.merge(normalized_dimension_std_err, how="left", on="backbone")

        output[submission]["raw_per_dataset"] = raw_per_dataset_final
        output[submission]["normalized_per_dataset"] = normalized_per_dataset_final
        output[submission]["normalized_overall"] = normalized_overall
        output[submission]["raw_per_dataset_err"] = raw_per_dataset_final_err
        output[submission]["normalized_per_dataset_err"] = normalized_per_dataset_final_err
        output[submission]["normalized_overall_err"] = normalized_overall_std_err
        output[submission]["submission_dimensions"] = submission_dimensions
    return output



def format_values(x):
    x = x*100
    x = round(x,1)
    return x


def format_errors(x):
    x = x*100
    x = round(x,1)
    return x


def get_config_setting_string(row) -> str:
    config_settings = f"""
                        Early Stop Patience: {row.early_stop_patience} /
                        Decoder: {row.decoder} /
                        # trials: {row.n_trials} /
                        Data : {row.data_percentages}% /
                        Batch Size Selection: {row.batch_size_selection}
                        """
    config_settings = config_settings.replace("early_stopping_50", "50").replace("n_trials_16", "16").replace("data_100_perc", "100") 
    return config_settings


def get_overall_performance_table(all_submission_results: dict, 
                                all_iqms: dict
                                ) -> Dict:
    """
    create tables for 'Aggregated Performance' page.

    Args:
        all_submission_results: dict containing all results
        all_iqms: dict containing all computed results

    """
    output = {}
    result_type = ["normalized"]
    for value in result_type:
        all_tables = []
        all_tables_err = []
        for submission in all_submission_results:
            #get results
            submission_data = all_iqms[submission][f"{value}_overall"].copy()
            submission_data["Model"] = "-"
            submission_data["# params"] = "-"
            submission_data["submission"] = submission

            submission_data_err = all_iqms[submission][f"{value}_overall_err"].copy()
            submission_data_err["Config Settings"] = "-"
            submission_data_err["Model"] = "-"
            submission_data_err["# params"] = "-"
            submission_data_err["submission"] = submission

            #get parameters
            parameters = all_submission_results[submission]["results"]
            for backbone in all_submission_results[submission]["all_backbones"]:
                submission_data.loc[submission_data["backbone"] == backbone, "Model"] = parameters.loc[parameters["backbone"] == backbone]["Model"].tolist()[0]
                submission_data.loc[submission_data["backbone"] == backbone, "# params"] = parameters.loc[parameters["backbone"] == backbone]["# params"].tolist()[0]

                submission_data_err.loc[submission_data_err["backbone"] == backbone, "Model"] = parameters.loc[parameters["backbone"] == backbone]["Model"].tolist()[0]
                submission_data_err.loc[submission_data_err["backbone"] == backbone, "# params"] = parameters.loc[parameters["backbone"] == backbone]["# params"].tolist()[0]
            all_tables.append(submission_data)
            all_tables_err.append(submission_data_err)
            print(f'\n\n\n {submission} {value} {submission_data[["Core", "Detection (Object/Instance)", "Model", "submission"]].head(50)=}')

        all_tables = pd.concat(all_tables)
        all_tables_err = pd.concat(all_tables_err)
        all_tables.loc[:, COLUMN_ORDER[value]["overall_table"]] = all_tables[COLUMN_ORDER[value]["overall_table"]].round(DIGITS_FOR_VALUES).apply(lambda series: series.apply(format_values))    
        all_tables_err.loc[:, COLUMN_ORDER[value]["overall_table"]]= all_tables_err[COLUMN_ORDER[value]["overall_table"]].round(DIGITS_FOR_ERRORS).apply(lambda series: series.apply(format_errors))  
        all_tables = all_tables[COLUMN_ORDER["all_tables"] + COLUMN_ORDER[value]["overall_table"]]
        all_tables_err = all_tables_err[COLUMN_ORDER["all_tables"] + COLUMN_ORDER[value]["overall_table"]]
        for col in COLUMN_ORDER[value]["overall_table"]:
            new_column = f"{col}"
            all_tables = all_tables.rename(columns={col: new_column})
            all_tables_err = all_tables_err.rename(columns={col: new_column})
        output[value] = all_tables
        output[f"{value}_err"] = all_tables_err
    return output

        
def get_performance_by_dimension_table(all_submission_results: dict, 
                                       all_iqms: dict
                                        ) -> Dict:
    """
    create tables for 'Capabilities' page.

    Args:
        all_submission_results: dict containing all results
        all_iqms: dict containing all computed results

    """
    output = {}
    result_type = ["normalized"]
    for value in result_type:
        all_tables = {}
        all_tables_err = {}
        for dimension in DIMENSIONS:
            dimension_tables = []
            dimension_tables_err = []
            for submission in all_submission_results:
                #get results
                submission_data = all_iqms[submission][f"{value}_per_dataset"][DIMENSIONS[dimension]+["backbone"]].copy()
                dimension_results = all_iqms[submission][f"{value}_overall"][[dimension]+["backbone"]].copy()
                submission_data = submission_data.merge(dimension_results, how="left", on="backbone")
                submission_data["Model"] = "-"
                submission_data["# params"] = "-"
                submission_data["submission"] = submission

                submission_data_err = all_iqms[submission][f"{value}_per_dataset_err"][DIMENSIONS[dimension]+["backbone"]].copy()
                dimension_results_err = all_iqms[submission][f"{value}_overall_err"][[dimension]+["backbone"]].copy()
                submission_data_err = submission_data_err.merge(dimension_results_err, how="left", on="backbone")
                submission_data_err["Model"] = "-"
                submission_data_err["# params"] = "-"
                submission_data_err["submission"] = submission

                #get parameters
                parameters = all_submission_results[submission]["results"]
                for backbone in all_submission_results[submission]["all_backbones"]:
                    submission_data.loc[submission_data["backbone"] == backbone, "Model"] = parameters.loc[parameters["backbone"] == backbone]["Model"].tolist()[0]
                    submission_data.loc[submission_data["backbone"] == backbone, "# params"] = parameters.loc[parameters["backbone"] == backbone]["# params"].tolist()[0]

                    submission_data_err.loc[submission_data_err["backbone"] == backbone, "Model"] = parameters.loc[parameters["backbone"] == backbone]["Model"].tolist()[0]
                    submission_data_err.loc[submission_data_err["backbone"] == backbone, "# params"] = parameters.loc[parameters["backbone"] == backbone]["# params"].tolist()[0]
                dimension_tables.append(submission_data)
                dimension_tables_err.append(submission_data_err)
                # print(f'\n\n\n {submission} {dimension} {submission_data[[dimension, "Model", "submission"]].head(50)=}')

            dimension_tables = pd.concat(dimension_tables)
            dimension_tables.loc[:, DIMENSIONS[dimension]] = dimension_tables[DIMENSIONS[dimension]].round(DIGITS_FOR_VALUES).apply(lambda series: series.apply(format_values))        
            dimension_tables.loc[:, dimension] = dimension_tables[dimension].round(DIGITS_FOR_VALUES).apply(format_values)
            dimension_tables = dimension_tables[COLUMN_ORDER["all_tables"] + [dimension] + COLUMN_ORDER[value]["dimension_tables"]  + DIMENSIONS[dimension]]
            new_column = f"{dimension}"
            dimension_tables = dimension_tables.rename(columns={dimension: new_column})
            all_tables[dimension] = dimension_tables

            dimension_tables_err = pd.concat(dimension_tables_err)
            dimension_tables_err.loc[:, DIMENSIONS[dimension]] = dimension_tables_err[DIMENSIONS[dimension]].round(DIGITS_FOR_ERRORS).apply(lambda series: series.apply(format_errors))        
            dimension_tables_err.loc[:, dimension] = dimension_tables_err[dimension].round(DIGITS_FOR_ERRORS).apply(format_errors)
            dimension_tables_err = dimension_tables_err[COLUMN_ORDER["all_tables"] + [dimension] + COLUMN_ORDER[value]["dimension_tables"]  + DIMENSIONS[dimension]]
            dimension_tables_err  = dimension_tables_err.rename(columns={dimension: new_column})
            all_tables_err[f"{dimension}_err"] = dimension_tables_err

        output[value] = all_tables
        output[f"{value}_err"] = all_tables_err
    return output  


def get_datasets_tables(all_submission_results: dict, 
                        all_iqms: dict
                        ) -> Dict:
    """
    creates tables for dataset tab.

    Args:
        all_submission_results: dict containing all results
        all_iqms: dict containing all computed results
    """
    output = {}
    result_type = ["normalized","raw"]
    for value in result_type:
        all_tables = {}
        all_tables_err = {}
        for dataset in DATASETS:
            dataset_tables = []
            dataset_tables_err = []
            for submission in all_submission_results:
                #get results
                submission_data = all_iqms[submission][f"{value}_per_dataset"][["backbone", dataset]].copy()
                submission_data["Model"] = "-"
                submission_data["# params"] = "-"
                submission_data["submission"] = submission

                submission_data_err = all_iqms[submission][f"{value}_per_dataset_err"][["backbone", dataset]].copy()
                submission_data_err["Model"] = "-"
                submission_data_err["# params"] = "-"
                submission_data_err["submission"] = submission

                #get parameters
                parameters = all_submission_results[submission]["results"]
                new_column = "IQM" if value == "normalized" else "Mean"

                for backbone in all_submission_results[submission]["all_backbones"]:
                    submission_data.loc[submission_data["backbone"] == backbone, "Model"] = parameters.loc[parameters["backbone"] == backbone]["Model"].tolist()[0]
                    submission_data.loc[submission_data["backbone"] == backbone, "# params"] = parameters.loc[parameters["backbone"] == backbone]["# params"].tolist()[0]
                    submission_data = submission_data.rename(columns={dataset: new_column})

                    submission_data_err.loc[submission_data_err["backbone"] == backbone, "Model"] = parameters.loc[parameters["backbone"] == backbone]["Model"].tolist()[0]
                    submission_data_err.loc[submission_data_err["backbone"] == backbone, "# params"] = parameters.loc[parameters["backbone"] == backbone]["# params"].tolist()[0]
                    submission_data_err = submission_data_err.rename(columns={dataset: new_column})
                    #TODO: add columns
                dataset_tables.append(submission_data)
                dataset_tables_err.append(submission_data_err)
            column = "IQM" if value == "normalized" else "Mean"
            dataset_tables = pd.concat(dataset_tables)
            dataset_tables.loc[:, column] = dataset_tables[column].round(DIGITS_FOR_VALUES).apply(format_values)
            all_tables[dataset] = dataset_tables[COLUMN_ORDER["all_tables"] + COLUMN_ORDER[value]["dataset_tables"]]

            dataset_tables_err = pd.concat(dataset_tables_err)
            dataset_tables_err.loc[:, column] = dataset_tables_err[column].round(DIGITS_FOR_ERRORS).apply(format_errors)
            all_tables_err[dataset] = dataset_tables_err[COLUMN_ORDER["all_tables"] + COLUMN_ORDER[value]["dataset_tables"]]

        output[value] = all_tables
        output[f"{value}_err"] = all_tables_err
    return output   


def get_submission_tables(all_submission_results: dict):
    output = {}
    frozen_or_full_ft = ["frozen" ,"full_ft"]
    config_info = []
    for method in frozen_or_full_ft:
        for sub in all_submission_results[method]:
            config = all_submission_results[method][sub]["config_info"]
            config = config.to_frame().T
            config["submission"] = sub
            config["backbone method"] = method
            config_info.append(config)
    output = pd.concat(config_info)
    output = output[COLUMN_ORDER["submission_info"]]
    return output   




if __name__ == "__main__":
    #load results
    all_submission_results, all_model_names, all_submissions = load_results(folder=RESULTS_DIR)

    #COMBINED NORM
    norm_base_results= [] 
    for method in NORM_BASE_SUBMISSION:
        for sub in NORM_BASE_SUBMISSION[method]:
            norm_base_results.append(all_submission_results[method][sub]["results"].copy())
    norm_base_results = pd.concat(norm_base_results)
    benchmark_name = "leaderboard_combined"  
    compute_tools.make_normalizer(norm_base_results.reset_index(), 
                                    metrics=("test metric",), 
                                    benchmark_name=benchmark_name)

    overall_performance_tables = {}
    performance_by_dimension_tables = {}
    datasets_tables = {}
    for method in ["full_ft","frozen"]:
        method_iqms = compute_all_iqms(
                                all_submission_results =  all_submission_results[method],
                                benchmark_name = benchmark_name,
                                )
    
        #create tables to be rendered
        overall_performance_tables[method] = get_overall_performance_table(all_submission_results=all_submission_results[method],
                                                                    all_iqms=method_iqms)
        performance_by_dimension_tables[method] = get_performance_by_dimension_table(all_submission_results=all_submission_results[method],
                                                                            all_iqms=method_iqms)
        datasets_tables[method] = get_datasets_tables(all_submission_results=all_submission_results[method],
                                                all_iqms=method_iqms)

    submission_info_table = get_submission_tables(all_submission_results=all_submission_results)
        
    compiled_results = {
                        "overall_performance_tables": overall_performance_tables,
                        "performance_by_dimension_tables": performance_by_dimension_tables,
                        "datasets_tables": datasets_tables,
                        "submission_info_table": submission_info_table
                        }

    with open(f'{RESULTS_DIR}/compiled.pkl', 'wb') as handle:
        pickle.dump(compiled_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
