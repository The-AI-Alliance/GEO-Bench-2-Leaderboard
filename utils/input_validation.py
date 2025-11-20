import pandas as pd
import json
import os
import numpy as np
import uuid
from utils.constants import (NEW_SUBMISSION_FOLDER, CSV_FILE, JSON_FILE, DIMENSIONS, 
                        NEW_SUBMISSION_COLUMN_INFO, NEW_SUBMISSION_COLUMN_NAMES,
                        JSON_FORMAT, MODEL_INFO_FILE, RESULTS_DIR, REQUIRED_SEEDS)


def check_correct_file_type(folder_contents) -> bool:
    """ 
        checks that folder has 2 items: a csv file and a json file
    """
    contains_correct_files = (len(folder_contents) == 2) and (CSV_FILE in folder_contents) and (JSON_FILE in folder_contents)
    if not contains_correct_files:
        print("\nInput Validation Error: Please check that the {NEW_SUBMISSION_FOLDER} contains the files: \
              {CSV_FILE} and {JSON_FILE}")
        return False
    return True


def check_csv_columns_datatypes() -> bool:
    """ 
        checks that csv file has only required columns and columns have correct data types
    """
    #check for correct columns
    csv_data = pd.read_csv(f"{NEW_SUBMISSION_FOLDER}/{CSV_FILE}")
    submitted_csv_column_names = set(csv_data.columns)
    expected_column_names = set(NEW_SUBMISSION_COLUMN_NAMES)
    for item in expected_column_names:
        if item not in submitted_csv_column_names:
            print(f"The following column is missing: {item}")
    correct_columns = expected_column_names.issubset(submitted_csv_column_names)
    if not correct_columns:
        print(f"\nInput Validation Error: Please ensure that the csv file contains the following columns: {NEW_SUBMISSION_COLUMN_NAMES}")
    
    #check for correct dtype
    correct_dtypes = []
    for col in NEW_SUBMISSION_COLUMN_INFO["string_cols"]:
        if col in csv_data.columns:
            print(f"{col} is string/object:{pd.api.types.is_object_dtype(csv_data[col])}")
            correct_dtypes.append(pd.api.types.is_object_dtype(csv_data[col]))
    for col in NEW_SUBMISSION_COLUMN_INFO["integer_cols"]:
        if col in csv_data.columns:
            print(f"{col} is numeric: {pd.api.types.is_numeric_dtype(csv_data[col])}")
            correct_dtypes.append(pd.api.types.is_numeric_dtype(csv_data[col]))
    for col in NEW_SUBMISSION_COLUMN_INFO["float_cols"]:
        if col in csv_data.columns:
            print(f"{col} is numeric: {pd.api.types.is_numeric_dtype(csv_data[col])}")
            correct_dtypes.append(pd.api.types.is_numeric_dtype(csv_data[col]))
    correct_dtypes = all(correct_dtypes)
    if not correct_dtypes:
        print(f"\nInput Validation Error: Please ensure that the csv columns have the correct datatypes as follows: \n\
                string/object type columns: {NEW_SUBMISSION_COLUMN_INFO['string_cols']} \n\
                numeric/integer type columns: {NEW_SUBMISSION_COLUMN_INFO['integer_cols']}\
                                                {NEW_SUBMISSION_COLUMN_INFO['float_cols']}")
    return correct_columns, correct_dtypes 


def check_correct_entries_per_dataset(required_seeds: int = REQUIRED_SEEDS) -> bool:
    """
        checks for correct number of runs per backbone/dataset combination
        checks for required number of unique seeds
    """
    csv_data = pd.read_csv(f"{NEW_SUBMISSION_FOLDER}/{CSV_FILE}")
    count_values = csv_data.groupby(["backbone", "dataset"]).count()
    count_values = list(set(count_values["test metric"].tolist()))
    correct_num_values = (len(count_values) == 1) and (count_values[0] == required_seeds)
    if not correct_num_values:
        print(f"\nInput Validation Error: Please ensure that each backbone/dataset combination has {required_seeds} entries")

    count_seeds = csv_data.groupby(["backbone", "dataset"]).nunique()
    count_seeds = list(set(count_seeds["Seed"].tolist()))
    correct_num_seeds = (len(count_seeds) == 1) and (count_seeds[0] == required_seeds)
    if not correct_num_seeds:
        print(f"\nInput Validation Warning: Please ensure that each backbone/dataset combination has {required_seeds} unique seeds")
    return correct_num_values, correct_num_seeds


def check_json_keys() -> bool:
    """ 
        checks json file has required keys and subkeys, 
        check json file values have correct data type
    """
    with open(f"{NEW_SUBMISSION_FOLDER}/{JSON_FILE}") as f:
        json_submission_data = json.load(f)
    #TBD: check json file nested values have correct data type
    all_required_keys = []
    for key, value in JSON_FORMAT.items():
        if (key in json_submission_data) and (type(value) == type(json_submission_data[key])): 
            all_required_keys.append(True)
        else:
            all_required_keys.append(False)
    all_required_keys = all(all_required_keys)
    if not all_required_keys:
        print(f"\nInput Validation Error: Please ensure that json file has the correct keys and datatypes")
    
    return all_required_keys


def check_has_atleast_one_dimension() -> bool:
    """
        check that submission contains datasets required for at least one submission
    """
    csv_data = pd.read_csv(f"{NEW_SUBMISSION_FOLDER}/{CSV_FILE}")
    submitted_csv_datasets = set(csv_data["dataset"].tolist())
    contains_atleast_one_dimension = []
    for dimension, datasets in DIMENSIONS.items():
        datasets = set(datasets)
        contains_atleast_one_dimension.append(datasets.issubset(submitted_csv_datasets))
    contains_atleast_one_dimension = any(contains_atleast_one_dimension)
    if not contains_atleast_one_dimension:
        print("\nInput Validation Error: Please check that the submission contains all datasets for one or more dimensions")
        print(f'currently submitted datasets are: {submitted_csv_datasets}')
    return contains_atleast_one_dimension


def check_has_frozen_or_full_ft() -> bool:
    """
        check that submission has correct values in frozen_or_full_ft column
    """
    csv_data = pd.read_csv(f"{NEW_SUBMISSION_FOLDER}/{CSV_FILE}")
    frozen_or_full_ft = set(csv_data["frozen_or_full_ft"].tolist())
    correct_values = True
    for item in frozen_or_full_ft:
        if not ((item == "frozen") or (item == "full_ft")):
            correct_values = False
    if not correct_values:
        print("\nInput Validation Error: Please check that the frozen_or_full_ft column contains only 'frozen' or 'full_ft'")
        print(f'currently submitted values are: {frozen_or_full_ft}')
    return correct_values

def update_new_backbones_and_models():
    """
        checks if backbone exists in model_info.json (used to display results)
        if not, information on the new model is added to the json file
    """
    with open(f"{NEW_SUBMISSION_FOLDER}/{JSON_FILE}") as f:
        json_submission_data = json.load(f)

    #read model info
    with open(MODEL_INFO_FILE) as f:
        existing_model_info = json.load(f)
    for item in json_submission_data["New model info"]:
        submitted_backbone = item["unique_backbone_key"]
        if submitted_backbone not in existing_model_info["BACKBONE_NAMES"]:
            existing_model_info["BACKBONE_NAMES"][submitted_backbone] = item["model_display_name"]
            existing_model_info["MODEL_SIZE"][submitted_backbone] = item["model_size"]
    
    #save new information
    with open(MODEL_INFO_FILE, 'w') as fp:
        json.dump(existing_model_info, fp)


def validate_new_submission() -> bool:
    """
    
    """
    #get folder contents
    if not os.path.exists(NEW_SUBMISSION_FOLDER): return
    folder_contents = os.listdir(NEW_SUBMISSION_FOLDER)
    items_to_ignore = ['.DS_Store']
    for item in items_to_ignore:
        if item in folder_contents: folder_contents.remove(item)
    if len(folder_contents) == 0:
        print("no new submissions")
        return
    
    #check all conditions
    correct_file_type = check_correct_file_type(folder_contents)
    correct_columns, correct_dtypes = check_csv_columns_datatypes()
    correct_num_values, correct_num_seeds = check_correct_entries_per_dataset()
    correct_json_keys = check_json_keys()
    contains_atleast_one_dimension = check_has_atleast_one_dimension()
    correct_frozen_or_full_ft = check_has_frozen_or_full_ft()
    all_checks_passed = all([correct_file_type, correct_columns, correct_dtypes,
                            correct_json_keys, correct_num_values, #correct_num_seeds,
                            contains_atleast_one_dimension, correct_frozen_or_full_ft])
    
    if all_checks_passed:
        submission_id = uuid.uuid4()
        os.makedirs(f"{RESULTS_DIR}/{submission_id}")

        #copy only required keys in json file to new submission folder
        with open(f"{NEW_SUBMISSION_FOLDER}/{JSON_FILE}") as f:
            json_submission_data = json.load(f)
        new_dict = {}
        for key, value in JSON_FORMAT.items():
            if value == "TBD": continue
            new_dict[key] = json_submission_data[key]
        with open(f"{RESULTS_DIR}/{submission_id}/{JSON_FILE}", 'w') as fp:
            json.dump(new_dict, fp)

        #copy only required columns in csv file to new submission folder
        csv_data = pd.read_csv(f"{NEW_SUBMISSION_FOLDER}/{CSV_FILE}")
        csv_data = csv_data[NEW_SUBMISSION_COLUMN_NAMES]
        csv_data.to_csv(f"{RESULTS_DIR}/{submission_id}/{CSV_FILE}", index=False)

        #add any new model info to model_info.json
        update_new_backbones_and_models()

        #reset NEW_SUBMISSION_FOLDER
        os.system(f"rm -r {NEW_SUBMISSION_FOLDER}/")
        os.makedirs(NEW_SUBMISSION_FOLDER)
        return
    else:
        print("\nThe new sumbission has not been formatted correctly. Please fix the errors above")
        raise ValueError

    




    
if __name__ == "__main__":
    validate_new_submission()
