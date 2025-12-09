from pathlib import Path

NORM_BASE_SUBMISSION = {
        "full_ft": ["114be1f0-5a41-43a5-b4e6-7fb683bc01ec", "2ce4a907-7ae3-45d3-a07a-558f8d0d758b"],# "b5e5e59d-044b-4848-8323-aceb8c535ed9"],
        # "frozen": ["80100fc6-dc1a-4514-b946-341157eaf816"],
}
DIGITS_FOR_VALUES = 3
DIGITS_FOR_ERRORS = 6
REQUIRED_SEEDS = 5


DIMENSIONS = {
            "Multi-Spectral-Dependent": ["benv2", "biomassters", "pastis", "so2sat", "cloudsen12", "spacenet2", "burn_scars", "fotw",],#
            "Multi-Temporal": ['kuro_siwo','pastis', 'biomassters', 'dynamic_earthnet', ],
            "Pixel-wise": ['kuro_siwo', 'pastis', 'burn_scars', 'spacenet2', 'cloudsen12',  'caffe', 'flair2','dynamic_earthnet','biomassters',  "spacenet7", "fotw",],
            "Classification": ['so2sat', 'forestnet', 'benv2', 'treesatai'],
            "Detection (Object/Instance)": ["substation", "everwatch", "nzcattle", "pastis_r",],
            "Under 10m Resolution": ["spacenet2","treesatai", "flair2",'dynamic_earthnet',  "spacenet7"],
            "10m and Above Resolution": ["biomassters", "so2sat", "kuro_siwo", "cloudsen12", "pastis", "benv2", "forestnet", "burn_scars", "caffe", "fotw",],#
            "RGB/NIR": ["flair2",  "treesatai", 'dynamic_earthnet', "spacenet7", "fotw",], #
            "Core": ['kuro_siwo', 'pastis', 'burn_scars', 'cloudsen12',  'flair2', "spacenet7", 'benv2', 'treesatai', 'biomassters', "fotw","substation", "everwatch", ],
            }


DIMENSION_INFO = {
            "Multi-Spectral-Dependent": "datasets that have a statistically significant increase in perfromance when mutlispectral bands are used",
            "Multi-Temporal": "datasets with more than 1 timestamps used as an input",
            "Pixel-wise": "datasets for pixel-wise segmentation and regression",
            "Classification": "single-label and multi-label classification datasets",
            "Detection (Object/Instance)":"datasets for instance segmentation and object detection",
            "Under 10m Resolution": "datasets with resolution <= 1 metre",
            "10m and Above Resolution": "datasets with 10 metres =< resolution <= 30 metres",
            "RGB/NIR": "datasets using  Red, Green, Blue, and NIR bands",
            "Core": "subset with datasets from each dimension",
            }


DATASETS = [
            'biomassters', 'so2sat', 'forestnet', 'benv2', 'treesatai',
            'kuro_siwo', 'dynamic_earthnet', 'pastis', 'burn_scars', 'spacenet2',
            'cloudsen12', 'fotw', 'caffe', 'flair2',  "spacenet7",
            "substation", "everwatch", "nzcattle", "pastis_r",
            ]



DATASET_INFO = {
                "Dataset": [item.replace("_", " ").title() for item in DATASETS],
                "Description": [
                    "regression dataset for Above Ground Biomass (AGB) prediction", #biomassters
                    "multi-class classifictaion dataset for Global Local Climate Zones", #so2sat
                    "multi-class classifictaion dataset for deforestation drivers", #forestnet
                    "multi-label classifictaion dataset for land cover",#benv2
                    "multi-label classifictaion dataset for tree species",#treesatai
                    "SAR semantic segmentation dataset for rapid flood mapping",#kuro_siwo_mean
                    "semantic segmentation dataset for land use/land cover",#dynamic_earthnet
                    "semantic segmentation dataset for agricultural parcels",#pastis
                    "semantic segmentation dataset for burn scars",#burn_scars
                    "semantic segmentation dataset for building detection",#spacenet2
                    "semantic segmentation dataset for cloud and cloud shadow detection",#cloudsen12
                    "semantic/instance segmentation dataset for agricultural fields ",#fotw
                    "semantic/instance segmentation dataset for glacier calving front extraction",#caffe
                    "semantic segmentation dataset for land use/land cover",#flair2
                    "semantic segmentation dataset for building detection",#spacenet7
                    "instance segmentation dataset for substations",#substation
                    "object detection dataset for bird species",#everwatch
                    "object detection dataset for cattle",#nzcattle
                    "instance segmentation dataset for crop type mapping",#pastis_r
                    ],
                "Dimensions": [", ".join([dim for dim, data_list in DIMENSIONS.items() if dataset in data_list]) for dataset in DATASETS]
                }


COLUMN_ORDER = {
                "raw": {
                    "dataset_tables": ['Mean'],
                    "dimension_tables": []
                },
                "normalized": {
                    "overall_table": [
                        "Core",  "Multi-Spectral-Dependent",  "Multi-Temporal", "Pixel-wise", "Classification",  "Detection (Object/Instance)", 
                        "Under 10m Resolution", "10m and Above Resolution",
                        "RGB/NIR",
                        ], 
                    "dataset_tables": ['IQM'] ,
                    "dimension_tables": []

                },
                "all_tables": ['Model', '# params', 'submission'],
                "submission_info": ["submission", "backbone method", "decoder", "n_trials", "early_stop_patience", "data_percentages", "batch_size_selection" ],
                }

root = Path(__file__).parent.resolve()
root = "/".join(str(root).split("/")[:-1])
RESULTS_DIR = f"{root}/results"
MODEL_INFO_FILE = f"{root}/utils/model_info.json"
NORMALIZER_DIR = f"{root}/utils/normalizer"


#for validation of new submissions
NEW_SUBMISSION_FOLDER = f"{root}/new_submission"
CSV_FILE = "results_and_parameters.csv"
JSON_FILE = "additional_info.json"
NEW_SUBMISSION_COLUMN_INFO = {
                                "string_cols": ['dataset', 'Metric', 'experiment_name', 'partition name', 'backbone', 'decoder','batch_size_selection', 'frozen_or_full_ft'],
                                "integer_cols": ['early_stop_patience', 'n_trials', 'Seed', 'data_percentages', 'batch_size'],
                                "float_cols": ['weight_decay', 'lr', 'test metric', ]
                            }
NEW_SUBMISSION_COLUMN_NAMES = []
for key, value in NEW_SUBMISSION_COLUMN_INFO.items():
    NEW_SUBMISSION_COLUMN_NAMES.extend(value)
BIOMASSTERS_STD = 0.2537951171398163
                                
JSON_FORMAT = {
                "Paper Link": "N/A",
                "Code Repository Link ": "N/A",
                "License": "N/A",
                "Number of HPO trials": "16",
                "Additional information about submission": "N/A",
                "Comments on new models in submission": "N/A",
                "New model info": 
                    [
                        {
                            "model_display_name": "TBD",
                            "model_size": "TBD", 
                            "unique_backbone_key": "TBD"
                        }
                    ]
                }   