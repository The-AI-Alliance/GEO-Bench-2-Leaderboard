import pytest
from pathlib import Path
import pandas as pd
import pickle
from utils.compile_results import load_results, compute_all_iqms, get_overall_performance_table, get_performance_by_dimension_table, get_datasets_tables
from utils.constants import NORM_BASE_SUBMISSION

root = Path(__file__).parent.resolve()
root = str(root)

SUBMISSION_NAMES = ['test_submission_1']
MODEL_NAMES = ["ssl4eos12_resnet50_sentinel2_all_decur","ssl4eos12_resnet50_sentinel2_all_dino"]


@pytest.fixture
def expected_output():
    with open(f"{root}/resources/outputs/test_output.pkl", 'rb') as handle:
        expected_results = pickle.load(handle)
    return expected_results


class TestLoading:
    all_submission_results, all_model_names, all_submissions = load_results(folder=f"{root}/resources/inputs")    
    def test_submission_results(self):
        for _, value in self.all_submission_results.items():
            assert "results" in value
            assert isinstance(value["results"], pd.DataFrame)

            columns_to_be_added = {"# params", "Model", "Config Settings"}
            existing_columns = set(value["results"].columns)
            assert columns_to_be_added.issubset(existing_columns)

    def test_model_names(self):
        assert sorted(self.all_model_names) == sorted(MODEL_NAMES)

    def test_submission_names(self):
        assert sorted(self.all_submissions) == SUBMISSION_NAMES
 


class TestComputeResults:
    #compute output
    all_submission_results, all_model_names, all_submissions = load_results(folder=f"{root}/resources/inputs")    
    benchmark_name = f"leaderboard_{NORM_BASE_SUBMISSION}_main"  
    all_iqms = compute_all_iqms(all_submission_results =  all_submission_results,
                                benchmark_name = benchmark_name
                                )
    overall_performance_tables = get_overall_performance_table(all_submission_results=all_submission_results, 
                                                                all_iqms=all_iqms)
    performance_by_dimension_tables = get_performance_by_dimension_table(all_submission_results=all_submission_results, 
                                                                         all_iqms=all_iqms)
    datasets_tables = get_datasets_tables(all_submission_results=all_submission_results, 
                                            all_iqms=all_iqms)
    
    def test_compute_all_iqms(self, expected_output):
        assert sorted(self.all_iqms.keys()) == sorted(self.all_submission_results.keys())
        assert "overall_performance_tables" in expected_output

        for submission, submission_value in self.all_iqms.items():
            assert sorted(submission_value.keys()) == sorted(expected_output["all_iqms"][submission].keys())

            for table_name, table in submission_value.items():
                assert isinstance(table, pd.DataFrame)
                #assert table.equals(expected_output["all_iqms"][submission][table_name])


    
    def test_raw_values(self):
        assert "raw" in self.overall_performance_tables
        #dataset values
        #overall values
        #dimension values

    def test_normalized_values(self):
        assert "normalized" in self.overall_performance_tables
        #dataset values
        #overall values
        #dimension values
        pass