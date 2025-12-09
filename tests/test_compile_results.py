import pytest
from pathlib import Path
import pandas as pd
from typing import List
from utils.compile_results import load_results, compute_all_iqms, get_overall_performance_table, get_performance_by_dimension_table, get_datasets_tables

root = Path(__file__).parent.resolve()
root = str(root)

SUBMISSION_NAMES = ['test_submission_1']
MODEL_NAMES = ["satlas_resnet50_sentinel2_si_ms_satlas"]
TEST_BENCHMARK = "leaderboard_combined"  
METHOD = "full_ft"


@pytest.fixture
def expected_output():
    expected_results = pd.read_pickle(f"{root}/resources/outputs/test_output.pkl")
    return expected_results


class TestLoading:
    all_submission_results, all_model_names, all_submissions = load_results(folder=f"{root}/resources/inputs")    
    def test_submission_results(self):
        for _, value in self.all_submission_results[METHOD].items():
            assert "results" in value
            assert isinstance(value["results"], pd.DataFrame)

            columns_to_be_added = {"# params", "Model", "Config Settings"}
            existing_columns = set(value["results"].columns)
            assert columns_to_be_added.issubset(existing_columns)

    def test_model_names(self):
        assert sorted(self.all_model_names[METHOD]) == sorted(MODEL_NAMES)

    def test_submission_names(self):
        assert sorted(self.all_submissions) == SUBMISSION_NAMES
 


class TestComputeResults:
    #compute output
    all_submission_results, all_model_names, all_submissions = load_results(folder=f"{root}/resources/inputs")    
    all_iqms = compute_all_iqms(all_submission_results =  all_submission_results[METHOD],
                                benchmark_name = TEST_BENCHMARK
                                )
    overall_performance_tables = get_overall_performance_table(all_submission_results=all_submission_results[METHOD], 
                                                                all_iqms=all_iqms)
    performance_by_dimension_tables = get_performance_by_dimension_table(all_submission_results=all_submission_results[METHOD], 
                                                                         all_iqms=all_iqms)
    datasets_tables = get_datasets_tables(all_submission_results=all_submission_results[METHOD], 
                                            all_iqms=all_iqms)
    
    def test_compute_all_iqms(self, expected_output):
        assert sorted(self.all_iqms.keys()) == sorted(self.all_submission_results[METHOD].keys())
        assert "overall_performance_tables" in expected_output

        for submission, submission_value in self.all_iqms.items():
            assert sorted(submission_value.keys()) == sorted(expected_output["all_iqms"][submission].keys())

            for table_name, value in submission_value.items():
                if "submission_dimensions" != table_name:
                    assert isinstance(value, pd.DataFrame)
                else:
                    assert isinstance(value, List)
                #assert table.equals(expected_output["all_iqms"][submission][table_name])


