import pytest
import numpy as np
import os
from qtcorgi.experiments.utils import merge_intermittent_graphs

data_location = os.path.join(__file__[:13],"data","utils")


class TestMergeIntermittentGraphs:
    def test_merging(self):
        file = "graphs_info_n7_d3"
        def get_intermittent_file(ind):
            intermittent_loc = os.path.join(data_location, ".intermittent")
            return os.path.join(intermittent_loc, f"{file}_graph{ind}.npy")
        merged_values = merge_intermittent_graphs(get_intermittent_file,20)
        known_values = np.load(os.path.join(data_location, f"{file}.npy"))
        assert merged_values == known_values # TODO check if correct