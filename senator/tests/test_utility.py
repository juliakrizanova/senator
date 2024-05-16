from senator.utility import get_utility, get_utility_matrix, load_data, parse_data, FILE_PATH

def test_file_exists():
    assert len(load_data(FILE_PATH).shape) == 2