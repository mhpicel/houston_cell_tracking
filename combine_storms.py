import pandas as pd
from find_storms import get_file_tree


def add_storm_id(filename):
    storm_id = filename[-8:-4]
    tracks = pd.read_csv(filename)
    tracks['storm_id'] = [storm_id]*len(tracks)
    return tracks


if __name__ == '__main__':
    track_dir = ''
    out_dir=''
    pattern = 'stormcalcs_*'
    files = get_file_tree(track_dir, pattern)
    files.sort()
    storms = [add_storm_id(file) for file in files]
    all_storms = pd.concat(storms)
    all_storms.set_index(['storm_id', 'scan', 'uid'], inplace=True)
    all_storms.to_csv('combined_storms.csv')
