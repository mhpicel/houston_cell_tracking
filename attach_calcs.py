import pandas as pd
#import dask.bag as db
import dask
import datetime.datetime
from dask import delayed, compute
from find_storms import get_file_tree, parse_date_string
from marcus_calcs import attach_marcus_stats, filename_from_dt


def parse_storm_number(file):
    storm_name = file.split('/')[-1]
    storm_number = storm_name[-4:]
    return storm_number


def prep_storm_attach_stats(file, grid_dir):
    tracks = pd.read_csv(file)
    tracks.set_index(['scan', 'uid'], inplace=True)
    tracks['time'] = tracks['time'].apply(parse_date_string)
    tracks['file'] = tracks['time'].apply(lambda dt:
                                          filename_from_dt(dt, grid_dir))
    return parse_storm_number(file), attach_marcus_stats(tracks)


if __name__ == '__main__':
    track_dir = ''
    grid_dir = '/lcrc/group/earthscience/radar/houston/data/'
    out_dir = ''
    pattern = "storm_*"
    files = get_file_tree(track_dir, pattern)
    files.sort()

#    track_bag = db.from_sequence(files, npartitions=36)
#    track_bag = track_bag.map(lambda file:
#                              prep_storm_attach_stats(file, grid_dir))
#    out_tracks = track_bag.compute()
    start = datetime.now()

    track_graph = [delayed(prep_storm_attach_stats)(file, grid_dir)
                   for file in files.sort()]
    out_tracks = compute(*track_graph, get=dask.multiprocessing.get)

    for number, track in out_tracks:
        track.tracks.to_csv(out_dir + 'stormcalcs_' + number + '.csv')

    time_elapsed = datetime.now() - start
    print('nfiles: ', len(files))
    print(time_elapsed)
