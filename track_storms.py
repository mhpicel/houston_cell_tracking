from tracking.core import cell_tracking as ct
import pandas as pd
#from joblib import Parallel, delayed
import dask
from dask import bag as db
#from dask import compute, delayed
import datetime
from find_storms import Grid_iter
import multiprocessing as mp


def parse_date_string(date_string):
    return datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')


def tracks_from_iter(s_name, s_iter):
    tobj = ct.Cell_tracks()
    tobj.get_tracks(s_iter)
    return s_name, tobj


if __name__ == '__main__':
#    file_dir = '/lcrc/group/earthscience/radar/houston/data/'
#    out_dir = '/home/picel/khgx/july2015_kdp/'
#    storms_path = '/home/picel/khgx/july2015_kdp/storms_kdp.csv'

#    file_dir = '/home/mhpicel/blues_earthscience/radar/houston/data/'
#    out_dir = '/home/mhpicel/NASA/july2015/kdp_dataframes/'
#    storms_path = '/home/mhpicel/blues_home/khgx/july2015_kdp/storms_kdp.csv'

    start_time = datetime.datetime.now()

    storms = pd.read_csv(storms_path)

    ct.FIELD_THRESH = 32
    ct.MIN_SIZE = 32

    min_storm_length = 5
    long_enough = storms['storm_id'].value_counts() >= min_storm_length
    long_storm_ix = long_enough.keys()[long_enough]
    long_storms = storms.set_index('storm_id').loc[long_storm_ix.sort_values()]
    long_storm_dts = long_storms['begin'].apply(parse_date_string)
    storm_iters = [
        (storm_id, Grid_iter(long_storm_dts.loc[storm_id], file_dir))
        for storm_id
        in long_storm_dts.keys().unique()
        ]

    print('tracking', len(storm_iters), 'storms')

#    storm_tracks = Parallel(n_jobs=mp.cpu_count())(
#        delayed(tracks_from_iter)(name, storm) for name, storm in storm_iters
#        )

#    storm_graph = [delayed(tracks_from_iter)(name, storm)
#                   for name, storm in storm_iters]

#    storm_tracks = compute(*storm_graph, get=dask.multiprocessing.get)

    storm_bag = db.from_sequenc(storm_iters, npartitions=36)
    storm_bag = storm_bag.map(lambda s_iter:
                              tracks_from_iter(s_iter[0], s_iter[1]))

    storm_tracks = storm_bag.compute()

    for name, tracks in storm_tracks:
        tracks.tracks.to_csv(out_dir + 'storm_' + str(name).zfill(4))

    time_elapsed = datetime.datetime.now() - start_time
    print('time elapsed:', time_elapsed)
    meta = open(out_dir + 'tracking_meta.txt', 'w')
    meta.write('Number of Storms: ' + len(storm_tracks) + '\n')
    meta.write('Time Elapsed: ' + time_elapsed + '\n')
    meta.close()
