from datetime import datetime, timedelta
from scipy import ndimage
import multiprocessing as mp
import numpy as np
import gc
import pandas as pd
import pyart
from glob import glob
from joblib import Parallel, delayed


def parse_grid_datetime(grid_obj):
    """ Obtains datetime object from pyart grid_object. """
    dt_string = grid_obj.time['units'].split(' ')[-1]
    date = dt_string[:10]
    time = dt_string[11:19]
    dt = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
    return dt


def clear_small_echoes(label_image, min_size):
    """Takes in binary image and clears objects less than min_size."""
    flat_image = pd.Series(label_image.flatten())
    flat_image = flat_image[flat_image > 0]
    size_table = flat_image.value_counts(sort=False)
    small_objects = size_table.keys()[size_table < min_size]

    for obj in small_objects:
        label_image[label_image == obj] = 0
    label_image = ndimage.label(label_image)
    return label_image[0]


def get_vert_projection(grid, thresh=40):
    """Returns binary vertical projection from grid."""
    return np.any(grid > thresh, axis=0)


def get_filtered_frame(grid, min_size, thresh):
    """Returns a labeled frame from gridded radar data. Smaller objects are
    removed and the rest are labeled."""
    echo_height = get_vert_projection(grid, thresh)
    labeled_echo = ndimage.label(echo_height)[0]
    frame = clear_small_echoes(labeled_echo, min_size)
    return frame


def detect_cells(file_name, min_size, thresh):
    grid = pyart.io.read_grid(file_name)
    datetime = parse_grid_datetime(grid)
    try:
        data = grid.fields['reflectivity']['data']
    except (RuntimeError, OSError) as e:
        return file_name, datetime, False
    frame = get_filtered_frame(data, min_size, thresh)
    gc.collect()
    print(file_name)
    if np.max(frame) > 0:
        return file_name, datetime, True
    else:
        return file_name, datetime, False

#%%
if __name__ == '__main__':
    file_dir = '/lcrc/group/earthscience/radar/houston/data/'
    out_dir = 'home/picel/khgx/testrun_2015/'
    pattern = 'KHGX_grid_2016*'

    files = glob((file_dir + '/**/' + pattern + '*'), recursive=True)
    files.sort()

    print('number of files:', len(files))

    min_size = 24
    thresh = 32

    start_time = datetime.now()

    scans_unfiltered = Parallel(n_jobs=mp.cpu_count())(
        delayed(detect_cells)(file, min_size, thresh) for file in files
        )

    print('Cell Detection Complete')
    print('Parsing Times')
    times = [scan[1] for scan in scans_unfiltered if scan[2] is True]
    inventory = pd.DataFrame({'begin': times[:-1], 'end': times[1:]})
    inventory['interval'] = inventory['end'] - inventory['begin']

    max_interval = timedelta(minutes=30)
    print('Building Storms Dataframe')
    storms = inventory[inventory.interval < max_interval].copy()
    storms.end = storms.end.shift(1)
    storms.drop(storms.index[0], inplace=True)
    storms['new'] = storms.begin != storms.end
    storms['storm_id'] = storms.new.astype('i').cumsum()
    storms.to_csv(out_dir + 'storms.csv', index=False)

    coverage = pd.DataFrame(data=scans_unfiltered,
                            columns=['file', 'time', 'cells'])
    coverage.sort_values(by='file', inplace=True)
    coverage.to_csv('coverage.csv', index=False)

    metadata = open('meta.txt', 'w')
    metadata.write('number of files: ' + str(len(files)))
    metadata.write('total time: ' + str(datetime.now() - start_time))
    metadata.close()
    print('total time:', datetime.now() - start_time)
