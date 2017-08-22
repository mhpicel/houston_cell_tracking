from datetime import datetime, timedelta
from scipy import ndimage
import multiprocessing as mp
import numpy as np
import os
import gc
import pandas as pd
import pyart
from glob import glob
from joblib import Parallel, delayed
#import dask
#from dask import delayed

#from tracking.core import cell_tracking as ct


def get_file_tree(start_dir, pattern):
    """
    Make a list of all files matching pattern
    above start_dir
    Parameters
    ----------
    start_dir : string
        base_directory
    pattern : string
        pattern to match. Use * for wildcard
    Returns
    -------
    files : list
        list of strings
    """

    files = []

    for dir, _, _ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir, pattern)))
    return files


def parse_datetime(filepath):
    file = filepath.split('/')[-1]
    date = '%s-%s-%s' % (file[10:14], file[14:16], file[16:18])
    time = '%s:%s:%s' % (file[19:21], file[21:23], file[23:25])
    return datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')


def unparse_datetime(dt, base):
    date = dt.strftime('%Y%m%d')
    file = '/KHGX_grid_' + dt.strftime('%Y%m%d.%H%M%S')
    ext = '.nc'
    return base + date + file + ext


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
    try:
        data = grid.fields['reflectivity']['data']
    except:
        return file_name, False
    frame = get_filtered_frame(data, min_size, thresh)
    gc.collect()
    print(file_name)
    if np.max(frame) > 0:
        return file_name, True
    else:
        return file_name, False


class Grid_iter:
    def __init__(self, dts, dir_base):
        self.filenames = [unparse_datetime(dt, dir_base) for dt in dts]
        self.i = -1
        self.n = len(self.filenames)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n - 1:
            self.i += 1
            return pyart.io.read_grid(self.filenames[self.i])
        else:
            raise StopIteration

#%%
if __name__ == '__main__':

#    file_dir = '/home/mhpicel/blues_earthscience/radar/houston/data/20160720/'
#    file_dir = '/home/mhpicel/NASA/khgx_data/'
    file_dir = '/lcrc/group/earthscience/radar/houston/data/'

#    out_dir = '/home/mhpicel/NASA/test_output/'
#    out_dir = '/home/picel/khgx/testrun/'
#    out_dir = '/home/picel/khgx/may20_bigcells/'
#    out_dir = '/home/picel/khgx/july2015/'
    out_dir = 'home/picel/khgx/testrun_2015/'

    pattern = 'KHGX_grid_2016*'
#    pattern = 'KHGX_grid_201607*'
#    pattern = 'KHGX_grid_20150520*'
#    month_pattern = 'KHGX_grid_201507'
#    days = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16']

    files = get_file_tree(file_dir, pattern)
    files.sort()

#    files = []
#    for day in days:
#        files.extend(get_file_tree(file_dir, month_pattern + day + '*'))
#    files.sort()

    print('number of files:', len(files))

#    min_size = ct.MIN_SIZE
#    thresh = ct.FIELD_THRESH
    min_size = 32
    thresh = 32

    start_time = datetime.now()

    scans_unfiltered = Parallel(n_jobs=mp.cpu_count())(
        delayed(detect_cells)(file, min_size, thresh) for file in files
        )

#    scans = []
#    for file in files:
#        detection = delayed(detect_cells)(file, min_size, thresh)
#        scans.append(detection)
#    scans_computed = dask.compute(*scans)

    print('Cell Detection Complete')
    scans = [scan[0] for scan in scans_unfiltered if scan[1] is True]
    scans.sort()
    print('Parsing Times')
    times = [parse_datetime(scan) for scan in scans]
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

    coverage = pd.DataFrame(data=scans_unfiltered, columns=['file', 'cells'])
    coverage.sort_values(by='file', inplace=True)
    coverage['time'] = coverage.file.apply(parse_datetime)
    coverage.to_csv('coverage.csv', index=False)

    metadata = open('meta.txt', 'w')
    metadata.write('number of files: ' + str(len(files)))
    metadata.write('total time: ' + str(datetime.now() - start_time))
    metadata.close()
    print('total time:', datetime.now() - start_time)
