import pandas as pd
from dask import dataframe as dd
import numpy as np
import pyart
from datetime import datetime


def get_grid_size(grid_obj):
    z_size = grid_obj.z['data'][1] - grid_obj.z['data'][0]
    y_size = grid_obj.y['data'][1] - grid_obj.y['data'][0]
    x_size = grid_obj.x['data'][1] - grid_obj.x['data'][0]
    return np.array([z_size, y_size, x_size])


def get_grid_z(grid_size, alt_meters):
    return np.int(np.round(alt_meters/grid_size[0]))


def parse_date_string(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')


def filename_from_dt(dt, base):
    date = dt.strftime('%Y%m%d')
    file_name = '/KHGX_grid_' + dt.strftime('%Y%m%d.%H%M%S')
    ext = '.nc'
    return base + date + file_name + ext


def haversine(lat1, lon1, lat2, lon2):
    R = 6372800.  # Earth radius in kilometers

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    return R * c

#%%
def preprocess_data(data, pars, rho=None, zhh=None):
    data = np.ma.masked_values(data, pars['fill_value'])
    if rho is not None:
        data = np.ma.masked_where(rho < pars['rho_thresh'], data)
        data = np.ma.masked_where(zhh < pars['dbz_thresh'], data)
    return data


def get_neighborhood(dist_from_cent,
                     kdp_proc, kdp_int, pars, kdp_thresh=False):

    circle = dist_from_cent < pars['radius']
    circle = np.tile(circle, (kdp_proc.shape[0], 1, 1))
    layers = np.zeros_like(kdp_proc.data)
    layers[pars['zmelt']:pars['zmelt']+pars['nlayers']+1, :, :] = 1
    neighborhood = np.logical_and(circle, layers.astype('bool'))

    if kdp_thresh:
        kdp_95 = np.percentile(kdp_proc[neighborhood], 95.)
        filtered_circle = np.logical_and(circle, kdp_int > kdp_95)
        neighborhood = np.locical_and(filtered_circle, layers.astype('bool'))

    return neighborhood


def cell_calcs(lat, lon, kdp_proc, zdr_proc, zhh_proc,
               kdp_pei, zdr_pei, zhh_pei, kdp_int, pars):

    grid_ll = pars['grid_ll']
    dist_from_cent = haversine(lat, lon, grid_ll[1], grid_ll[0])

    neighborhood = get_neighborhood(dist_from_cent, kdp_proc, kdp_int, pars)
    kdp_pct = np.percentile(kdp_proc[neighborhood], pars['pctile'])
    zdr_pct = np.percentile(zdr_proc[neighborhood], pars['pctile'])
    zhh_pct = np.percentile(zhh_proc[neighborhood], pars['pctile'])

    if pars['use_kdp_thresh']:
        neighborhood = get_neighborhood(dist_from_cent, kdp_proc,
                                        kdp_int, kdp_thresh=True)

    if not np.any(neighborhood):
        print('isempty')
        kdp_pet = 0
        zdr_pet = 0
        zhh_pet = 0
    else:
        kdp_pet = kdp_pei[neighborhood].sum()/neighborhood.sum()
        zdr_pet = zdr_pei[neighborhood].sum()/neighborhood.sum()
        zhh_pet = zhh_pei[neighborhood].sum()/neighborhood.sum()

    return pd.Series({'kdp_pct': kdp_pct,
                      'zdr_pct': zdr_pct,
                      'zhh_pct': zhh_pct,
                      'kdp_pet': kdp_pet,
                      'zdr_pet': zdr_pet,
                      'zhh_pet': zhh_pet})


def marcus_stats(scan_group, pars):
    file_name = scan_group['file'].iloc[0]
    grid = pyart.io.read_grid(file_name)

    rho = grid.fields['cross_correlation_ratio']['data']
    zhh = grid.fields['reflectivity']['data']
    kdp = grid.fields['specific_differential_phase']['data']
    zdr = grid.fields['differential_reflectivity']['data']

    kdp_proc = preprocess_data(kdp, pars, rho=rho, zhh=zhh)
    zdr_proc = preprocess_data(zdr, pars, rho=rho, zhh=zhh)
    zhh_proc = preprocess_data(zhh, pars)

    kdp_int = np.sum(
        kdp_proc[pars['zmelt']:pars['zmelt']+pars['nlayers']+1, :, :],
        axis=0
    )

    z_column = grid.z['data'][:, np.newaxis, np.newaxis]
    kdp_pei = kdp_proc*z_column
    zdr_pei = zdr_proc*z_column
    zhh_pei = zhh_proc*z_column

    def get_cell_calcs(cell_row):
        """Wraps cell_calcs to be passed to apply call."""
        return cell_calcs(cell_row['lat'], cell_row['lon'],
                          kdp_proc, zdr_proc, zhh_proc,
                          kdp_pei, zdr_pei, zhh_pei,
                          kdp_int, pars)

    output = scan_group.apply(get_cell_calcs, axis=1)
    marcus_frame = pd.DataFrame(output)
    print(marcus_frame)

    del grid, rho, zhh, kdp, zdr, kdp_proc, zdr_proc, zhh_proc, kdp_int
    del kdp_pei, zdr_pei, zhh_pei
    return marcus_frame


def attach_marcus_stats(tracks_frame, zmelt_meters=4000, layer_size=1500,
                        radius=5000, pctile=98., use_kdp_thresh=False,
                        rho_thresh=0.8, dbz_thresh=15, fill_value=-9999.):
    # setup
    file_name = tracks_frame['file'].iloc[0]
    grid = pyart.io.read_grid(file_name)
    zmelt, nlayers, grid_ll = get_gpars(grid, zmelt_meters, layer_size)

    pars = {'zmelt': zmelt,
            'nlayers': nlayers,
            'grid_ll': grid_ll,
            'radius': radius,
            'pctile': pctile,
            'use_kdp_thresh': use_kdp_thresh,
            'rho_thresh': rho_thresh,
            'dbz_thresh': dbz_thresh,
            'fill_value': fill_value}

    stats = tracks_frame.groupby(level='scan').apply(lambda scan:
                                                     marcus_stats(scan, pars))
    return tracks_frame.join(stats)


def get_gpars(grid, zmelt_meters, layer_size):
    grid_size = get_grid_size(grid)
    zmelt = get_grid_z(grid_size, zmelt_meters)
    nlayers = get_grid_z(grid_size, layer_size)
    grid_ll = grid.get_point_longitude_latitude()
    return zmelt, nlayers, grid_ll


#%%
if __name__ =='__main__':
    grid_dir = '/home/mhpicel/blues_earthscience/radar/houston/data/'

    test_tracks_path = '/home/mhpicel/NASA/july2015/refl_long_storm.csv'
    test_tracks = pd.read_csv(test_tracks_path)
    test_tracks.set_index(['scan', 'uid'], inplace=True)
    test_tracks = test_tracks.loc[:10]
    test_tracks['time'] = test_tracks['time'].apply(parse_date_string)

    def get_filenames(dt):
        return filename_from_dt(dt, grid_dir)
    test_tracks['file'] = test_tracks['time'].apply(get_filenames)

    start = datetime.now()
    out_tracks = attach_marcus_stats(test_tracks)
    print(datetime.now()-start)
