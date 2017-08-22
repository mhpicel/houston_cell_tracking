import pandas as pd
from dask import dataframe as dd
import numpy as np
import pyart
import pyproj
from datetime import datetime

# Global Parameters
FILL_VALUE = -9999.
ZMELT = 5
NLAYERS = 3
RADIUS = 5000
PCTILE = 98.
USE_KDP_THRESH = False
RHO_THRESH = 0.8
DBZ_THRESH = 15.


#def get_grid_size(grid_obj):
#    """calculates grid size per dimension given a grid object."""
#    z_len = grid_obj.z['data'][-1] - grid_obj.z['data'][0]
#    x_len = grid_obj.x['data'][-1] - grid_obj.x['data'][0]
#    y_len = grid_obj.y['data'][-1] - grid_obj.y['data'][0]
#    z_size = z_len / (grid_obj.z['data'].shape[0] - 1)
#    x_size = x_len / (grid_obj.x['data'].shape[0] - 1)
#    y_size = y_len / (grid_obj.y['data'].shape[0] - 1)
#    return np.array([z_size, y_size, x_size])


def get_grid_alt(grid_size, alt_meters):
    return np.int(np.round(alt_meters/grid_size[0]))


def parse_date_string(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')


def filename_from_dt(dt, base):
    date = dt.strftime('%Y%m%d')
    file = '/KHGX_grid_' + dt.strftime('%Y%m%d.%H%M%S')
    ext = '.nc'
    return base + date + file + ext


def latlon_from_xy(xd,yd,lat_c,lon_c):
    #..Make a grids latgrid(x,y) and longrid(x,y) from x and y displacements
    g = pyproj.Geod(ellps='clrk66')
    #..Uses pyproj geod
    lat = np.empty([len(yd), len(xd)])
    lon = np.empty([len(yd), len(xd)])
    #..cacalate azimuth between radar and point
    for j in range(len(yd)):
        for i in range(len(xd)):
            faz = np.arctan2(xd[i], yd[j])
            dist = np.hypot(xd[i], yd[j])
            lon[j, i], lat[j, i], baz = g.fwd(lon_c, lat_c,
                                              180.*faz/np.pi,dist)
    return lat, lon


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
def preprocess_data(data, rho=None, zhh=None):
    data = np.ma.masked_values(data, FILL_VALUE)
    if rho is not None:
        data = np.ma.masked_where(rho < RHO_THRESH, data)
        data = np.ma.masked_where(zhh < DBZ_THRESH, data)
    return data


def get_neighborhood(dist_from_cent, kdp_proc, kdp_int, kdp_thresh=False):
    circle = dist_from_cent < RADIUS
    circle = np.tile(circle, (kdp_proc.shape[0], 1, 1))
    layers = np.zeros_like(kdp_proc.data)
    layers[ZMELT:ZMELT+NLAYERS+1, :, :] = 1
    neighborhood = np.logical_and(circle, layers.astype('bool'))

    if kdp_thresh:
        kdp_95 = np.percentile(kdp_proc[neighborhood], 95.)
        filtered_circle = np.logical_and(circle, kdp_int > kdp_95)
        neighborhood = np.locical_and(filtered_circle, layers.astype('bool'))

    return neighborhood


def cell_calcs(lat, lon, kdp_proc, zdr_proc, zhh_proc,
               kdp_pei, zdr_pei, zhh_pei, kdp_int, grid_ll):

    dist_from_cent = haversine(lat, lon, grid_ll[1], grid_ll[0])
    neighborhood = get_neighborhood(dist_from_cent, kdp_proc, kdp_int)
    kdp_pct = np.percentile(kdp_proc[neighborhood], PCTILE)
    zdr_pct = np.percentile(zdr_proc[neighborhood], PCTILE)
    zhh_pct = np.percentile(zhh_proc[neighborhood], PCTILE)

    if USE_KDP_THRESH:
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


def marcus_stats(scan_group):
    print(scan_group)
    file_name = scan_group['file'].iloc[0]
    grid = pyart.io.read_grid(file_name)
    grid_ll = grid.get_point_longitude_latitude()  # revise this

    rho = grid.fields['cross_correlation_ratio']['data']
    zhh = grid.fields['reflectivity']['data']
    kdp = grid.fields['specific_differential_phase']['data']
    zdr = grid.fields['differential_reflectivity']['data']

    kdp_proc = preprocess_data(kdp, rho=rho, zhh=zhh)
    zdr_proc = preprocess_data(zdr, rho=rho, zhh=zhh)
    zhh_proc = preprocess_data(zhh)

    kdp_int = np.sum(kdp_proc[ZMELT:NLAYERS+1, :, :], axis=0)

    z_column = grid.z['data'][:, np.newaxis, np.newaxis]
    kdp_pei = kdp_proc*z_column
    zdr_pei = zdr_proc*z_column
    zhh_pei = zhh_proc*z_column

    def get_cell_calcs(cell_row):
        """Wraps cell_calcs to be passed to apply call."""
        return cell_calcs(cell_row['lat'], cell_row['lon'],
                          kdp_proc, zdr_proc, zhh_proc,
                          kdp_pei, zdr_pei, zhh_pei,
                          kdp_int, grid_ll)

    output = scan_group.apply(get_cell_calcs, axis=1)
    marcus_frame = pd.DataFrame(output)

    del grid, rho, zhh, kdp, zdr, kdp_proc, zdr_proc, zhh_proc, kdp_int
    del kdp_pei, zdr_pei, zhh_pei
    return marcus_frame


def attach_marcus_stats(tracks_frame):
    # setup
#    file_name = tracks_frame['file'].iloc[0]
#    grid = pyart.io.read_grid(file_name)
#    gpars = get_gpars(grid)

    stats = tracks_frame.groupby(level='scan').apply(marcus_stats)
    return tracks_frame.join(stats)


#def get_gpars():

#%%
if __name__ =='__main__':
    grid_dir = '/home/mhpicel/blues_earthscience/radar/houston/data/'

    test_tracks_path = '/home/mhpicel/NASA/july2015/refl_long_storm.csv'
    test_tracks = pd.read_csv(test_tracks_path)
    test_tracks.set_index(['scan', 'uid'], inplace=True)
#    test_tracks.set_index('scan')
    test_tracks = test_tracks.loc[:10]
    test_tracks['time'] = test_tracks['time'].apply(parse_date_string)

    def get_filenames(file):
        return filename_from_dt(file, grid_dir)
    test_tracks['file'] = test_tracks['time'].apply(get_filenames)

    out_tracks = attach_marcus_stats(test_tracks)
