from matplotlib import pyplot as plt
from tracking.core import cell_tracking as ct
from datetime import datetime
import pandas as pd
import numpy as np
import pyart
import gc
from matplotlib.ticker import Formatter as formatter
from matplotlib import pyplot as plt
import matplotlib.animation
import matplotlib as mpl
import cartopy.feature as cfeature
import cartopy.crs as ccrs

out_dir = ''
tracks_path = '/Users/Mark/argonne/houston/combined_storms.csv'

tracks = pd.read_csv(tracks_path)
tracks.set_index(['storm_id', 'uid'], inplace=True)
cells = tracks.groupby(level=['storm_id', 'uid'])
tracks['life_iso'] = cells.apply(lambda x: np.all(x['isolated']))
tracks['nscans'] = cells.size()
ideal_cell = tracks.loc[tracks[tracks.life_iso]['nscans'].argmax()]
config_grid = pyart.io.read_grid(ideal_cell['file'].iloc[0])
grid_size = ct.get_grid_size(config_grid)

box_size = 50

a = 0
stepsize = 6
title_font = 20
axes_font = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

for ind, row in ideal_cell.iterrows(): 

    grid = pyart.io.read_grid(row['file'])
    display = pyart.graph.GridMapDisplay(grid)
    
    # Box Size
    tx = row['grid_x']
    ty = row['grid_y']
    lvxlim = np.array([tx - box_size, tx + box_size]) * grid_size[1]
    lvylim = np.array([ty - box_size, ty + box_size]) * grid_size[2]

    lat = grid.point_latitude['data'][0, int(ty), int(tx)]
    lon = grid.point_longitude['data'][0, int(ty), int(tx)]
    
    bsx = tx-grid.fields['reflectivity']['data'].shape[1]/2
    bsy = ty-grid.fields['reflectivity']['data'].shape[1]/2
    xlim = np.array([bsx - box_size, bsx + box_size]) * grid_size[1]/1000
    ylim = np.array([bsy - box_size, bsy + box_size]) * grid_size[2]/1000
    
    fig = plt.figure(figsize=(25,18))
    
    plt.title('Lagrangian View', fontsize=22)
    plt.axis('off')
    
    #Lagrangian View
    ax1 = fig.add_subplot(3, 2, (1, 3))

    display.plot_grid('reflectivity', level=ct.get_gs_alt(grid_size, 3000),
                      vmin=-8, vmax=64, mask_outside = False,
                      cmap=pyart.graph.cm.NWSRef,
                      ax = ax1, colorbar_flag = False, linewidth=4)
    display.plot_crosshairs(lon=lon, lat=lat, line_style='k--', linewidth=3)
    
    ax1.set_xlim(lvxlim[0], lvxlim[1])
    ax1.set_ylim(lvylim[0], lvylim[1])
    
    ax1.set_xticks(np.arange(lvxlim[0], lvxlim[1], (stepsize * 1000)))
    ax1.set_yticks(np.arange(lvylim[0], lvylim[1], (stepsize * 1000)))
    ax1.set_xticklabels(np.round((np.arange(xlim[0], xlim[1], stepsize)), 2))
    ax1.set_yticklabels(np.round((np.arange(ylim[0], ylim[1], stepsize)), 2))
    
    ax1.set_title('Top-Down View', fontsize = title_font)
    ax1.set_xlabel('East West Distance From Origin (km)' + '\n',
                   fontsize=axes_font)
    ax1.set_ylabel('North South Distance From Origin (km)',
                   fontsize=axes_font)
    
    #Latitude Cross Section
    ax2 = fig.add_subplot(3, 2, 2)
    display.plot_latitude_slice('reflectivity', lon=lon, lat=lat,
                                title_flag=False,
                                colorbar_flag=False, edges=False,
                                vmin=-8, vmax=64, mask_outside = False,
                                cmap=pyart.graph.cm.NWSRef,
                                ax = ax2)
    shift = 6
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.set_xticks(np.arange(xlim[0], xlim[1], stepsize))
    ax2.set_xticklabels(np.round((np.arange(xlim[0], xlim[1], stepsize)), 2))


    ax2.set_title('Latitude Cross Section', fontsize = title_font)
    ax2.set_xlabel('East West Distance From Origin (km)' + '\n',
                   fontsize=axes_font)
    ax2.set_ylabel('Distance Above Origin (km)', fontsize=axes_font)
    ax2.set_aspect(aspect = 1.4)

    #Longitude Cross Section
    ax3 = fig.add_subplot(3,2,4)
    display.plot_longitude_slice('reflectivity', lon=lon, lat=lat,
                                 title_flag=False, 
                                 colorbar_flag=False, edges=False,
                                 vmin=-8, vmax=64, mask_outside = False,
                                 cmap=pyart.graph.cm.NWSRef,
                                 ax = ax3)
    ax3.set_xlim(ylim[0], ylim[1])
    ax3.set_xticks(np.arange(ylim[0], ylim[1], stepsize))
    ax3.set_xticklabels(np.round((np.arange(ylim[0], ylim[1], stepsize)), 2))

    ax3.set_title('Longitudinal Cross Section', fontsize = title_font)
    ax3.set_xlabel('North South Distance From Origin (km)', fontsize=axes_font)
    ax3.set_ylabel('Distance Above Origin (km)', fontsize=axes_font)
    ax3.set_aspect(aspect = 1.4)

    #Statistics
    #   Data
    maxref = ideal_cell['max']
    maxalt = ideal_cell['max_alt']

    #   Time Calculation
    trackdatetime = ideal_cell['time']
    timeindex = trackdatetime.index
    plttime = []

    for t in trackdatetime.index:
        hour = (trackdatetime[t].hour)*100
        minute = trackdatetime[t].minute
        timecalc = hour + minute
        plttime.append(int(timecalc))

    #   Plot
    ax4 = fig.add_subplot(3,2,(5,6))
    #ax4.plot(plttime, vol[timeindex])
    ax4.plot(plttime, maxref[timeindex], color='b', linewidth=3)
    #ax4.plot(plttime, maxalt[timeindex], color='g')
    ax4.axvline(x=plttime[a], linewidth=4, color='r')
    ax4.set_title('Time Series', fontsize = title_font)
    ax4.set_xlabel('Time (UTC) \n Lagrangian Viewer Time (vertical line, red)',
                   fontsize=axes_font)
    ax4.set_ylabel('Maximum Reflectivity (dBZ, blue)', fontsize=axes_font)
    ax4.set_aspect(aspect = 5)  

    a = a+1

    #plot and save figure
    fig.savefig(outdir + 'scan_' + str(row['scan']) + '.png')
    plt.clf()
    del grid
    gc.collect()

