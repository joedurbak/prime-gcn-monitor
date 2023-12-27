from random import randint
import warnings

from astropy.coordinates import SkyCoord
from pandas import read_csv
from pandas.errors import SettingWithCopyWarning
import numpy as np

import settings

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

target_coords = SkyCoord('02h25m00.00s', " -04d30m00.0s")
# grid_df = read_csv('PRIME_sq_r1.2deg_packing.tess', sep=' ')
grid_df = read_csv('obsable_all_sky_grid.csv', sep=',')
output_name = 'some_name_here.csv'


def calculate_distance(ra, dec, target=target_coords):
    # grid_coords = SkyCoord(ra*units.deg, dec*units.deg)
    grid_coords = SkyCoord(ra, dec, unit=('hourangle', 'degree'))
    sep = target.separation(grid_coords)
    return sep.degree


def calculate_distance_all(target=target_coords, grid=grid_df):
    distances = []
    for ra, dec in zip(grid['RA'], grid['DEC']):
        distances.append(calculate_distance(ra, dec, target))
    d_arr = np.asarray(distances)
    min_dist = np.min(d_arr)
    index = np.where(d_arr == min_dist)
    # print(d_arr)
    grid_df['distance'] = d_arr
    return index, min_dist, grid_df


def generate_observation_csv(target, save_name, grid=grid_df):
    i, dist, new_df = calculate_distance_all(target, grid)
    new_df = new_df.sort_values('distance')
    new_df = new_df.iloc[0:settings.OBSERVATIONS.shape[0]]
    new_df = new_df.copy()
    new_df['Comment1'] = new_df['distance'].values[0]
    new_df['ObjectName'] = new_df['ObjectName'].values[0]
    new_df['ObjectType'] = new_df['ObjectType'].values[0]
    new_df['RA'] = new_df['RA'].values[0]
    new_df['DEC'] = new_df['DEC'].values[0]
    new_df = new_df.drop(columns=['distance'])
    new_df['Observer'] = 'NASA'
    new_df['DitherType'] = 'Random'
    new_df['DitherRadius'] = 90
    print(settings.OBSERVATIONS)
    new_df['Filter1'][:] = settings.OBSERVATIONS.filter1[:]
    new_df['Filter2'][:] = settings.OBSERVATIONS.filter2[:]
    new_df['IntegrationTime'][:] = settings.OBSERVATIONS.exposure_time_per_frame[:]
    new_df['DitherTotal'][:] = \
        (settings.OBSERVATIONS.total_exposure_time_seconds / settings.OBSERVATIONS.exposure_time_per_frame)[:]
    new_df['DitherTotal'] = new_df['DitherTotal'].astype(int)
    block_ids = ['P{:06d}'.format(randint(0, 999999)) for i in range(0, settings.OBSERVATIONS.shape[0])]
    new_df['BlockID'][:] = np.asarray(block_ids)[:]
    print(new_df.to_csv(save_name, index=False))
    print(new_df.to_string())
    return save_name


if __name__ == '__main__':
    generate_observation_csv(target_coords, output_name)
