from astropy.coordinates import SkyCoord
from astropy import units
from pandas import read_csv
import numpy as np


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
    new_df = new_df.iloc[0:1]
    new_df['Comment1'] = new_df['distance']
    new_df = new_df.drop(columns=['distance'])
    new_df['Observer'] = 'NASA'
    new_df['DitherType'] = 'Random'
    new_df['DitherRadius'] = 90
    new_df['IntegrationTime'] = 20
    new_df['DitherTotal'] = 30 * 60 / 20
    print()
    print(new_df.to_csv(save_name, index=False))
    print()
    print('Printed observation list is for 30 minute J band exposure. You may want to edit DitherTotal, IntegrationTime, Filter1 or Filter2')
    return save_name


if __name__ == '__main__':
    generate_observation_csv(target_coords, output_name)
