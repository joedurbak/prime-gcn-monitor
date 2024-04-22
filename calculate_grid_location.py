from random import randint
import warnings

from astropy.coordinates import SkyCoord
from astropy import units as u
from pandas import read_csv
from pandas.errors import SettingWithCopyWarning
import numpy as np

import settings

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# target_coords = SkyCoord('08h11m36.81s', "-54d39m25.20s")
# target_coords = SkyCoord('09h54m05.13s', '-49d36m43.20s')
# target_coords = SkyCoord('10h51m15.43s', '+05d48m24.69s')
# target_coords = SkyCoord(165.4746, 9.9258, unit=u.deg)  # in the gap
# target_coords = SkyCoord(161.5702, 8.8237, unit=u.deg)  # in the gap
# target_coords = SkyCoord(157.5978, 4.1621, unit=u.deg)
# target_coords = SkyCoord('06:17:11.7864', '-44:44:39.876', unit=(u.hourangle, u.deg))
target_coords = SkyCoord(94.5677, -45.0114, unit=u.deg)
# 	23h25m54.33s	-55d07m55.20s
# grid_df = read_csv('PRIME_sq_r1.2deg_packing.tess', sep=' ')
grid_df = read_csv('obsable_all_sky_grid.csv', sep=',')
output_name = 'GRB240419a.csv'


def get_chip(ra_offset, dec_offset):
    ra_sign = ra_offset >= 0
    dec_sign = dec_offset >= 0
    if not ra_sign and dec_sign:
        return 1
    elif ra_sign and dec_sign:
        return 2
    elif not ra_sign and not dec_sign:
        return 3
    else:
        return 4


def calculate_distance(ra, dec, target=target_coords):
    # grid_coords = SkyCoord(ra*units.deg, dec*units.deg)
    grid_coords = SkyCoord(ra, dec, unit=('hourangle', 'degree'))
    sep = target.separation(grid_coords)
    sep_ra = target.ra.arcmin - grid_coords.ra.arcmin
    sep_dec = target.dec.arcmin - grid_coords.dec.arcmin
    return sep.arcmin, sep_ra, sep_dec, grid_coords.ra.degree, grid_coords.dec.degree, get_chip(sep_ra, sep_dec)


def calculate_distance_all(target=target_coords, grid=grid_df):
    distances = []
    ra_offsets = []
    dec_offsets = []
    ra_degrees = []
    dec_degrees = []
    chips = []
    for ra, dec in zip(grid['RA'], grid['DEC']):
        distance, ra_off, dec_off, ra_d, dec_d, chip = calculate_distance(ra, dec, target)
        distances.append(distance)
        ra_offsets.append(ra_off)
        dec_offsets.append(dec_off)
        ra_degrees.append(ra_d)
        dec_degrees.append(dec_d)
        chips.append(chip)
    d_arr = np.asarray(distances)
    min_dist = np.min(d_arr)
    index = np.where(d_arr == min_dist)
    # print(d_arr)
    grid_df['distance'] = d_arr
    grid_df['ra_offsets'] = np.asarray(ra_offsets)
    grid_df['dec_offsets'] = np.asarray(dec_offsets)
    grid_df['ra_degrees'] = np.asarray(ra_degrees)
    grid_df['dec_degrees'] = np.asarray(dec_degrees)
    grid_df['chip'] = np.asarray(chips)
    return index, min_dist, grid_df


def generate_point_source_csv(dataframe):
    new_df = dataframe.copy()
    print(new_df[['distance', 'ra_offsets', 'dec_offsets']].head(10))
    # print(new_df['ra_offsets'] > settings.MIN_RA_DEC_OFFSET_ARCMIN)
    ra_abs = new_df['ra_offsets'].abs()
    dec_abs = new_df['dec_offsets'].abs()
    new_df = new_df.loc[(ra_abs > settings.MIN_RA_DEC_OFFSET_ARCMIN) & (ra_abs < settings.MAX_RA_DEC_OFFSET_ARCMIN)]
    new_df = new_df.loc[(dec_abs > settings.MIN_RA_DEC_OFFSET_ARCMIN) & (dec_abs < settings.MAX_RA_DEC_OFFSET_ARCMIN)]
    print(new_df)
    expected_rows = settings.OBSERVATIONS.shape[0]
    current_rows = new_df.shape[0]
    if current_rows == 0:
        raise ValueError('no on grid location for this object')
    new_df = new_df.append([new_df] * expected_rows, ignore_index=True)
    new_df = new_df.iloc[0:settings.OBSERVATIONS.shape[0]]
    new_df = new_df.copy()
    new_df['Comment1'] = new_df['distance'].values[0]
    chip = new_df['chip'].values[0]
    if chip == 4:
        new_df['ROToffset'] = new_df['ROToffset'].values[0] + 90 * 60 * 60
        chip = 2
    new_df['Comment2'] = chip
    new_df['ObjectName'] = new_df['ObjectName'].values[0]
    new_df['ObjectType'] = new_df['ObjectType'].values[0]
    new_df['RA'] = new_df['RA'].values[0]
    new_df['DEC'] = new_df['DEC'].values[0]
    # new_df = new_df.drop(columns=['distance'])
    print(settings.OBSERVATIONS)
    new_df['Filter1'][:] = settings.OBSERVATIONS.filter1[:]
    new_df['Filter2'][:] = settings.OBSERVATIONS.filter2[:]
    new_df['IntegrationTime'][:] = settings.OBSERVATIONS.exposure_time_per_frame[:]
    new_df['DitherTotal'][:] = \
        (settings.OBSERVATIONS.total_exposure_time_seconds / settings.OBSERVATIONS.exposure_time_per_frame)[:]
    new_df['DitherTotal'] = new_df['DitherTotal'].astype(int)
    block_ids = ['P{:06d}'.format(randint(0, 999999)) for i in range(0, settings.OBSERVATIONS.shape[0])]
    new_df['BlockID'][:] = np.asarray(block_ids)[:]
    return new_df


def generate_observation_csv(target, save_name, grid=grid_df, tile_radius=None):
    i, dist, new_df = calculate_distance_all(target, grid)
    # new_df.to_csv('PRIME.tess', columns=['ObjectName', 'ra_degrees', 'dec_degrees'], sep=' ', index=False)
    new_df = new_df.sort_values('distance')
    if tile_radius is None:
        new_df = generate_point_source_csv(new_df)
    else:
        new_df = new_df.loc[new_df['distance'] < tile_radius]
        new_df['Comment1'] = new_df['distance']
        print(new_df)
    new_df = new_df.drop(columns=['distance', 'ra_offsets', 'dec_offsets', 'ra_degrees', 'dec_degrees', 'chip'])
    new_df['Observer'] = 'NASA'
    new_df['DitherType'] = 'Random'
    new_df['DitherRadius'] = 90
    print(new_df.to_csv(save_name, index=False))
    print(new_df.to_string())
    return save_name


if __name__ == '__main__':
    generate_observation_csv(target_coords, output_name)  # , tile_radius=3.14)
