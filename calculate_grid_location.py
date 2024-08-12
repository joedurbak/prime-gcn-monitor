from random import randint
import warnings
from datetime import date

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
# target_coords = SkyCoord(122.3954, -24.3272, unit=u.deg)
# target_coords = SkyCoord(121.8567, -29.4609, unit=u.deg)
# target_coords = SkyCoord('22:30:38.84', '-39:07:20.06', unit=(u.hourangle, u.deg))
target_coords = SkyCoord(74.7900, -39.9100, unit=u.deg)
error_radius = 1

# target_coords = SkyCoord(94.5677, -45.0114, unit=u.deg)
# 	23h25m54.33s	-55d07m55.20s
# grid_df = read_csv('PRIME_sq_r1.2deg_packing.tess', sep=' ')
bulge_grid_df = read_csv('bulge_grid.csv', sep=',')
grid_df = read_csv('obsable_all_sky_grid.csv', sep=',')
offset_grid_df = read_csv('offset_obsable_all_sky_grid.csv', sep=',')
# output_name = 'S240422ed_GCN36278_x101.csv'
output_name = 'GRB240812C.csv'
# output_name = 'swiftgrb.csv'
default_rot_offset = 48 * 60 * 60


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
    grid['distance'] = d_arr
    grid['ra_offsets'] = np.asarray(ra_offsets)
    grid['dec_offsets'] = np.asarray(dec_offsets)
    grid['ra_degrees'] = np.asarray(ra_degrees)
    grid['dec_degrees'] = np.asarray(dec_degrees)
    grid['chip'] = np.asarray(chips)
    return index, min_dist, grid


def obtain_point_source_grid_df(dataframe):
    new_df = dataframe.copy()
    print(new_df[['distance', 'ra_offsets', 'dec_offsets']].head(10))
    # print(new_df['ra_offsets'] > settings.MIN_RA_DEC_OFFSET_ARCMIN)
    ra_abs = new_df['ra_offsets'].abs()
    dec_abs = new_df['dec_offsets'].abs()
    new_df = new_df.loc[(ra_abs > settings.MIN_RA_DEC_OFFSET_ARCMIN) & (ra_abs < settings.MAX_RA_DEC_OFFSET_ARCMIN)]
    new_df = new_df.loc[(dec_abs > settings.MIN_RA_DEC_OFFSET_ARCMIN) & (dec_abs < settings.MAX_RA_DEC_OFFSET_ARCMIN)]
    print(new_df)
    current_rows = new_df.shape[0]
    if current_rows == 0:
        raise ValueError('no on grid location for this object')
    return new_df


def generate_point_source_csv(dataframe):
    new_df = obtain_point_source_grid_df(dataframe)

    expected_rows = settings.OBSERVATIONS.shape[0]
    # current_rows = new_df.shape[0]
    new_df = new_df._append([new_df] * expected_rows, ignore_index=True)
    new_df = new_df.iloc[0:settings.OBSERVATIONS.shape[0]]
    new_df = new_df.copy()
    ra_off = new_df['ra_offsets'].values[0]
    dec_off = new_df['dec_offsets'].values[0]
    chip = new_df['chip'].values[0]
    if chip == 4:
        new_df['ROToffset'] = new_df['ROToffset'].values[0] + 90 * 60 * 60
        chip = 2
    new_df['Comment1'] = 'ra_off:{:+0.02f}+dec_off:{:+0.02f}+C{}'.format(ra_off, dec_off, chip)
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


def generate_observation_csv(
        target, save_name, grid=grid_df, backup_grid=offset_grid_df, tile_radius=None,
        default_rotation=default_rot_offset
):
    message = 'https://airmass.org/chart/obsid:{}/date:{}/object:gcnobject/ra:{:.6f}/dec:{:.6f}/object:bulge/ra:262.116667/dec:-31.020197/object:bulge%202/ra:271.033125/dec:-27.400156/'.format(
        settings.AIRMASS_ORG_LOCATION, date.today().strftime('%Y-%m-%d'),
        target.fk5.ra.deg, target.fk5.dec.deg
    )
    print(message)
    i, dist, new_df = calculate_distance_all(target, grid)
    # new_df.to_csv('PRIME.tess', columns=['ObjectName', 'ra_degrees', 'dec_degrees'], sep=' ', index=False)
    new_df = new_df.sort_values('distance')
    new_df['ROToffset'] = default_rotation
    if tile_radius is None:
        try:
            new_df = generate_point_source_csv(new_df)
        except ValueError:
            print('source is in the grid gaps, switching to offset grid')
            i, dist, new_df = calculate_distance_all(target, backup_grid)
            new_df = new_df.sort_values('distance')
            new_df['ROToffset'] = default_rotation
            new_df = generate_point_source_csv(new_df)
    else:
        new_df = new_df.loc[new_df['distance'] < tile_radius*60]
        new_df['Comment1'] = new_df['distance']
        print(new_df)
    new_df = new_df.drop(columns=['distance', 'ra_offsets', 'dec_offsets', 'ra_degrees', 'dec_degrees', 'chip'])
    new_df['Observer'] = 'NASA'
    new_df['DitherType'] = 'Random'
    new_df['DitherRadius'] = 90
    new_df['Comment2'] = save_name
    print(new_df.to_csv(save_name, index=False))
    print(new_df.to_string())
    return save_name


if __name__ == '__main__':
    generate_observation_csv(target_coords, output_name, grid=grid_df, tile_radius=error_radius)
