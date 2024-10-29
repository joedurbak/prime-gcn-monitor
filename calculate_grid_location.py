import os.path
from random import randint
import warnings
from datetime import date

from astropy.coordinates import SkyCoord
from astropy import units as u
from pandas import read_csv
from pandas import DataFrame
from pandas.errors import SettingWithCopyWarning
import numpy as np
from argparse import ArgumentParser

import settings

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# target_coords = SkyCoord(28.852, 5.957, unit=u.deg)
target_coords = SkyCoord(325.327, 5.086, unit=u.deg)
# target_coords = SkyCoord('00:30:37.0344','-89:20:29.244', unit=(u.hourangle, u.deg))
error_radius = None

bulge_grid_df = read_csv('bulge_grid.csv', sep=',')
grid_df = read_csv('obsable_all_sky_grid.csv', sep=',')
offset_grid_df = read_csv('offset_obsable_all_sky_grid.csv', sep=',')
# output_name = 'S240422ed_GCN36278_x101.csv'
output_name = 'GRB241029a.csv'
# output_name = 'swiftgrb.csv'
output_name = os.path.join(settings.OBSLIST_DIR, output_name)
default_rot_offset = 48 * 60 * 60

default_csv_fields = settings.OBSERVATION_LIST_DEFAULTS.copy()


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


def calculate_distance(ra, dec, target=target_coords, lazy=False):
    """

    :param ra:
    :param dec:
    :param target:
    :param lazy: add the absolute values of ra and dec offsets to save computation time.
        Useful if only trying to get a rough measure
    :return:
    """
    # grid_coords = SkyCoord(ra*units.deg, dec*units.deg)
    grid_coords = SkyCoord(ra, dec, unit=('hourangle', 'degree'))
    sep_ra = target.ra.arcmin - grid_coords.ra.arcmin
    sep_dec = target.dec.arcmin - grid_coords.dec.arcmin
    if not lazy:
        sep = target.separation(grid_coords).arcmin
    else:
        sep = abs(sep_ra) + abs(sep_dec)
    return sep, sep_ra, sep_dec, grid_coords.ra.degree, grid_coords.dec.degree, get_chip(sep_ra, sep_dec)


def calculate_distance_all(target=target_coords, grid=grid_df, lazy=False):
    distances = []
    ra_offsets = []
    dec_offsets = []
    ra_degrees = []
    dec_degrees = []
    chips = []
    for ra, dec in zip(grid['RA'], grid['DEC']):
        distance, ra_off, dec_off, ra_d, dec_d, chip = calculate_distance(ra, dec, target, lazy=lazy)
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


def obtain_point_source_grid_df(dataframe, point_source_radius=3/60, dither_radius=90/60):
    new_df = dataframe.copy()
    print(new_df[['distance', 'ra_offsets', 'dec_offsets']].head(10))
    # print(new_df['ra_offsets'] > settings.MIN_RA_DEC_OFFSET_ARCMIN)
    ra_abs = new_df['ra_offsets'].abs()
    dec_abs = new_df['dec_offsets'].abs()
    min_offset = settings.MIN_RA_DEC_OFFSET_ARCMIN + point_source_radius + dither_radius
    max_offset = settings.MAX_RA_DEC_OFFSET_ARCMIN - point_source_radius - dither_radius
    new_df = new_df.loc[(ra_abs > min_offset) & (ra_abs < max_offset)]
    new_df = new_df.loc[(dec_abs > min_offset) & (dec_abs < max_offset)]
    print(new_df)
    current_rows = new_df.shape[0]
    if current_rows == 0:
        raise ValueError('no on grid location for this object')
    return new_df


def observation_list_to_submission_format(
        observation_list_df, grid_type, target_name,
        object_type=settings.OBSERVATION_LIST_DEFAULTS['ObjectType'],
        pi_name=settings.OBSERVATION_LIST_DEFAULTS['PIName'],
        specific_time=settings.OBSERVATION_LIST_DEFAULTS['SpecificTime'],
        priority=settings.OBSERVATION_LIST_DEFAULTS['Priority']
):
    n_rows = observation_list_df.shape[0]
    columns = {
        'Observer(PI institute)': observation_list_df['Observer'].to_list(),
        'GridType': [grid_type for i in range(n_rows)],
        'ObjectType': [object_type for i in range(n_rows)],
        'RA(h:m:s)': observation_list_df['RA'].to_list(),
        'DEC(d:m:s)': observation_list_df['DEC'].to_list(),
        'RAoffset(")': observation_list_df['RAoffset'].to_list(),
        'DECoffset(")': observation_list_df['DECoffset'].to_list(),
        'ROToffset(°)': ((observation_list_df['ROToffset'] - default_rot_offset) /3600).to_list(),
        'Filter1': observation_list_df['Filter1'].to_list(),
        'Filter2': observation_list_df['Filter2'].to_list(),
        'DitherType': observation_list_df['DitherType'].to_list(),
        'DitherRadius(")': observation_list_df['DitherRadius'].to_list(),
        'DitherPhase(°)': observation_list_df['DitherPhase'].to_list(),
        'DitherTotal': observation_list_df['DitherTotal'].to_list(),
        'Images': observation_list_df['Images'].to_list(),
        'IntegrationTime(s)': observation_list_df['IntegrationTime'].to_list(),
        'PIName': [pi_name for i in range(n_rows)],
        'TargetName': [target_name for i in range(n_rows)],
        'SpecificTime': [specific_time for i in range(n_rows)],
        'Priority': [priority for i in range(n_rows)],
        'Comment': observation_list_df['Comment1'].to_list()
    }
    return DataFrame(columns)


def generate_point_source_centered_csv(dataframe, coords):
    new_df = dataframe.copy()
    expected_rows = settings.OBSERVATIONS.shape[0]
    new_df = new_df.iloc[:expected_rows]
    # new_df = new_df._append([new_df] * expected_rows, ignore_index=True)
    new_df['RA'] = coords.fk5.ra.to_string(unit=u.hour, sep=':')
    new_df['DEC'] = coords.fk5.dec.to_string(unit=u.degree, sep=':')
    offset = 1125
    new_df['RAoffset'] = offset
    new_df['DECoffset'] = offset
    new_df['Comment1'] = 'ra_off:{:+0.02f}+dec_off:{:+0.02f}+C{}+{}'.format(offset, offset, 3, 'no_grid')
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


def generate_point_source_csv(dataframe, point_source_radius=3/60):
    new_df = obtain_point_source_grid_df(dataframe, point_source_radius=point_source_radius)

    expected_rows = settings.OBSERVATIONS.shape[0]
    # current_rows = new_df.shape[0]
    new_df = new_df._append([new_df] * expected_rows, ignore_index=True)
    new_df = new_df.iloc[0:settings.OBSERVATIONS.shape[0]]
    new_df = new_df.copy()
    ra_off = new_df['ra_offsets'].values[0]
    dec_off = new_df['dec_offsets'].values[0]
    chip = new_df['chip'].values[0]
    rotated = ''
    if chip == 4:
        new_df['ROToffset'] = new_df['ROToffset'].values[0] + 90 * 60 * 60
        chip = 2
        rotated = '+rotated90'
    new_df['Comment1'] = 'ra_off:{:+0.02f}+dec_off:{:+0.02f}+C{}+{}{}'.format(ra_off, dec_off, chip, new_df['ObjectName'].values[0], rotated)
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
        default_rotation=default_rot_offset, target_name=None, point_source_radius=3/60, force_centered=False
):
    message = 'https://airmass.org/chart/obsid:{}/date:{}/object:gcnobject/ra:{:.6f}/dec:{:.6f}/object:bulge/ra:262.116667/dec:-31.020197/object:bulge%202/ra:271.033125/dec:-27.400156/'.format(
        settings.AIRMASS_ORG_LOCATION, date.today().strftime('%Y-%m-%d'),
        target.fk5.ra.deg, target.fk5.dec.deg
    )
    if target_name is None:
        target_name = '.'.join(os.path.basename(save_name).split('.')[:-1])
    print(message)
    lazy = False
    if tile_radius is None:
        lazy = True
    i, dist, new_df = calculate_distance_all(target, grid, lazy=lazy)
    # new_df.to_csv('PRIME.tess', columns=['ObjectName', 'ra_degrees', 'dec_degrees'], sep=' ', index=False)
    new_df = new_df.sort_values('distance')
    new_df['ROToffset'] = default_rotation
    grid_type = 'all_sky_grid'
    if force_centered:
        new_df = generate_point_source_centered_csv(grid, target)
    elif tile_radius is None:
        try:
            new_df = generate_point_source_csv(new_df, point_source_radius=point_source_radius)
        except ValueError:
            print('source is in the grid gaps, switching to offset grid')
            grid_type = 'no_grid'
            i, dist, new_df = calculate_distance_all(target, backup_grid)
            new_df = new_df.sort_values('distance')
            new_df['ROToffset'] = default_rotation
            try:
                new_df = generate_point_source_csv(new_df, point_source_radius=point_source_radius)
            except ValueError:
                print('source is in the grid gaps, going off grid')
                new_df = generate_point_source_centered_csv(grid, target)
    else:
        new_df = new_df.loc[new_df['distance'] < tile_radius*60]
        new_df['Comment1'] = new_df['distance']
        print(new_df)
    new_df = new_df.drop(columns=['distance', 'ra_offsets', 'dec_offsets', 'ra_degrees', 'dec_degrees', 'chip'])
    new_df['Observer'] = 'NASA'
    new_df['DitherType'] = 'Random'
    new_df['DitherRadius'] = 90
    new_df['Comment2'] = save_name
    save_df = observation_list_to_submission_format(new_df, grid_type=grid_type, target_name=target_name)
    print(save_df.to_csv(save_name, index=False))
    print(save_df.to_string())
    print("Submit {} observation file to https://docs.google.com/forms/d/e/1FAIpQLSeXhoZNJcR8_4zFdiFDUNi-Q2kV9oZuTB-PrJUWwrXBZCSgFQ/viewform".format(save_name))
    return save_name


def generate_observation_csv_ra_dec(
    target_ra, target_dec, save_name, grid=grid_df, backup_grid=offset_grid_df, tile_radius=None,
    default_rotation=default_rot_offset, target_name=None, point_source_radius=3/60, force_centered=False
):
    try:
        target_ra = float(target_ra)
        target_dec = float(target_dec)
        coords = SkyCoord(target_ra, target_dec, unit=u.deg)
    except ValueError:
        coords = SkyCoord(target_ra, target_dec, unit=(u.hourangle, u.deg))
    generate_observation_csv(
        coords, save_name, grid, backup_grid, tile_radius, default_rotation, target_name, point_source_radius,
        force_centered
    )



def main():
    parser = ArgumentParser()

    parser.add_argument('ra', type=str, help='[float or str] should either be in degrees (fff.fff) or hour angle (HHH:MM:SS.SSS or HHHhMMmSS.SSSs)')
    parser.add_argument('dec', type=str, help='[float or str] should either be in degrees (fff.fff) or dms (+DD:MM:SS.SSS or +DDdMMmSS.SSSs)')
    parser.add_argument(
        'filename', type=str, help="[str], output filename",
    )
    parser.add_argument(
        '-t', '--tile', type=float, help='[float] optional, radius for tiling',
        default=None
    )
    parser.add_argument(
        '-e', '--error_radius', type=float, help='[float] optional, error radius for well localized point source',
        default=3/60
    )
    parser.add_argument(
        '-n', '--name', type=str, help='[str] optional, name to use for target if it does not match filename'
        , default=None
    )
    parser.add_argument(
        '-c', '--center', action='store_true', help='force well localized point source to center of chip 3',
        default=False
    )
    args = parser.parse_args()
    generate_observation_csv_ra_dec(
        args.ra, args.dec, args.filename, tile_radius=args.tile, target_name=args.name,
        point_source_radius=args.error_radius, force_centered=args.center
    )


if __name__ == '__main__':
    main()
    # generate_observation_csv(target_coords, output_name, grid=grid_df, tile_radius=error_radius, point_source_radius=3)
