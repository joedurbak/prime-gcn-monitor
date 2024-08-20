import os

import gcn
import numpy as np
from pandas import DataFrame
from astropy import units

AIRMASS_LIMIT = 2.0
MARKDOWN = True
DCT_LOC_PICKLE = 'salt_loc.pickle'
DCT_ASTROPLAN_LOC_PICKLE = 'salt_astroplan_loc.pickle'
ARCHIVED_XML_DIR = "processed_xml"
OUTPUT_HTML_DIR = "output_html"
TEMPLATE_HTML_DIR = "html_templates"
LOCATION_NAME = "Sutherland"
AIRMASS_ORG_LOCATION = 'salt'
OBSERVABLE_TIME_MINIMUM_MINUTES = 15
MAX_RA_DEC_OFFSET_ARCMIN = 37
MIN_RA_DEC_OFFSET_ARCMIN = 4

_max_tiling_time_s = 60 * 60 * 2
_tile_time_s = 10 * 10 * 2
_n_tiles = int(_max_tiling_time_s / _tile_time_s)
_fov_sq_deg = 1.56
_max_tile_area = _fov_sq_deg * _n_tiles
MAX_ERROR_RADIUS = (np.sqrt(_max_tile_area/np.pi)) * units.deg
print('MAX_ERROR_RADIUS', MAX_ERROR_RADIUS)
MIN_TILE_ERROR_RADIUS = 0.5*units.deg

FALSE_FLAGS = [
        "./What/Group/Param[@name='StarTrack_Lost_Lock']",
        "./What/Group/Param[@name='Def_NOT_a_GRB']"
]
TRUE_FLAGS = []

HTML_TEMPLATES_DICT = {
    'astro_coords_html': 'astro_coords.html',
    'author_html': 'author.html',
    'authors_html': 'authors.html',
    'citation_html': 'citation.html',
    'citations_html': 'citations.html',
    'event_ivorn_html': 'event_ivorn.html',
    'event_ivorns_html': 'event_ivorns.html',
    'field_html': 'field.html',
    'fields_html': 'fields.html',
    'how_html': 'how.html',
    'inference_html': 'inference.html',
    'inferences_html': 'inferences.html',
    'observation_location_html': 'observation_location.html',
    'observatory_location_html': 'observatory_location.html',
    'param_html': 'param.html',
    'params_html': 'params.html',
    'position2d_html': 'position2d.html',
    'position3d_html': 'position3d.html',
    'reference_html': 'reference.html',
    'references_html': 'references.html',
    'table_html': 'table.html',
    'tables_html': 'tables.html',
    'time_instant_html': 'time_instant.html',
    'voevent_html': 'voevent.html',
    'what_html': 'what.html',
    'wherewhen_html': 'wherewhen.html',
    'who_html': 'who.html',
    'why_html': 'why.html',
    'icon_html': 'icon.html',
    'modal_html': 'modal.html',
    'groups_html': 'groups.html',
    'group_html': 'group.html',
    'container_html': 'container.html',
    'simple_row_html': 'simple_row.html',
}

if MARKDOWN:
    for k, v in HTML_TEMPLATES_DICT.items():
        HTML_TEMPLATES_DICT[k] = v.replace('.html', '.md')
    OUTPUT_HTML_DIR = "output_md"
    TEMPLATE_HTML_DIR = "markdown_templates"

for k, v in HTML_TEMPLATES_DICT.items():
    HTML_TEMPLATES_DICT[k] = os.path.join(TEMPLATE_HTML_DIR, v)

for directory in [ARCHIVED_XML_DIR, OUTPUT_HTML_DIR]:
    if not os.path.isdir(directory):
        os.makedirs(directory)

INCLUDE_ALERT_MESSAGES = (
    # gcn.notice_types.GRB_COORDS,
    # gcn.notice_types.TEST_COORDS,
    # gcn.notice_types.IM_ALIVE,
    # gcn.notice_types.KILL_SOCKET,
    # gcn.notice_types.MAXBC,
    # gcn.notice_types.BRAD_COORDS,
    # gcn.notice_types.GRB_FINAL,
    # gcn.notice_types.HUNTS_SRC,
    # gcn.notice_types.ALEXIS_SRC,
    # gcn.notice_types.XTE_PCA_ALERT,
    # gcn.notice_types.XTE_PCA_SRC,
    # gcn.notice_types.XTE_ASM_ALERT,
    # gcn.notice_types.XTE_ASM_SRC,
    # gcn.notice_types.COMPTEL_SRC,
    # gcn.notice_types.IPN_RAW,
    # gcn.notice_types.IPN_SEG,
    # gcn.notice_types.SAX_WFC_ALERT,
    # gcn.notice_types.SAX_WFC_SRC,
    # gcn.notice_types.SAX_NFI_ALERT,
    # gcn.notice_types.SAX_NFI_SRC,
    # gcn.notice_types.XTE_ASM_TRANS,
    # gcn.notice_types.spare38,
    gcn.notice_types.IPN_POS,
    # gcn.notice_types.HETE_ALERT_SRC,
    # gcn.notice_types.HETE_UPDATE_SRC,
    # gcn.notice_types.HETE_FINAL_SRC,
    # gcn.notice_types.HETE_GNDANA_SRC,
    # gcn.notice_types.HETE_TEST,
    # gcn.notice_types.GRB_CNTRPART,
    # gcn.notice_types.SWIFT_TOO_FOM,
    # gcn.notice_types.SWIFT_TOO_SC_SLEW,
    # gcn.notice_types.DOW_TOD,
    # gcn.notice_types.spare50,
    # gcn.notice_types.INTEGRAL_POINTDIR,
    # gcn.notice_types.INTEGRAL_SPIACS,
    # gcn.notice_types.INTEGRAL_WAKEUP,
    # gcn.notice_types.INTEGRAL_REFINED,
    # gcn.notice_types.INTEGRAL_OFFLINE,
    # gcn.notice_types.INTEGRAL_WEAK,
    # gcn.notice_types.AAVSO,
    # gcn.notice_types.MILAGRO_POS,
    # gcn.notice_types.KONUS_LC,
    gcn.notice_types.SWIFT_BAT_GRB_ALERT,
    # gcn.notice_types.SWIFT_BAT_GRB_POS_ACK,
    # gcn.notice_types.SWIFT_BAT_GRB_POS_NACK,
    gcn.notice_types.SWIFT_BAT_GRB_LC,
    gcn.notice_types.SWIFT_BAT_SCALEDMAP,
    # gcn.notice_types.SWIFT_FOM_OBS,
    # gcn.notice_types.SWIFT_SC_SLEW,
    gcn.notice_types.SWIFT_XRT_POSITION,
    # gcn.notice_types.SWIFT_XRT_SPECTRUM,
    # gcn.notice_types.SWIFT_XRT_IMAGE,
    # gcn.notice_types.SWIFT_XRT_LC,
    # gcn.notice_types.SWIFT_XRT_CENTROID,
    # gcn.notice_types.SWIFT_UVOT_DBURST,
    # gcn.notice_types.SWIFT_UVOT_FCHART,
    # gcn.notice_types.SWIFT_BAT_GRB_LC_PROC,
    # gcn.notice_types.SWIFT_XRT_SPECTRUM_PROC,
    # gcn.notice_types.SWIFT_XRT_IMAGE_PROC,
    # gcn.notice_types.SWIFT_UVOT_DBURST_PROC,
    # gcn.notice_types.SWIFT_UVOT_FCHART_PROC,
    # gcn.notice_types.SWIFT_UVOT_POS,
    # gcn.notice_types.SWIFT_BAT_GRB_POS_TEST,
    # gcn.notice_types.SWIFT_POINTDIR,
    gcn.notice_types.SWIFT_BAT_TRANS,
    # gcn.notice_types.SWIFT_XRT_THRESHPIX,
    # gcn.notice_types.SWIFT_XRT_THRESHPIX_PROC,
    # gcn.notice_types.SWIFT_XRT_SPER,
    # gcn.notice_types.SWIFT_XRT_SPER_PROC,
    # gcn.notice_types.SWIFT_UVOT_POS_NACK,
    gcn.notice_types.SWIFT_BAT_ALARM_SHORT,
    gcn.notice_types.SWIFT_BAT_ALARM_LONG,
    # gcn.notice_types.SWIFT_UVOT_EMERGENCY,
    # gcn.notice_types.SWIFT_XRT_EMERGENCY,
    # gcn.notice_types.SWIFT_FOM_PPT_ARG_ERR,
    # gcn.notice_types.SWIFT_FOM_SAFE_POINT,
    # gcn.notice_types.SWIFT_FOM_SLEW_ABORT,
    # gcn.notice_types.SWIFT_BAT_QL_POS,
    # gcn.notice_types.SWIFT_BAT_SUB_THRESHOLD,
    gcn.notice_types.SWIFT_BAT_SLEW_POS,
    # gcn.notice_types.AGILE_GRB_WAKEUP,
    # gcn.notice_types.AGILE_GRB_GROUND,
    # gcn.notice_types.AGILE_GRB_REFINED,
    # gcn.notice_types.SWIFT_ACTUAL_POINTDIR,
    # gcn.notice_types.AGILE_MCAL_ALERT,
    # gcn.notice_types.AGILE_POINTDIR,
    # gcn.notice_types.AGILE_TRANS,
    # gcn.notice_types.AGILE_GRB_POS_TEST,
    gcn.notice_types.FERMI_GBM_ALERT,
    # gcn.notice_types.FERMI_GBM_FLT_POS,
    # gcn.notice_types.FERMI_GBM_GND_POS,
    # gcn.notice_types.FERMI_GBM_LC,
    # gcn.notice_types.FERMI_GBM_GND_INTERNAL,
    gcn.notice_types.FERMI_GBM_FIN_POS,
    # gcn.notice_types.FERMI_GBM_ALERT_INTERNAL,
    # gcn.notice_types.FERMI_GBM_FLT_INTERNAL,
    gcn.notice_types.FERMI_GBM_TRANS,
    # gcn.notice_types.FERMI_GBM_POS_TEST,
    gcn.notice_types.FERMI_LAT_POS_INI,
    gcn.notice_types.FERMI_LAT_POS_UPD,
    gcn.notice_types.FERMI_LAT_POS_DIAG,
    gcn.notice_types.FERMI_LAT_TRANS,
    # gcn.notice_types.FERMI_LAT_POS_TEST,
    gcn.notice_types.FERMI_LAT_MONITOR,
    # gcn.notice_types.FERMI_SC_SLEW,
    # gcn.notice_types.FERMI_LAT_GND,
    # gcn.notice_types.FERMI_LAT_OFFLINE,
    # gcn.notice_types.FERMI_POINTDIR,
    # gcn.notice_types.SIMBADNED,
    # gcn.notice_types.FERMI_GBM_SUBTHRESH,
    gcn.notice_types.SWIFT_BAT_MONITOR,
    # gcn.notice_types.MAXI_UNKNOWN,
    # gcn.notice_types.MAXI_KNOWN,
    # gcn.notice_types.MAXI_TEST,
    # gcn.notice_types.OGLE,
    # gcn.notice_types.CBAT,
    # gcn.notice_types.MOA,
    # gcn.notice_types.SWIFT_BAT_SUBSUB,
    # gcn.notice_types.SWIFT_BAT_KNOWN_SRC,
    # gcn.notice_types.VOE_11_IM_ALIVE,
    # gcn.notice_types.VOE_20_IM_ALIVE,
    # gcn.notice_types.FERMI_SC_SLEW_INTERNAL,
    # gcn.notice_types.COINCIDENCE,
    # gcn.notice_types.FERMI_GBM_FIN_INTERNAL,
    # gcn.notice_types.SUZAKU_LC,
    # gcn.notice_types.SNEWS,
    # gcn.notice_types.LVC_PRELIMINARY,
    # gcn.notice_types.LVC_INITIAL,
    # gcn.notice_types.LVC_UPDATE,
    # gcn.notice_types.LVC_TEST,
    # gcn.notice_types.LVC_COUNTERPART,
    # gcn.notice_types.AMON_ICECUBE_COINC,
    # gcn.notice_types.AMON_ICECUBE_HESE,
    # gcn.notice_types.CALET_GBM_FLT_LC,
    # gcn.notice_types.CALET_GBM_GND_LC,
    # gcn.notice_types.LVC_EARLY_WARNING,
    # gcn.notice_types.LVC_RETRACTION,
    # gcn.notice_types.GWHEN_COINC,
    # gcn.notice_types.AMON_ICECUBE_EHE,
    # gcn.notice_types.HAWC_BURST_MONITOR,
    # gcn.notice_types.AMON_NU_EM_COINC,
    # gcn.notice_types.ICECUBE_ASTROTRACK_GOLD,
    # gcn.notice_types.ICECUBE_ASTROTRACK_BRONZE,
    # gcn.notice_types.SK_SN,
    # gcn.notice_types.ICECUBE_CASCADE,
    # gcn.notice_types.GECAM_FLT,
    # gcn.notice_types.GECAM_GND
)

_observations = [
    # (filter1, filter2, total_exposure_time_seconds, exposure_time_per_frame)
    ('Open', 'J', 30*60, 60),
    ('Open', 'H', 30*60, 11.44),
    # ('Open', 'Y', 30*60, 20),
    # ('Z', 'Open', 30*60, 20)
]

OBSERVATIONS = DataFrame(_observations)
OBSERVATIONS.columns = ('filter1', 'filter2', 'total_exposure_time_seconds', 'exposure_time_per_frame')
_total_observation_time = OBSERVATIONS.total_exposure_time_seconds.sum()
_total_telescope_time = _total_observation_time * 2

_message = 'Observation list contains the following exposures:\n\n'
_message += '```\n' + str(OBSERVATIONS.to_markdown()) + '\n```\n'
_message += '*Some notes*\n'
_message += '  - To observe this target, send the observation list csv to the obs-request channel'
_message += ' on the PRIME discord\n'
_message += '  - Total telescope time, {} minutes, is ~2x total exposure time for all filters, {} minutes.\n'.format(
    int(_total_telescope_time/60), int(_total_observation_time/60)
)
_message += '  - If desired, it is recommended to edit the exposure time by altering the DitherTotal field in the csv\n'
_message += '  - When adding new lines to the observation list csv, the BlockID values must be unique\n'
_message += '  - Filter1 options are Open, Z, Narrow and Dark\n'
_message += '  - Filter2 options are Y, J, H and Open\n'
_message += '  - If you have trouble reaching the observer, try calling the observatory: +27 231000191'

SLACK = {
    'initial_message': "GCN Alert! More info incoming\n\nObservatory weather: https://suthweather.saao.ac.za/",
    'slack_token': os.environ.get('SLACKTOKEN', 'NA'),
    # 'slack_channel': 'C06APCA2H99',  # bot_testing
    'slack_channel': 'C06A8KNSV0X',  # gcn
    'observation_list_message': _message
}

VERIFY_SSL = False
