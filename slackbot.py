from time import sleep
import ssl
from datetime import date

from slack_sdk import WebClient
from astropy.coordinates import SkyCoord

from calculate_grid_location import generate_observation_csv
import settings


def post_gcn_alert(markdown_file, coordinates, images=tuple()):
    slack_token = settings.SLACK['slack_token']
    gcn_channel_id = settings.SLACK['slack_channel']
    message = settings.SLACK['initial_message']
    message += '\nInteractive airmass plot, etc. can be found here:\n'
    message += 'https://airmass.org/chart/obsid:{}/date:{}/object:gcnobject/ra:{:.6f}/dec:{:.6f}'.format(
        settings.AIRMASS_ORG_LOCATION, date.today().strftime('%Y-%m-%d'),
        coordinates.fk5.ra.deg, coordinates.fk5.dec.deg
    )
    if not settings.VERIFY_SSL:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    else:
        ssl_context = None
    client = WebClient(token=slack_token, ssl=ssl_context)
    response = client.chat_postMessage(
        channel=gcn_channel_id,
        text=message
    )
    print(message)

    # slack_markdown_file = markdown_to_slack_post(markdown_file)
    slack_markdown_file = markdown_file

    file_upload = client.files_upload_v2(
        file=slack_markdown_file,
        channel=gcn_channel_id,
        initial_comment='',
        thread_ts=response['ts'],
        title='GCN alert'
    )
    print(slack_markdown_file)
    print(file_upload)
    # thread_ts = file_upload['ts']
    sleep(3)
    remote_id = file_upload['files'][0]['id']
    client.files_remote_update(
        file=remote_id,
        filetype='post'
    )

    for image in images:
        client.files_upload_v2(
            file=image,
            channel=gcn_channel_id,
            initial_comment='',
            thread_ts=response['ts'],
            title='plot'
        )

    # message = 'Interactive airmass plot, etc. can be found here:\n'
    # message += 'https://airmass.org/chart/obsid:{}/date:{}/object:gcnobject/ra:{:.6f}/dec:{:.6f}'.format(
    #     settings.AIRMASS_ORG_LOCATION, date.today().strftime('%Y-%m-%d'),
    #     coordinates.fk5.ra.deg, coordinates.fk5.dec.deg
    # )
    # https://airmass.org/chart/obsid:salt/date:2024-02-06/object:gcnobject/ra:351.476375/dec:-55.132000
    # print(message)
    # next_response = client.chat_postMessage(
    #     channel=gcn_channel_id,
    #     text=message,
    #     thread_ts=response['ts']
    # )
    csv_file = markdown_file.replace('.md', '.csv')
    generate_observation_csv(coordinates, csv_file)
    message = settings.SLACK['observation_list_message']
    file_upload = client.files_upload_v2(
        file=csv_file,
        channel=gcn_channel_id,
        initial_comment=message,
        thread_ts=response['ts'],
        title='Observation list'
    )
    print(message)
    print(csv_file)


if __name__ == '__main__':
    markdown_f = r'C:\PycharmProjects\prime-gcn-monitor\output_md\SWIFT%23BAT_Lightcurve_1209398-335_63.md'
    target_coords = SkyCoord('02h25m00.00s', " -04d30m00.0s")
    post_gcn_alert(markdown_f, SkyCoord('02h25m00.00s', " -04d30m00.0s"))
