from time import sleep

from slack_sdk import WebClient
from astropy.coordinates import SkyCoord

from calculate_grid_location import generate_observation_csv
import settings


def post_gcn_alert(markdown_file, coordinates, images=tuple()):
    slack_token = settings.SLACK['slack_token']
    gcn_channel_id = settings.SLACK['slack_channel']
    message = settings.SLACK['initial_message']

    client = WebClient(token=slack_token)
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
    markdown_f = '/Users/jdurbak/PycharmProjects/prime-gcn-monitor/output_md/SWIFT%23Point_Dir_2023-12-20T05%3A25%3A00.00_369155702-847_83.md'
    target_coords = SkyCoord('02h25m00.00s', " -04d30m00.0s")
    post_gcn_alert(markdown_f, SkyCoord('02h25m00.00s', " -04d30m00.0s"))
