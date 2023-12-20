import os
from io import StringIO
from time import sleep

from slack_sdk import WebClient
from pandas import read_csv
from astropy.coordinates import SkyCoord

from calculate_grid_location import generate_observation_csv


def format_markdown_table(markdown_table_str):
    print(markdown_table_str)
    df = read_csv(StringIO(markdown_table_str), delimiter='|')
    df.dropna(axis=1, how='all')
    df2 = df[df.columns[1:-1]]
    df3 = df2.iloc[1:]
    df_str = str(df3.to_markdown(index=False))
    # df_str = str(df3.to_string(line_width=80, index=False))
    return_table_str = '```\n' + df_str + '\n```'
    print(return_table_str)
    return return_table_str


def markdown_to_slack_post(markdown_file):
    output_f = markdown_file.replace('.md', '.slack.md')
    with open(markdown_file) as _f:
        markdown_lines = _f.readlines()

    # removing comments
    no_comment_lines = [l for l in markdown_lines if not l.startswith('[comment]: <>')]

    # removing multiple blank lines
    output_lines = []
    previous_blank = False
    for l in no_comment_lines:
        blank_line = l.strip() == ''
        if not previous_blank or not blank_line:
            output_lines.append(l)
        if blank_line:
            previous_blank = True
        else:
            previous_blank = False
    output_message = ''.join(output_lines)
    table_str = ''
    previous_pipe = False
    for l in output_lines:
        start_pipe = l.startswith('|')
        if not start_pipe and previous_pipe:
            new_table = format_markdown_table(table_str)
            output_message = output_message.replace(table_str, new_table)
            table_str = ''
            previous_pipe = False
        if start_pipe:
            table_str += l
            previous_pipe = True
    with open(output_f, 'w') as _f:
        _f.write(output_message)
    return output_f


def post_gcn_alert(markdown_file, coordinates, images=tuple()):

    slack_token = os.environ['SLACKTOKEN']
    client = WebClient(token=slack_token)
    # gcn_channel_id = 'C06A8KNSV0X'  # gcn
    gcn_channel_id = 'C06APCA2H99'  # bot_testing

    message = "GCN Alert! More info incoming\n\n(Observatory weather)[https://suthweather.saao.ac.za/]"
    response = client.chat_postMessage(
        channel=gcn_channel_id,
        text=message
    )

    slack_markdown_file = markdown_to_slack_post(markdown_file)

    file_upload = client.files_upload_v2(
        file=slack_markdown_file,
        channel=gcn_channel_id,
        initial_comment='',
        thread_ts=response['ts'],
        title='GCN alert'
    )
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
    message = 'Observation list is for 30 minute J band exposure.\n'
    message += 'You may want to edit DitherTotal, IntegrationTime, Filter1 or Filter2\n'
    message += 'Filter1 can be Open, Z, Narrow or Dark\n'
    message += 'Filter2 can be Y, J, H or Open'
    file_upload = client.files_upload_v2(
        file=csv_file,
        channel=gcn_channel_id,
        initial_comment=message,
        thread_ts=response['ts'],
        title='Observation list'
    )


if __name__ == '__main__':
    markdown_f = '/Users/jdurbak/PycharmProjects/prime-gcn-monitor/output_md/SWIFT%23Point_Dir_2023-12-20T05%3A25%3A00.00_369155702-847_83.md'
    target_coords = SkyCoord('02h25m00.00s', " -04d30m00.0s")
    post_gcn_alert(markdown_f, SkyCoord('02h25m00.00s', " -04d30m00.0s"))
