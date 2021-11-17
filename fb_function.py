# reply function

import json
import requests
import numpy as np
import pytz
from datetime import datetime, timedelta
import dateutil.relativedelta
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import matplotlib
from requests_toolbelt import MultipartEncoder
import calendar
from bson.objectid import ObjectId
import os
import csv
import urllib.request
from video_process import *
from SIFT import *


matplotlib.rc("font", family='YouYuan')
tzinfo = pytz.timezone('Asia/Shanghai')

# Facebook Token
fb_token = ""


def create_image(df, client_id, title, o_size):
    """
    create a temporary image
    :param o_size:
    :param title:
    :param client_id:
    :param df:
    :return:
    """
    plt.cla()
    fig = plt.figure(figsize=(o_size[0], o_size[1]), dpi=1400)  # dpi表示清晰度
    ax = fig.add_subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_title(title)

    table(ax, df, loc='center')  # 将df换成需要保存的dataframe即可
    plt.savefig(f'temple_images/{client_id}.png')


def send_message(recipient_id, message_text):
    """
    send message to client
    :param recipient_id:
    :param message_text:
    :return:
    """
    params = {
        "access_token": fb_token
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v12.0/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        print('send failed')


def send_image(recipient_id):
    """
    send image to client
    :param recipient_id:
    :return:
    """
    params = {
        "access_token": fb_token
    }
    image_open = open(f'send_images/{recipient_id}.png', 'rb')
    data = {
        # encode nested json to avoid errors during multipart encoding process
        'recipient': json.dumps({
            'id': recipient_id
        }),
        # encode nested json to avoid errors during multipart encoding process
        'message': json.dumps({
            'attachment': {
                'type': 'image',
                'payload': {}
            }
        }),
        'filedata': (os.path.basename(f'send_images/{recipient_id}.png'), image_open, 'image/png')
    }
    # multipart encode the entire payload
    multipart_data = MultipartEncoder(data)

    # multipart header from multipart_data
    multipart_header = {
        'Content-Type': multipart_data.content_type
    }
    r = requests.post("https://graph.facebook.com/v12.0/me/messages", params=params, headers=multipart_header, data=multipart_data)
    image_open.close()
    if r.status_code != 200:
        print('send failed')
    return


def reply(messaging_event, sender_id):
    """
    Deal with request message
    :param sender_id:
    :param messaging_event: request's content
    :return:
    """

    # video or image
    if messaging_event[0]['type'] == 'video':
        # get the video url
        v_url = messaging_event[0]['payload']['url']

        # save the video
        video_name = f'{sender_id}.mp4'
        urllib.request.urlretrieve(v_url, 'videos/' + video_name)

        # get the clear images
        save_file = all_path(video_name)
        cv2.destroyAllWindows()
        image_clear = image_de(video_name, save_file)

        # 检测image_clear是否足够长
        if len(image_clear) >= 2:
            # SIFT start
            result, vis = stitch([cv2.imread(image_clear[-2]), cv2.imread(image_clear[-1])])
            for r in range(2, len(image_clear)):
                result, vis = stitch([cv2.imread(image_clear[-r]), result])
            cv2.imwrite('send_images/' + sender_id + '.png', result)
            send_image(sender_id)
        else:
            send_message(sender_id, '视频过于模糊，未提取到有效帧')
    else:
        send_message(sender_id, '暂时只支持视频')
