'''
Primary Execution Loop
'''

import os
import math
import datetime
import schedule
import time
from itertools import chain

import googleapiclient.errors

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import numpy as np

from .client import YouTubeClient
from .classifier import YouTubeModel, YouTubeVocabEncoder
from .test_queue import process_daemon


def handle_new_uploads(youtube, model, playlist_names, playlist_ids):
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    subscription_channel_ids = youtube.get_subscriptions()
    upload_ids, upload_titles = zip(*reduce(
        lambda a, b: a + b, youtube.get_subscription_newuploads(channel_id, yesterday), []))
    predicted_playlist_names = model.predict(upload_titles)

    # TODO: turn prediction into an evaluation step?
    retraining = []
    for title, label, vid_id, playlist_id in zip(upload_titles, predicted_playlist_names, upload_ids, playlist_ids):
        updates = process_daemon(title, label, vid_id,
                                 playlist_id, playlist_names)
        if 'correct' in updates.keys():
            res = youtube.add_video_to_playlist(vid_id, playlist_id)
            retraining.append(
                (title, playlist_names[playlist_ids.index(playlist_id)]))
        elif 'different' in updates.keys():
            playlist_id = playlist_ids[playlist_names.index(
                updates['different'])]
            res = youtube.add_video_to_playlist(vid_id, playlist_id)
            retraining.append((title, updates['different']))
        elif 'create' in updates.keys():
            playlist_id = youtube.create_playlist(updates['create'])
            res = youtube.add_video_to_playlist(vid_id, playlist_id)
            retraining.append((title, updates['different']))


def main():
    youtube = YouTubeClient()
    playlist_ids, playlist_names = youtube.get_playlists()
    playlist_video_infos = [(video_info['name'], playlist_name) for playlist_id, playlist_name in zip(
        playlist_ids, playlist_names) for video_info in youtube.get_playlist_videos(playlist_id)]

    BUFFER_SIZE = 5000  # ???
    HIDDEN_SIZE = 64  # decrease to more reasonable batch size
    BATCH_SIZE = 8
    TAKE_SIZE = 500  # change to WATCH_LATER

    labeled_dataset = tf.data.Dataset \
        .from_tensor_slices(playlist_video_infos)
    .shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

    voc_encoder = YouTubeVocabEncoder()
    for vid_tensor, _ in labeled_dataset:
        voc_encoder.add_video_names(vid_tensor)
    voc_encoder.encode_init()
    encoded_labeled_dataset = labeled_dataset.map(voc_encoder.encode_video)
    vocab_size = voc_encoder.get_word_count() + 1

    train_data = encoded_labeled_dataset \
        .skip(TAKE_SIZE) \
        .shuffle(BUFFER_SIZE) \
        .padded_batch(BATCH_SIZE)
    test_data = encoded_labeled_dataset \
        .take(TAKE_SIZE) \
        .padded_batch(BATCH_SIZE)

    model = YouTubeModel(vocab_size, HIDDEN_SIZE, BATCH_SIZE)
    model.compile()
    model.fit(train_data, epochs=3)
    eval_loss, eval_acc = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(
        eval_loss, eval_acc))

    schedule \
        .every().day \
        .at('00:00') \
        .do(handle_new_uploads,
            youtube, model,
            playlist_names,
            playlist_ids)
    while True:
        schedule.run_pending()
        time.sleep(1000)


if __name__ == "__main__":
    main()
