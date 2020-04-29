'''
Run YouTube API commands
'''

# -*- coding: utf-8 -*-

# Sample Python code for youtube.playlists.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os
import math
import datetime
import schedule
import time


import googleapiclient.errors

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import numpy as np

from .client import YouTubeClient


def strikethrough(text):
    return ''.join([u'\u0336{}'.format(c) for c in text])


def initiate_api_client():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    #os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    # Get credentials and create an API client

    return youtube


def get_authuser_playlist_info(youtube):
    # Get playlist names and IDs


def get_authuser_playlists_videonames(youtube, playlist_names, playlist_ids):
    # Get names of videos for each playlist


def get_model_data(playlist_video_names, video_playlist_names, playlist_names):
    # AI model

    # Generate text file of video names
    with open('video_names.txt', 'w') as file:
        file.writelines(playlist_video_names)

    video_names_dataset = tf.data.TextLineDataset('video_names.txt')
    labeled_dataset = video_names_dataset.map(
        lambda vid: (vid, video_playlist_names[vid]))

    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    TAKE_SIZE = 5000

    labeled_dataset = labeled_dataset.shuffle(
        BUFFER_SIZE, reshuffle_each_iteration=False)

    tokenizer = tfds.features.text.Tokenizer()
    vocab_set = set()
    for vid_tensor, _ in labeled_dataset:
        tokens = tokenizer.tokenize(vid_tensor.numpy())
        vocab_set.update(tokens)

    encoder = tfds.features.text.TokenTextEncoder(vocab_set)

    def encode(vid_tensor, label):
        encoded_vid_name = encoder.encode(vid_tensor.numpy())
        return encoded_vid_name, label

    def encode_map_fn(video_name, label):
        encoded_vid_name, label = tf.py_function(
            encode, inp=[video_name, label], Tout=(tf.int64, tf.string))
        encoded_vid_name.set_shape([None])
        label.set_shape([])
        return encoded_vid_name, label
    encoded_labeled_dataset = labeled_dataset.map(encode_map_fn)

    train_data = encoded_labeled_dataset.skip(
        TAKE_SIZE).shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    test_data = encoded_labeled_dataset.take(
        TAKE_SIZE).padded_batch(BATCH_SIZE)
    vocab_size = len(vocab_set) + 1

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(len(playlist_names)))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model, train_data, test_data


def train_model(model, train_data, test_data):
    model.fit(train_data, epochs=3, validation_data=test_data)
    eval_loss, eval_acc = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(
        eval_loss, eval_acc))


def listen_subscribed_uploads(youtube):
    # Listen for new uploads from subscribed channels
    yesterday = datetime.date.today() - datetime.timedelta(days=1)


def handle_new_uploads(youtube, model, playlist_names, playlist_ids):
    all_new_uploads_titles, all_new_uploads_ids = listen_subscribed_uploads(
        youtube)
    predicted_labels = model.predict(all_new_uploads_titles)
    add_vid_to_playlist_info = []
    for title, label, vid_id, playlist_id in zip(all_new_uploads_titles, predicted_labels, all_new_uploads_ids, playlist_ids):
        inp = input(
            f'Is {label} the correct playlist for {title}? [yes/no/cancel]: ')
        if inp == 'yes':
            print('Thank you! Your playlist will be updated and the model trained.')
            add_vid_to_playlist_info.append(
                tuple(title, label, vid_id, playlist_id))
        elif inp == 'no':
            print('What is the correct playlist then?')
            for i, playlist_name in enumerate(playlist_names):
                if playlist_name == label:
                    print(strikethrough(f'[{i+1}] {playlist_name}'))
                else:
                    print(f'[{i+1}] {playlist_name}')
            print(f'[{i+2}] None of the above')
            inp = int(input('Please input the index of the correct playlist: '))
            if inp < i+2 and inp > 0:
                print(
                    f'Thank you! This video, \"{title}\", will be added to your \"{playlist_names[inp-1]}\" playlist and the model trained thereafter.')
                add_vid_to_playlist_info.append(
                    tuple(title, playlist_names[inp-1], vid_id, playlist_ids[inp-1]))
            elif inp == i+2:
                print('Noted! Would you like to create a new playlist?')
                inp = input(
                    'If yes, specify the name. Otherwise, just type no: ')
                if inp == 'no':
                    print(
                        f'{title} has been skipped and the model won\'t be updated to reflect your choice.')
                else:
                    print(f'Your playlist, \"{inp}\" will be created now.')
                    playlist_create_request = youtube.playlists().insert(
                        body={'snippet': {'title': inp}})
                    playlist_create_response = playlist_create_request.execute(
                    )  # pylint: disable=unused-variable
                    playlist_names.append(inp)
                    playlist_ids.append(playlist_create_response['id'])
                    print(
                        f'Playlist created. This video, \"{title}\", will be added to your \"{inp}\" playlist and the model trained thereafter.')
                    add_vid_to_playlist_info.append(
                        tuple(title, inp, vid_id, playlist_create_response['id']))
            else:
                print('Input invalid! Video skipped.')
        elif inp == 'cancel':
            print(
                f'{title} has been skipped and the model won\'t be updated to reflect your choice.')
        else:
            print(
                f'Input invalid! {title} has been skipped and the model won\'t be updated to reflect your choice.')


def main():
    youtube = initiate_api_client()
    playlist_names, playlist_ids = get_authuser_playlist_info(youtube)
    playlist_video_names, video_playlist_names = get_authuser_playlists_videonames(
        youtube, playlist_names, playlist_ids)
    model, train_data, test_data = get_model_data(
        playlist_video_names, video_playlist_names, playlist_names)
    train_model(model, train_data, test_data)
    schedule.every().day.at('00:00').do(handle_new_uploads,
                                        youtube, model, playlist_names, playlist_ids)
    while True:
        schedule.run_pending()
        time.sleep(1000)


if __name__ == "__main__":
    main()
