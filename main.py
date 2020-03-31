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

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

import tensorflow as tf 
from tensorflow import keras
import tensorflow_datasets as tfds 

import numpy as np 

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"] # not .readonly?

def strikethrough(text):
    return ''.join([u'\u0336{}'.format(c) for c in text])

def initiate_api_client():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    #os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "YOUR_CLIENT_SECRET_FILE.json"

    # Get credentials and create an API client
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file,
        scopes)
    credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name,
        api_version,
        credentials=credentials)
    return youtube

def get_authuser_playlist_info(youtube):
    # Get playlist names and IDs
    playlist_names_request = youtube.playlists().list(part="snippet", mine=True) #pylint: disable=no-member
    playlist_names_response = playlist_names_request.execute()
    playlist_names = [playlist['snippet']['title'] for playlist in playlist_names_response['items']]
    playlist_ids = [playlist['id'] for playlist in playlist_names_response['items']]
    return playlist_names, playlist_ids

def get_authuser_playlists_videonames(youtube, playlist_names, playlist_ids):
    # Get names of videos for each playlist
    playlist_video_names = []
    video_playlist_names = {}
    for playlist_id, playlist_name in zip(playlist_ids, playlist_names):
        request = youtube.playlistItems().list(part="snippet", playlistId=playlist_id) #pylint: disable=no-member
        response = request.execute()
        video_names = [video['snippet']['title'] for video in response['items']]
        for video_name in video_names:
            if video_name in video_playlist_names.keys():
                video_name += ' '
            video_playlist_names[video_name] = playlist_name
            playlist_video_names.append(video_name)
    return playlist_video_names, video_playlist_names

def get_model_data(playlist_video_names, video_playlist_names, playlist_names):
    # AI model

    # Generate text file of video names
    with open('video_names.txt', 'w') as file:
        file.writelines(playlist_video_names)

    video_names_dataset = tf.data.TextLineDataset('video_names.txt')
    labeled_dataset = video_names_dataset.map(lambda vid: (vid, video_playlist_names[vid]))

    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    TAKE_SIZE = 5000

    labeled_dataset = labeled_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

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
        encoded_vid_name, label = tf.py_function(encode, inp=[video_name, label], Tout=(tf.int64, tf.string))
        encoded_vid_name.set_shape([None])
        label.set_shape([])
        return encoded_vid_name, label
    encoded_labeled_dataset = labeled_dataset.map(encode_map_fn)

    train_data = encoded_labeled_dataset.skip(TAKE_SIZE).shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    test_data = encoded_labeled_dataset.take(TAKE_SIZE).padded_batch(BATCH_SIZE)
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
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))

def listen_subscribed_uploads(youtube):
    # Listen for new uploads from subscribed channels
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    sub_request = youtube.subscriptions().list(part='snippet', mine=True, maxResults=50) #pylint: disable=no-member
    sub_response = sub_request.execute()
    subscriptions = sub_response['items']
    total_results = sub_response['pageInfo']['totalResults']
    results_per_page = sub_response['pageInfo']['resultsPerPage']
    num_more_calls = int(math.ceil((total_results - len(subscriptions))/results_per_page))
    for _ in range(num_more_calls):
        sub_request = youtube.subscriptions().list(part='snippet', mine=True, pageToken=sub_response['prevPageToken']) #pylint: disable=no-member
        sub_response = sub_request.execute()
        subscriptions += sub_response['items']
    all_new_uploads_titles = []
    all_new_uploads_ids = []
    for sub in subscriptions:
        channel_id = sub['snippet']['resourceId']['channelId']
        new_uploads_request = youtube.search().list(
            part='snippet',
            channelId=channel_id,
            publishedAfter=yesterday.isoformat() + 'T00:00:00.000Z',
            type='video')
        new_uploads_response = new_uploads_request.execute()
        new_uploads_titles = [vid['snippet']['title'] for vid in new_uploads_response['items']]
        new_uploads_ids = [vid['id']['videoId'] for vid in new_uploads_response['items']]
        all_new_uploads_ids += new_uploads_ids
        all_new_uploads_titles += new_uploads_titles
    return all_new_uploads_titles, all_new_uploads_ids

def handle_new_uploads(youtube, model, playlist_names, playlist_ids):
    all_new_uploads_titles, all_new_uploads_ids = listen_subscribed_uploads(youtube)
    predicted_labels = model.predict(all_new_uploads_titles)
    add_vid_to_playlist_info = []
    for title, label, vid_id, playlist_id in zip(all_new_uploads_titles, predicted_labels, all_new_uploads_ids, playlist_ids):
        inp = input(f'Is {label} the correct playlist for {title}? [yes/no/cancel]: ')
        if inp == 'yes':
            print('Thank you! Your playlist will be updated and the model trained.')
            add_vid_to_playlist_info.append(tuple(title, label, vid_id, playlist_id))
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
                print(f'Thank you! This video, \"{title}\", will be added to your \"{playlist_names[inp-1]}\" playlist and the model trained thereafter.')
                add_vid_to_playlist_info.append(tuple(title, playlist_names[inp-1], vid_id, playlist_ids[inp-1]))
            elif inp == i+2:
                print('Noted! Would you like to create a new playlist?')
                inp = input('If yes, specify the name. Otherwise, just type no: ')
                if inp == 'no':
                    print(f'{title} has been skipped and the model won\'t be updated to reflect your choice.')
                else:
                    print(f'Your playlist, \"{inp}\" will be created now.')
                    playlist_create_request = youtube.playlists().insert(body={'snippet':{'title':inp}})
                    playlist_create_response = playlist_create_request.execute() #pylint: disable=unused-variable
                    playlist_names.append(inp)
                    playlist_ids.append(playlist_create_response['id'])
                    print(f'Playlist created. This video, \"{title}\", will be added to your \"{inp}\" playlist and the model trained thereafter.')
                    add_vid_to_playlist_info.append(tuple(title, inp, vid_id, playlist_create_response['id']))
            else:
                print('Input invalid! Video skipped.')
        elif inp == 'cancel':
            print(f'{title} has been skipped and the model won\'t be updated to reflect your choice.')
        else:
            print(f'Input invalid! {title} has been skipped and the model won\'t be updated to reflect your choice.')


def main():
    youtube = initiate_api_client()
    playlist_names, playlist_ids = get_authuser_playlist_info(youtube)
    playlist_video_names, video_playlist_names = get_authuser_playlists_videonames(youtube, playlist_names, playlist_ids)
    model, train_data, test_data = get_model_data(playlist_video_names, video_playlist_names, playlist_names)
    train_model(model, train_data, test_data)
    schedule.every().day.at('00:00').do(handle_new_uploads, youtube, model, playlist_names, playlist_ids)
    while True:
        schedule.run_pending()
        time.sleep(1000)

if __name__ == "__main__":
    main()
