import os
import math
import google_auth_oauthlib.flow
import googleapiclient.discovery
from dotenv import load_dotenv

load_dotenv()

SCOPES = [os.getenv('SCOPES')]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
CLIENT_SECRETS_FILE = "CLIENT_SECRET_FILE.json"


class YouTubeClient:
    def __init__(self):
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_console()
        self.youtube = googleapiclient.discovery.build(
            API_SERVICE_NAME, API_VERSION, credentials=credentials)

    def get_playlists(self):
        request = self.youtube.playlists().list(part="snippet", mine=True)
        response = request.execute()
        playlists = [(playlist['id'], playlist['snippet']['title'])
                     for playlist in response['items']]
        return zip(*playlists)

    def get_subscriptions(self):
        request = self.youtube.subscriptions().list(
            part="snippet", mine=True, maxResults=50)
        response = request.execute()
        if response['nextPageToken']:
            response += YouTubeClient.process_multiple_pages(
                self.youtube.subscriptions().list, response['nextPageToken'])
        subscriptions = [subscription['snippet']['channelId']
                         for subscription in response['items']]
        return subscriptions

    def get_playlist_videos(self, playlist_id):
        request = self.youtube.playlistItems().list(
            part="snippet", playlistId=playlist_id)
        response = request.execute()
        if response['nextPageToken']:
            response += YouTubeClient.process_multiple_pages(
                self.youtube.playlistItems().list, response['nextPageToken'])
        videos = [YouTubeClient.extract_video_data(
            video, 'playlistItem') for video in response['items']]
        return videos

    def get_subscription_newuploads(self, channel_id, time_ref):
        request = self.youtube.search().list(part="snippet", channelId=channel_id,
                                             publishedAfter=time_ref.isoformat(), type='video')
        # TODO: do I need to format date by appending 'T00:00:00.000Z'
        response = request.execute()
        if response['nextPageToken']:
            response += YouTubeClient.process_multiple_pages(
                self.youtube.search().list, response['nextPageToken'])
        videos = [YouTubeClient.extract_video_data(
            video, 'search') for video in response['items']]
        return videos

    def add_video_to_playlist(self, video_id, playlist_id):
        request = self.youtube.playlistItems().insert(part="snippet", body={'snippet': {
            'playlistId': playlist_id,
            'resourceId': video_id}})
        response = request.execute()
        return response

    def create_playlist(self, playlist_name):
        request = self.youtube.playlists().insert(
            body={'snippet': {'title': playlist_name}})
        response = request.execute()
        playlist_id = response['id']
        return playlist_id

    @staticmethod
    def extract_video_data(video, req_type):
        if req_type == 'search':
            v_id = video['id']['videoId']
        elif req_type == 'playlistItem':
            v_id = video['id']
        name = video['snippet']['title']
        return {
            'id': v_id,
            'name': name,
        }

    @staticmethod
    def process_multiple_pages(command, pageToken):
        res = command(pageToken=pageToken).execute()
        if res['nextPageToken']:
            return res + YouTubeClient.process_multiple_pages(command, res['nextPageToken'])
        return res
