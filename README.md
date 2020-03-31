# ytorganize

If you're like me and follow hundreds of amazing YouTube content creators but don't always have the time to process each of your favorite channels' recent editions, this repository is for you.

YTORGANIZE trains a simple neural network to classify recent uploads from your subscribed channels into your channel's many private playlists based on video name.

Don't worry, playlists are only actually updated when approved by the user, the previous day's uploads being funneled into an approval queue after prediction. Once all confirmed, denied, corrected, or skipped, YouTube API POST updates are made and the model retrained.
