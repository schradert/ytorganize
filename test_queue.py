import time


def strikethrough(text):
    return ''.join([u'\u0336{}'.format(c) for c in text])


def process_daemon(title, label, vid_id, playlist_id, playlist_names):
    updates = {}
    inp = input(
        f'Is {label} the correct playlist for {title}? [yes/no/cancel]: ')
    if inp == 'yes':
        print('Thank you! Your playlist will be updated and the model trained.')
        updates['correct'] = True
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
            updates['different'] = playlist_names[inp-1]
        elif inp == i+2:
            print('Noted! Would you like to create a new playlist?')
            inp = input(
                'If yes, specify the name. Otherwise, just type no: ')
            if inp == 'no':
                print(
                    f'{title} has been skipped and the model won\'t be updated to reflect your choice.')
            else:
                print(f'Your playlist, \"{inp}\" will be created now.')
                updates['create'] = inp
                time.sleep(5000)
                print(
                    f'Playlist created. This video, \"{title}\", will be added to your \"{inp}\" playlist and the model trained thereafter.')
        else:
            print('Input invalid! Video skipped.')
    elif inp == 'cancel':
        print(
            f'{title} has been skipped and the model won\'t be updated to reflect your choice.')
    else:
        print(
            f'Input invalid! {title} has been skipped and the model won\'t be updated to reflect your choice.')
    return updates
