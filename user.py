import uuid

user_id_list = [] # { id: "1234-3123d-a23-sa213" }
tracks_list = [] # { file_name: "dasd", output_midi_filename: "/dasds/asd/qewqe.mid", user_id: "1234-3123d-a23-sa213"  }


def add_user():
  id = uuid.uuid4()
  user_id_list.append({ 'id': str(id) })
  return id

def get_user(user_id):
  print(user_id_list)
  for user in user_id_list:
    if user.get('id', None) == user_id:
      return user

  return None

def delete_user(user_id):
  for i, user in enumerate(user_id_list):
    if user.get('id', None) == user_id:
      del user_id_list[i:i+1]
      return True
  return False




def add_track(user_id, file_name, output_midi_filename):
  tracks_list.append({
    file_name: file_name,
    output_midi_filename: output_midi_filename,
    user_id: user_id
  })
  return True

def delete_track(user_id, file_name):
  for i, track in enumerate(tracks_list):
    if track.get(file_name, None) == file_name and track.get(user_id, None) == user_id:
      del tracks_list[i:i+1]
      return True
  return False

def get_tracks_by_user_id(user_id):
  tracks = []
  for track in tracks_list:
    if track.get(user_id, None) == user_id:
      tracks.append(track)
  return tracks

def get_track_by_file_name(file_name):
  for track in tracks_list:
    if track.get(file_name, None) == file_name:
      return track
  return None
