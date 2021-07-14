import uuid
import firebase_admin
from firebase_admin import credentials, auth
from config import ROOT_DIR


user_id_list = [] # { id: "1234-3123d-a23-sa213", firebase_uid: "23123-312-3-123-21-3-32134" }
tracks_list = [] # { file_name: "qewqe", output_midi_filename: "/dasds/asd/qewqe.mid", user_id: "1234-3123d-a23-sa213"  }

cred = credentials.Certificate(ROOT_DIR+"/music-generator-9578a-firebase-adminsdk-sa6xv-7dd5a859e8.json")
firebase_admin.initialize_app(cred)

class FireBaseTokenRevokedError(Exception):
  pass

class FireBaseTokenInvalidError(Exception):
  pass

def validate_firebase_token(id_token):
  try:
      # Verify the ID token while checking if the token is revoked by
      # passing check_revoked=True.
      decoded_token = auth.verify_id_token(id_token, check_revoked=True)
      # Token is valid and not revoked.
      uid = decoded_token['uid']
      return uid
  except auth.RevokedIdTokenError:
      # Token revoked, inform the user to reauthenticate or signOut().
      raise FireBaseTokenRevokedError
  except auth.InvalidIdTokenError:
      # Token is invalid
      raise FireBaseTokenInvalidError

def add_user(id_token):
  id = uuid.uuid4()

  #uid = validate_firebase_token(id_token)

  user_id_list.append({ 'id': str(id) }) #, 'firebase_uid': uid
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
