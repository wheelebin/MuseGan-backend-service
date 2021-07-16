import uuid
import firebase_admin
from firebase_admin import credentials, auth
from config import ROOT_DIR
from struct import error
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

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
      decoded_token = auth.verify_id_token(id_token)
      uid = decoded_token['uid']
      return uid
  except:
    raise

def get_or_add_user_by_id_token(id_token):

  try:
      uid = validate_firebase_token(id_token)
  except auth.RevokedIdTokenError:
      return None, "Revoked id token!"
  except auth.InvalidIdTokenError:
      return None, "Invalid id token!"
  except ValueError:
      return None, "Id token can't be empty!"

  if uid == None:
      return None, "id token is not valid"


  user = get_or_add_user(uid), None

  return user
  

def get_or_add_user(uid):

  user = get_user(uid)

  if user == None:
    user = add_user(uid)

  return user


def r_get(key):
  result = r.get(key)
  if result == None:
    return None
  return json.loads(result)
  
def r_set(key, value):
  r.set(key, json.dumps(value))













def get_all(key, create_if_none=True):
  list = r_get(key)

  if list == None and create_if_none == True:
    r_set(key, [])
    return []

  return list

def get(key, id_key, id_val):

  list = get_all(key)

  for list_item in list:
    if list_item.get(id_key, None) == id_val:
      return list_item

  return None


def add(key, value, id_key=None):


  list = get_all(key)

  if id_key != None:
    for list_item in list:
      if list_item[id_key] == value[id_key]:
        return None

  list.append(value)

  r_set(key, list)

  return value



def delete(key, id_key, id_val):
  list = get_all(key)
  for i, list_item in enumerate(list):
    if list_item.get(id_key, None) == id_val:
      del list[i]
      r_set(key, list)
      return True
  return False









def get_users():
  return get_all('users')

def get_user(uid):
  return get('users', 'uid', uid)

def add_user(uid):
  new_user = { 'uid': str(uid) }
  return add('users', new_user, 'uid')

def delete_user(uid):
  return delete('users', 'uid', uid)






def add_track(uid, file_name, output_midi_filename):
  add('tracks', {
    'file_name': file_name,
    'output_midi_filename': output_midi_filename,
    'uid': uid
  })
  return True

def delete_track(uid, file_name):
  list = get_all('tracks')
  for i, list_item in enumerate(list):
    if list_item.get('file_name', None) == file_name and list_item.get('uid', None) == uid:
      del list[i]
      print(list)
      r_set("tracks", list)
      return True
  return False

def get_tracks_by_uid(uid):
  tracks = []
  for track in get_all('tracks'):
    if track.get('uid', None) == uid:
      tracks.append(track)
  return tracks

def get_track_by_file_name(file_name):
  for track in get_all('tracks'):
    if track.get('file_name', None) == file_name:
      return track
  return None
