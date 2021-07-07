from glob import glob

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

# Custom imports
import config
from utilities import get_file_name_for_saving, make_project_dirs
from music_generation_service import get_wav_by_name, MusicGenKing

from pydantic import BaseModel
from typing import Optional

import user as user_srvc


make_project_dirs()

# Generator stuff
generator_file_path = glob(config.CHECKPOINT_PATH + "/tensor_*")[0]
genKing = MusicGenKing(generator_file_path)

# FastAPI stuff
app = FastAPI()

class ChangeInstruments(BaseModel):
    track_1: Optional[int] = None
    track_2: Optional[int] = None
    track_3: Optional[int] = None
    track_4: Optional[int] = None
    track_5: Optional[int] = None

class ChangeInstruments(BaseModel):
    track_1: Optional[int] = None
    track_2: Optional[int] = None
    track_3: Optional[int] = None
    track_4: Optional[int] = None
    track_5: Optional[int] = None

class SongRequest(BaseModel):
    change_instruments: Optional[ChangeInstruments] = None
    add_chords: Optional[list] = []
    add_drums: Optional[bool]
    set_bpm: Optional[int]
    modify_length: Optional[int]
    genre: Optional[str]


@app.get("/")
def read_root():
    return "ok"




@app.post("/users")
async def add_user():
    user_id = user_srvc.add_user()
    return {'user_id': user_id}

@app.get("/users/{user_id}/tracks")
async def get_user_tracks(user_id: str):
    tracks = user_srvc.get_tracks_by_user_id(user_id)
    return tracks

@app.get("/tracks/{file_name}")
async def get_track(file_name: str):
    track = user_srvc.get_track_by_file_name(file_name)
    return track

@app.post("/songs")
async def song(gen_type: str, user_id: str, song_request: SongRequest):

    user = user_srvc.get_user(user_id)
    if user == None:
        return HTTPException(status_code=400, detail="User id is invalid!")

    ops = {}
    if gen_type == 'ai1' or gen_type == 'ai2':
        ops = song_request
    elif gen_type == 'ai3' or gen_type == 'ai4':
        ops["genre"] = song_request.genre
        ops["modify_length"] = song_request.modify_length
    else:
        return HTTPException(status_code=400, detail="Sound not created")

    encodedOps = jsonable_encoder(ops)

    file_name, output_file_path = genKing.run_generation(gen_type, encodedOps)

    track_added = user_srvc.add_track(user.get('id', None), file_name, output_file_path)

    if track_added:
        return { 'file_name': file_name }
        #return user_srvc.get_track_by_file_name(file_name)

    return HTTPException(status_code=400, detail="Sound not created")

@app.get("/songs/{song_id}")
def get_song(song_id):
    file_path = get_wav_by_name(song_id)
    return FileResponse(file_path)
