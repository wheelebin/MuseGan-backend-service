from glob import glob

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

# Custom imports
import config
from utilities import get_file_name_for_saving, make_project_dirs
from music_generation_service import get_wav_by_name, MusicGenKing

from pydantic import BaseModel
from typing import Optional


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


@app.post("/songs")
async def song(gen_type: str, song_request: SongRequest):
    ops = {}
    if gen_type == 'ai1' or gen_type == 'ai2':
        ops = song_request
    elif gen_type == 'ai3' or gen_type == 'ai4':
        ops["genre"] = song_request.genre
        ops["modify_length"] = song_request.modify_length
    else:
        return FileResponse()

    file_name, output_file_path = genKing.run_generation(gen_type, jsonable_encoder(ops))
    return FileResponse(output_file_path)

@app.get("/songs/{song_id}")
def get_song(song_id):
    file_path = get_wav_by_name(song_id)
    return FileResponse(file_path)
