from glob import glob

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

# Custom imports
import config
from utilities import make_project_dirs
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
    add_chords: list = []
    add_drums: bool
    set_bpm: int
    modify_length: int


@app.get("/")
def read_root():
    return "ok"


@app.post("/songs")
async def song(gen_type: str, song_request: SongRequest):
    print(gen_type, song_request)
    file_name, output_file_path = genKing.run_generation(gen_type, jsonable_encoder(song_request))
    print(file_name, output_file_path)
    return FileResponse(output_file_path)

@app.get("/songs/{song_id}")
def get_song(song_id):
    file_path = get_wav_by_name(song_id)
    return FileResponse(file_path)
