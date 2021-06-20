from glob import glob
from midi_utilities import change_instruments

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

import jsonpickle

# Custom imports
import config
from predictions import init_generator
from utilities import make_project_dirs
from music_generation_service import run_generation, get_wav_by_name

from pydantic import BaseModel
from typing import Optional


make_project_dirs()

# Generator stuff
generator_file_path = glob(config.ROOT_DIR + "/models/LPD/*_tensor*")[0]
generator = init_generator(generator_file_path)

# FastAPI stuff
app = FastAPI()

class ChangeInstruments(BaseModel):
    track_1: Optional[int] = None
    track_2: Optional[int] = None
    track_3: Optional[int] = None
    track_4: Optional[int] = None
    track_5: Optional[int] = None

class SongRequest(BaseModel):
    change_instruments: Optional[ChangeInstruments] = None
    add_chords: list = []


@app.get("/")
def read_root():
    return "ok"


@app.post("/songs")
async def song(song_request: SongRequest):
    output_file_path = run_generation(
        generator, jsonable_encoder(song_request)

    )  # This should take a generator which is loaded on init in here
    print(output_file_path)
    return FileResponse(output_file_path)

@app.get("/songs/{song_id}")
def get_song(song_id):
    file_path = get_wav_by_name(song_id)
    return FileResponse(file_path)
