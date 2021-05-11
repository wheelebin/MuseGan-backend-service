from glob import glob 

from fastapi import FastAPI
from fastapi.responses import FileResponse

# Custom imports
import config
from predictions import predict, init_generator
from utilities import make_project_dirs


make_project_dirs()

app = FastAPI()

# Generator stuff
generator_file_path = glob(config.ROOT_DIR + "/models/LPD/final_check_tensor*")[0]
generator = init_generator(generator_file_path)

@app.get("/")
def read_root():
    return "ok"


@app.get("/song")
def song():
    predict(generator)  # This should take a generator which is loaded on init in here
    return FileResponse("test.wav")
