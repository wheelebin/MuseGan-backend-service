from glob import glob

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

# Custom imports
import config
from predictions import init_generator
from utilities import make_project_dirs
from music_generation_service import run_generation


make_project_dirs()

# Generator stuff
generator_file_path = glob(config.ROOT_DIR + "/models/LPD/final_check_tensor*")[0]
generator = init_generator(generator_file_path)

# FastAPI stuff
app = FastAPI()


@app.get("/")
def read_root():
    return "ok"


@app.post("/song")
async def song(request: Request):
    req_operations = await request.json()
    print(req_operations)
    output_file_path = run_generation(
        generator, req_operations

    )  # This should take a generator which is loaded on init in here
    print(output_file_path)
    return FileResponse(output_file_path)
