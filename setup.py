from distutils.core import setup

setup(
    name="MuseGAN - Restful API",
    version="1.0.0",
    author="Joakim Allen",
    author_email="jln.allen95@gmail.com",
    # packages=['towelstuff', 'towelstuff.test'],
    url="https://github.com/wheelebin/MuseGan-backend-service/",
    description="Restful API for MuseGAN",
    install_requires=[
        # pip install [ torch==1.7.0+cu101 / torch==1.7.0+cpu ] -f https://download.pytorch.org/whl/torch_stable.html
        "torch == 1.7.0+cpu", # +cpu=cpu_only, cu101=cuda
        "matplotlib",
        "tqdm",
        "livelossplot",
        "gdown",
        "aiofiles",
        "mido",
        "music21",
        "fastapi",
        "uvicorn",
        "midi2audio",
        "ipywidgets",
        "pypianoroll >= 1.0.2",
    ],
)
