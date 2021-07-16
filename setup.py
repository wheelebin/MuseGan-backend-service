from distutils.core import setup

setup(
    name="museganKing",
    version="1.0.0",
    author="Joakim Allen",
    author_email="jln.allen95@gmail.com",
    # packages=['towelstuff', 'towelstuff.test'],
    url="https://github.com/wheelebin/MuseGan-backend-service/",
    description="Restful API for MuseGAN",
    install_requires=[
        # pip install [ torch==1.7.0+cu101 / torch==1.7.0+cpu ] -f https://download.pytorch.org/whl/torch_stable.html
        "torch == 1.7.0+cpu",  # +cpu=cpu_only, cu101=cuda
        "matplotlib",
        "tqdm",
        "livelossplot",
        "gdown",
        "aiofiles",
        # "mido",
        "music21",
        "redis",
        "fastapi",
        "uvicorn",
        "firebase-admin",
        "midi2audio",
        "ipywidgets",
        "gunicorn",
        "pypianoroll >= 1.0.2",
        # "absl-py",
        "jsonpickle",
        # Bellow is from musegan tensorflow
        "absl-py",  # == 0.4.1",
        "astor",  # == 0.7.1",
        "gast",  # == 0.2.0",
        "grpcio",  # == 1.14.2",
        "imageio",  # == 2.3.0",
        "Markdown",  # == 2.6.11",
        "mido",  # == 1.2.8",
        # "numpy == 1.15",  # 1.14.5",
        "Pillow",  # == 5.2.0",
        "pretty-midi",  # == 0.2.8",
        "protobuf",  # == 3.6.1",
        # "pypianoroll == 0.4.6",
        "PyYAML",  # == 3.13",
        "scipy",  # == 1.1.0",
        "SharedArray == 3.0.0",
        "six",  # == 1.11.0",
        "tensorboard",  # == 1.10.0",
        "tensorflow == 1.15",  # == 1.10.1",  # tensorflow-gpu
        "termcolor",  # == 1.1.0",
        "Werkzeug",  # == 0.14.1",
    ],
)
