#!/bin/bash

nvidia-docker run -d --name test_tacotron2 --publish 8000:8000 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/workspace/tacotron2/ tacotron2 bash
