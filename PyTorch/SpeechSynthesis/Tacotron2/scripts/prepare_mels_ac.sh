#!/usr/bin/env bash

set -e

DATADIR="AC-Voice-Cloning-Data"
FILELISTSDIR="AC-Voice-Cloning-Data/filelists"

TRAINLIST="$FILELISTSDIR/audio/acs_audio_text_train_filelist.txt"
VALLIST="$FILELISTSDIR/audio/acs_audio_text_validation_filelist.txt"

TRAINLIST_MEL="$FILELISTSDIR/mel/acs_mel_text_train_filelist.txt"
VALLIST_MEL="$FILELISTSDIR/mel/acs_mel_text_validation_filelist.txt"

mkdir -p "$DATADIR/mels"
if [ $(ls $DATADIR/mels | wc -l) -ne 13100 ]; then
    python preprocess_audio2mel.py --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL" --mel-fmin 0 --mel-fmax 8000 --n-mel-channels 80
    python preprocess_audio2mel.py --wav-files "$VALLIST" --mel-files "$VALLIST_MEL" --mel-fmin 0 --mel-fmax 8000 --n-mel-channels 80	
fi	
