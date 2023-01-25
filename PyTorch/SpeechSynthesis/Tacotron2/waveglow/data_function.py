# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\

import torch
import tacotron2_common.layers as layers
from tacotron2_common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
import os
import math

class MelAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, trainset, args):
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        self.segment_length = args.segment_length
        self.trainset = trainset
        self.hop_length = args.hop_length

        # predicted mels
        self.use_predicted_mels = args.use_predicted_mels
        self.output_audio_path = args.output_audio_path
        self.output_mel_path = args.output_mel_path
        self.prefix_audio = args.prefix_audio
        self.prefix_mel = args.prefix_mel
        
    def get_mel_audio_pair(self, index):

        filename=self.audiopaths_and_text[index][0]

        audio, sampling_rate = load_wav_to_torch(filename)

        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = torch.randint(0, max_audio_start + 1, size=(1,)).item()
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.size(0)), 'constant').data

        audio = audio / self.max_wav_value
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.squeeze(0)

        return (melspec, audio, len(audio))

    def get_predicted_mel_audio_segment_pair(self, index):

        if self.trainset:
            fileprefix='train'
        else: fileprefix='validation'

        filename_audio = self.prefix_audio + str(index + 1) + '.pt'
        filename_mel = self.prefix_mel + str(index + 1) + '.pt'

        audio=torch.load(os.path.join(self.output_audio_path,fileprefix,filename_audio), map_location='cpu')
        melspec=torch.load(os.path.join(self.output_mel_path,fileprefix,filename_mel), map_location='cpu').squeeze(0)

        return (melspec, audio, len(audio))

    def get_mel_audio_pair_function(self, index):
        if self.use_predicted_mels:
            return self.get_predicted_mel_audio_segment_pair(index)
        else:
            return self.get_mel_audio_pair(index)

    def get_mel_audio_segments(self, index, use_predicted_mel, predicted_mel, max_segments):

        filename=self.audiopaths_and_text[index][0]

        audio, sampling_rate = load_wav_to_torch(filename)

        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
 
        # Get Segments
        segments_audio=math.ceil(audio.size(0)/self.segment_length)
        
        if use_predicted_mel:
            predicted_mel=predicted_mel.squeeze(0)
            #print('starting predicted mel size: '+str(predicted_mel.size()))

            segments_mel_unrounded=(predicted_mel.size(1)*self.hop_length)/self.segment_length
            if segments_mel_unrounded - math.floor(segments_mel_unrounded) < self.hop_length/self.segment_length:
                segments_mel=math.floor(segments_mel_unrounded)
            else: segments_mel=math.ceil(segments_mel_unrounded)

            print('mel frames: '+str(predicted_mel.size(1)*self.hop_length))
            print('audio frames: '+str(audio.size(0)))
            if audio.size(0)>predicted_mel.size(1)*self.hop_length:
                print('warning: audio length greater than predicted mel length at input audio file {}'.format(filename))
                if segments_mel>segments_audio:
                    print('warning: audio segments exceeds mel segments so an empty mel segment may be added to input audio file {}'.format(filename))
            
            if segments_audio != segments_mel: 
                print('warning: mel segments count of {} differ from audio segments count of {}'.format(segments_mel,segments_audio))
                if segments_mel*self.segment_length - audio.size(0) >= self.segment_length:
                    print('warning: forecasting padded mel a full segment greater than audio at input audio file {}'.format(filename))
                    mel=predicted_mel[0: int((predicted_mel.size(1)*self.hop_length - self.segment_length)/self.hop_length)]
                    segments_mel=math.floor(segments_mel_unrounded)
                    print('mel segment at input audio file {} floored to {}'.format(filename, segments_mel))
                
            segments=min(segments_mel, max_segments)

        audio_segments = []
        mel_segments = []
        audio_norm = []
        segment_length = []

        for i in range(segments):
            start = self.segment_length * i
            if audio.size(0) >= start + self.segment_length:
                audio_segment_add = audio[start:(start + self.segment_length)]
            else:
                print('audio segment length before padding: '+str(audio[start:].size(0)))
                print('padding length added to audio segment: '+str(start + self.segment_length - audio.size(0)))
                audio_segment_add = torch.nn.functional.pad(audio[start:], (0, start + self.segment_length - audio.size(0)), 'constant').data
            audio_segments.append(audio_segment_add)
            segment_length.append(audio_segments[i].size(0))

            audio_segments[i] = audio_segments[i] / self.max_wav_value

            if use_predicted_mel: 
                if predicted_mel.size(1) >= start/self.hop_length:
                    if predicted_mel.size(1) < int((start + self.segment_length)/self.hop_length):
                        mel_segment = predicted_mel[:,int(start/self.hop_length):predicted_mel.size(1)]
                        mel_segment_padded = torch.Tensor(mel_segment.size(0), int(self.segment_length/self.hop_length))
                        mel_segment_padded.zero_()
                        for i in range(mel_segment.size(0)):
                            mel_channel = mel_segment[i]
                            mel_segment_padded[i, :mel_channel.size(0)] = mel_channel
                            mel_segment_add = mel_segment_padded
                        print('mel segment size before padding: '+str(mel_segment.size()))
                        print('mel segment size after padding: '+str(mel_segment_padded.size()))
                    else:
                        mel_segment_add = predicted_mel[:,int(start/self.hop_length):int((start + self.segment_length)/self.hop_length)]
                else:
                    mel_segment_add = torch.zeros(1,predicted_mel.size(0),predicted_mel.size(1))
                    print('warning: zeroed mel segment added at segment {} for file {}'.format(i, filename))
                mel_segments.append(mel_segment_add)
                print(mel_segment_add)
            else:
                audio_segments.append(audio_segment_add)
                audio_norm.append(audio_segments[i].unsqueeze(0))
                audio_norm[i] = torch.autograd.Variable(audio_norm[i], requires_grad=False)
                mel_segments.append(self.stft.mel_spectrogram(audio_norm[i]))
                mel_segments[i] = mel_segments[i].squeeze(0)

        return (mel_segments, audio_segments, segment_length)

    def __getitem__(self, index):
        return self.get_mel_audio_pair_function(index)

    def __len__(self):
        return len(self.audiopaths_and_text)


def batch_to_gpu(batch):
    x, y, len_y = batch
    x = to_gpu(x).float()
    y = to_gpu(y).float()
    len_y = to_gpu(torch.sum(len_y))
    return ((x, y), y, len_y)
