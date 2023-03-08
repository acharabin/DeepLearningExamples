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
# *****************************************************************************

# Import packages

from __Tacotron2.tacotron2.text import text_to_sequence
from __Tacotron2 import models
import torch
import argparse
import os
import numpy as np
from scipy.io.wavfile import write
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from waveglow.denoiser import Denoiser
from __Tacotron2.tacotron2 import data_function
import boto3
import configparser

import glob
import tqdm
import argparse
from omegaconf import OmegaConf
from __Tacotron2.univnet.model.generator import Generator
import re

def parse_args(parser):
    """
    Parse commandline arguments.
    """

    # File Path Parameters
    parser.add_argument('-i', '--input', type=str,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output',
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-v', '--vocoder', type=str, choices=['univnet','WaveGlow'], default='WaveGlow',
                        help='Choose whether to use univnet or WaveGlow to synthesize audio from mels')
    parser.add_argument('--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('--univnet', type=str,
                        help='full path to the univnet model checkpoint file')
    
    # Inference Configuration Parameters
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--use-ground-truth-mels', action='store_true',
                        help='Uses mel from audio instead of text to evaluate the WaveGlow model')
    parser.add_argument('--gtm-index', default=0, type=int,
                        help='Ground truth mel index')

    # Run Mode Parameters
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--fp16', action='store_true',
                        help='Run inference with mixed precision')
    run_mode.add_argument('--cpu', action='store_true',
                        help='Run inference on CPU')
    run_mode.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    run_mode.add_argument('--return-audio-vector', action='store_true',
                        help='When selected, returns the vector of synthesized audio instead of writing a wav file')

    # Dataset Parameters
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    parser.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')
    
    # Analytics Parameters
    parser.add_argument('--upload-to-s3', action='store_true',
                          help='Uploads inferences and alignments to s3')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')

    # Univnet Parameters
    parser.add_argument('-uc', '--univnet-config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")

    return parser


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, fp16_run, cpu_run, univnet_config, forward_is_infer=False):
    
    if model_name == 'univnet':
        if cpu_run:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint)
        
        if univnet_config is not None:
            hp = OmegaConf.load(univnet_config)
        else:
            hp = OmegaConf.create(checkpoint['hp_str'])

        model = Generator(hp).cuda()
        saved_state_dict = checkpoint['model_g']
        new_state_dict = {}

        for k, v in saved_state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict['module.' + k]
            except:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.eval(inference=True)

        return model

    model_parser = models.model_parser(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()
    
    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)

    model.eval()

    if fp16_run:
        model.half()

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cpu_run=False):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths


class MeasureTime():
    def __init__(self, measurements, key, cpu_run=False):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    if not 'args' in locals():
        parser = argparse.ArgumentParser(
            description='PyTorch Tacotron 2 Inference')
        parser = parse_args(parser)
        args, _ = parser.parse_known_args()
    
    if args.upload_to_s3:
        envparser = configparser.ConfigParser()
        envparser.read('.env')

    log_file = os.path.join(args.output, args.log_file)
    if not os.path.exists(args.output): os.makedirs(args.output)
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})
    
    if args.vocoder == 'univnet':
        univnet = load_and_setup_model('univnet', parser, args.univnet,
                                    args.fp16, args.cpu, args.univnet_config, forward_is_infer=True)

    else: 
        waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.fp16, args.cpu, None, forward_is_infer=True)
        denoiser = Denoiser(waveglow)
        if not args.cpu:
            denoiser.cuda()
    
    if args.use_ground_truth_mels:
        
        model_name = 'Tacotron2'
        parser = models.model_parser(model_name, parser)
        args, _ = parser.parse_known_args()

        args.text_cleaners = ['english_cleaners']
        args.load_mel_from_disk=False

        dataset=data_function.TextMelLoader('', args.input, True, args)
        
        audiopaths_and_text=data_function.load_filepaths_and_text('', args.input) 
        
        audiopath_and_text=audiopaths_and_text[args.gtm_index][0]

        mel = dataset.get_mel(audiopath_and_text)
        
        mel = mel.unsqueeze(0)
        
        print(mel.size())
        
        mel_lengths = [mel.size(2)]

        measurements={}

    else: 

        tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.fp16, args.cpu, None, forward_is_infer=True)

        jitted_tacotron2 = torch.jit.script(tacotron2)

        texts = []
        try:
            f = open(args.input, 'r')
            texts = f.readlines()
        except:
            print("Could not read file")
            sys.exit(1)

        if args.include_warmup:
            sequence = torch.randint(low=0, high=148, size=(1,50)).long()
            input_lengths = torch.IntTensor([sequence.size(1)]).long()
            if not args.cpu:
                sequence = sequence.cuda()
                input_lengths = input_lengths.cuda()
            for i in range(3):
                with torch.no_grad():
                    mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths)

        measurements = {}

        sequences_padded, input_lengths = prepare_input_sequence(texts, args.cpu)

        with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu):
            mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)

        tacotron2_infer_perf = mel.size(0)*mel.size(2)/measurements['tacotron2_time']
        DLLogger.log(step=0, data={"tacotron2_items_per_sec": tacotron2_infer_perf})
        DLLogger.log(step=0, data={"tacotron2_latency": measurements['tacotron2_time']})

    if args.vocoder == 'univnet':
        with torch.no_grad(): 
            audios = univnet.inference(mel)
            audios = audios.float()
            if len(audios.shape) == 1:
                audios=audios.unsqueeze(0)
    else:
        with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu):
            audios = waveglow(mel, sigma=args.sigma_infer)
            audios = audios.float()
        with torch.no_grad(), MeasureTime(measurements, "denoiser_time", args.cpu):
            audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)
        print("Stopping after",mel.size(2),"decoder steps")
    
        waveglow_infer_perf = audios.size(0)*audios.size(1)/measurements['waveglow_time']

        DLLogger.log(step=0, data={"waveglow_items_per_sec": waveglow_infer_perf})
        DLLogger.log(step=0, data={"waveglow_latency": measurements['waveglow_time']})
        DLLogger.log(step=0, data={"denoiser_latency": measurements['denoiser_time']})
        if args.use_ground_truth_mels == False:
            DLLogger.log(step=0, data={"latency": (measurements['tacotron2_time']+measurements['waveglow_time']+measurements['denoiser_time'])})
    
    if args.return_audio_vector:
        return audios

    for i, audio in enumerate(audios):
        
        if args.upload_to_s3:
            s3 = boto3.client('s3',aws_access_key_id=envparser.get('s3', 'aws_access_key_id'),
                            aws_secret_access_key=envparser.get('s3', 'aws_secret_access_key'))
            s3_bucket=envparser.get('s3', 'bucket')
            s3_prefix=envparser.get('s3', 'prefix')

        if args.use_ground_truth_mels == False:
            plt.imshow(alignments[i].float().data.cpu().numpy().T, aspect="auto", origin="lower")
            figure_path = os.path.join(args.output,"alignment_"+str(i)+args.suffix+".png")
            plt.savefig(figure_path)
            if args.upload_to_s3 == True:
                try:
                    s3.upload_file("/workspace/tacotron2/{}".format(figure_path),s3_bucket,"{}/{}".format(s3_prefix, figure_path))
                    print("Uploaded alignment to {}/{}/{}".format(s3_bucket, s3_prefix, figure_path))
                except Exception as e:
                    print(e)

        audio = audio[:mel_lengths[i]*args.stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        audio_path = os.path.join(args.output,"audio_"+str(i)+args.suffix+".wav")
        write(audio_path, args.sampling_rate, audio.cpu().numpy())

        if args.upload_to_s3 == True:
            try:
                s3.upload_file("/workspace/tacotron2/{}".format(audio_path),s3_bucket,"{}/{}".format(s3_prefix, audio_path))
                print("Uploaded inference to {}/{}/{}".format(s3_bucket, s3_prefix, audio_path))
            except Exception as e:
                print(e)

    DLLogger.flush()

if __name__ == '__main__':
    main()
