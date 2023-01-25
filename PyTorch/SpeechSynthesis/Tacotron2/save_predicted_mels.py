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

from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import os
import numpy as np
from scipy.io.wavfile import write
import matplotlib
import matplotlib.pyplot as plt
import re
import sys
import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from waveglow.denoiser import Denoiser
import preprocess_audio2mel
from tacotron2_common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
import os
from waveglow.data_function import MelAudioLoader

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='path to dataset')
    parser.add_argument('-i','--input-path', type=str, required=True,
                        help='full path to filelist with audio paths and text; audio will only be used in --split-segments mode')
    parser.add_argument('--file-index-min', default=0, type=int,
                        help='first file index from input path to process')
    parser.add_argument('--file-index-max', default=100000, type=int,
                        help='last file index from input path to process; default value processes all')
    parser.add_argument('--input-path-subset', type=str, default='AC-Voice-Cloning-Data/filelists/audio/acs_audio_text_train_filelist_subset.txt',
                        help='full path to filelist with audio paths and text; audio will only be used in --split-segments mode')
    parser.add_argument('-oa', '--output-audio-path', default='AC-Voice-Cloning-Data/wavs/predicted',
                        help='output folder to save predicted audio segments in --split-segments mode')
    parser.add_argument('-om', '--output-mel-path', default='AC-Voice-Cloning-Data/mels/predicted',
                        help='output folder to save predicted mels')
    parser.add_argument('-t','--train', action='store_true', help='adds train/ subfolder in output path')
    parser.add_argument('-v', '--validation', action='store_true', help='adds validation/ subfolder in output path')
    parser.add_argument('-pa','--prefix-audio', default='AC_audio_segment_',
                        help='prefix used when saving audio segments afterwhich the index is pasted')
    parser.add_argument('-pm','--prefix-mel', default='AC_predicted_mel_segment_',
                        help='prefix used when saving predicted mels afterwhich the index is pasted')
    #parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    #parser.add_argument('--waveglow', type=str,
                        #help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    #parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='sampling rate')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='filename for logging')
    parser.add_argument('--skip-if-exists', action='store_true', help='if predicted mel file exists already, skip to the next mel; use --resume-from-last instead if --split-segments is used')
    parser.add_argument('--split-segments', action='store_true', help='splits audio by segment length and saves each segment as a separate file')
    parser.add_argument('--max-segments', default=100, type=int, help='enforces a max number of audio file segments when --split-segments is used')
    parser.add_argument('--filelist-output-path', default='AC-Voice-Cloning-Data/filelists/audio/acs_audio_segment_text_train_filelist.txt',
                        help='if --split-segments mode is used, output folder to save generated text/audio segment filelist for WaveGlow training')
    parser.add_argument('--reset-filelist', action='store_true', help='remove filelist at --filelist-output-path before appending new lines')
    parser.add_argument('--empty-output-path', action='store_true', help='removes all files from audio and mel output directories')
    parser.add_argument('--test', action='store_true', help='doesnt save to output path and prevents file loss')
    parser.add_argument('--resume-from-last', action='store_true', help='continues saving audio and mel from the last file in the output directory; any segments from the last file are overwritter')

    # Required for Tacotron Data Loader
    parser.add_argument('--use-predicted-mels',
                         action='store_true',
                         help='loads predicted mel tensors from the predicted mel path instead of computing mels from ground truth audio')
    parser.add_argument('--predicted-mel-path',
                         default='AC-Voice-Cloning-Data/mels/predicted/',
                         type=str, help='Path to training filelist. Note that train/ or validation/ will be joined to this path depending on the data loader')
    parser.add_argument('--predicted-mel-prefix', default='AC_predicted_mel_segment_',
                        help='prefix used when saving predicted mels (i.e. AC_predicted_mel_) afterwhich the index is pasted')
    
    parser.add_argument('--no-padding', action='store_true',
                        help='Forces no padding of input passages')

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--fp16', action='store_true',
                        help='run inference with mixed precision')
    run_mode.add_argument('--cpu', action='store_true',
                        help='run inference on CPU')

    #parser.add_argument('--include-warmup', action='store_true',
                        #help='Include warmup')
    #parser.add_argument('--stft-hop-length', type=int, default=256,
                        #help='STFT hop length for estimating audio length from mel size')
    #parser.add_argument('--text-cleaners', nargs='*',
                         #default=['english_cleaners'], type=str,
                         #help='Type of text cleaners for input text')
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
    #parser.add_argument('--load-mel-from-disk', action='store_true',
                       #help='Loads mel spectrograms from disk instead of computing them on the fly')
    #parser.add_argument('--n-mel-channels', default=80, type=int,
                        #help='Number of bins in mel-spectrograms')
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


def load_and_setup_model(model_name, parser, checkpoint, fp16_run, cpu_run, forward_is_infer=False):
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
    
    #print(d)
    
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
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = os.path.join(args.log_file)
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2, args.fp16, args.cpu, forward_is_infer=True)
    #waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    #args.fp16, args.cpu, forward_is_infer=True)
    #denoiser = Denoiser(waveglow)
    #if not args.cpu:
        #denoiser.cuda()

    jitted_tacotron2 = torch.jit.script(tacotron2)
    
    #if args.resume_from_last:
        #with open(args.input_path, 'r') as f:
            #lines=f.readlines()
            #for line in lines:
                #file_number=re.search('_(.*).wav', line).group(1) 
    #return print(file_number)

    audiopaths_and_text = load_filepaths_and_text(args.dataset_path, args.input_path)
    
    audiopaths_and_text = audiopaths_and_text[args.file_index_min:]
    
    data_loader_path=args.input_path

    if len(audiopaths_and_text) < 1249: 
        with open(args.input_path_subset, 'w') as f:
            for line in audiopaths_and_text:
                f.write(str(line[0])+'|'+str(line[1])+'\n')
        data_loader_path=args.input_path_subset
        f.close()

    if args.file_index_max != 100000:
        audiopaths_and_text = audiopaths_and_text[:args.file_index_max + 1 - args.file_index_min]
    
    if args.no_padding:
       
        sequences_padded=[None]*len(audiopaths_and_text)
        input_lengths=[None]*len(audiopaths_and_text)
        mels=[None]*len(audiopaths_and_text)
        mel_lengths=[None]*len(audiopaths_and_text)
        alignments=[None]*len(audiopaths_and_text)
        measurements = {}
        
        for i, audiopath_and_text in enumerate(audiopaths_and_text):
            text=[]
            text.append(audiopath_and_text[1])
            sequences_padded[i], input_lengths[i] = prepare_input_sequence(text, args.cpu)
            with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu):
                mels[i], mel_lengths[i], alignments[i] = jitted_tacotron2(sequences_padded[i], input_lengths[i])
    
        
        #return print(mels[0].size())

    #for text in enumerate(args.input_path):
        #texts = []
        #try:
            #f = open(args.input, 'r')
            #texts.append(f.readlines())
        #except:
            #print("Could not read file")
            #sys.exit(1)

    #if args.include_warmup:
        #sequence = torch.randint(low=0, high=148, size=(1,50)).long()
        #input_lengths = torch.IntTensor([sequence.size(1)]).long()
        #if not args.cpu:
            #sequence = sequence.cuda()
            #input_lengths = input_lengths.cuda()
        #for i in range(3):
            #with torch.no_grad():
                #mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths)
                #_ = waveglow(mel)

    else: 
        
        measurements = {}

        sequences_padded, input_lengths = prepare_input_sequence(texts, args.cpu)

        with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu):
            mels, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)
    
        #return print(mels[0].size())

    if args.train:
        subfolder='/train'
    elif args.validation:
        subfolder='/validation'
    else: subfolder=''

    if args.split_segments:
        
        model_name = 'WaveGlow'
        parser = models.model_parser(model_name, parser)
        args, _ = parser.parse_known_args()
        
        dataset=MelAudioLoader(args.dataset_path, data_loader_path, True, args)
        
        if args.reset_filelist and not args.test:
            if os.path.exists(args.filelist_output_path):
                os.remove(args.filelist_output_path)
                print('Reset filelist {}'.format(args.filelist_output_path))

        audio_directory=args.output_audio_path+subfolder
        mel_directory=args.output_mel_path+subfolder

        if args.empty_output_path and not args.test:
            if os.path.exists(audio_directory):
                for filename in os.listdir(audio_directory):
                    os.remove(os.path.join(audio_directory,filename))
                print('Emptied directory {}'.format(audio_directory))
            if os.path.exists(mel_directory):
                for filename in os.listdir(mel_directory):
                    os.remove(os.path.join(mel_directory,filename))
                print('Emptied directory {}'.format(mel_directory))

        counter=0
        lines=[]
        
        for i in range(len(audiopaths_and_text)):
            
            text=audiopaths_and_text[i][1]

            file_number=int(re.search('_(.*).wav', audiopaths_and_text[i][0]).group(1))

            print('starting file number {}'.format(str(file_number)))

            mel_segments, audio_segments, segment_lengths = dataset.get_mel_audio_segments(i, True, mels[i], args.max_segments)
            
            print('{} audio and mel segments pairs generated for file number {}'.format(len(mel_segments),str(file_number)))

            #print(mel_segments[1].size())
            #print(audio_segments[1].size())
            #print(segment_lengths[1])

            #print(len(mel_segments))
            #print(len(audio_segments))

            #return 1
        
            new_lines=[]

            for m, mel_segment in enumerate(mel_segments):
                
                audio_segment=audio_segments[m]
                segment_length=segment_lengths[m]
                
                if segment_length != args.segment_length:
                    print(segment_length)
                    return print('error: audio segment length not equal to args.segment_length')

                audio_segment=audio_segments[m]

                counter += 1
                audio_filename=args.prefix_audio+str(counter)+'.pt'
                mel_filename=args.prefix_mel+str(counter)+'.pt'

                audio_path=os.path.join(audio_directory,audio_filename)
                mel_path=os.path.join(mel_directory,mel_filename)

                new_lines.append(audio_path+'|'+text+'\n')
                
                if not args.test:
                    torch.save(audio_segment, audio_path)
                    print("Saved audio segment {} to {}".format(str(counter), audio_path))
                
                    torch.save(mel_segment, mel_path)
                    print("Saved predicted mel segment {} to {}".format(str(counter), mel_path))

            if not args.test:
                with open(args.filelist_output_path, 'a+') as f:
                    for new_line in new_lines:
                        f.write(new_line.encode('ascii', errors='ignore').decode())
                f.close()
                print("Appended lines {}:{} to {}".format(str(counter - (m + 1)), str(counter), args.filelist_output_path))

    #return print(mels[0])
    else: 
        for i, mel in enumerate(mels,start=1):
            filename=args.prefix_mel+str(i)+'.pt'

            mel_path=os.path.join(args.output_mel_path+subfolder,filename)
            if args.skip_if_exists:
                if not os.path.exists(fullpath) and not args.test:
                    torch.save(mel,fullpath)
                    print("Saved predicted mel {} to {}".format(str(i), fullpath))
            else:
                if not args.test:
                    torch.save(mel,fullpath)
                    print("Saved predicted mel {} to {}".format(str(i), fullpath))

    #with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu):
        #audios = waveglow(mel, sigma=args.sigma_infer)
        #audios = audios.float()
    #with torch.no_grad(), MeasureTime(measurements, "denoiser_time", args.cpu):
        #audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    #print("Stopping after",mel.size(2),"decoder steps")
    #tacotron2_infer_perf = mel.size(0)*mel.size(2)/measurements['tacotron2_time']
    #waveglow_infer_perf = audios.size(0)*audios.size(1)/measurements['waveglow_time']

    #DLLogger.log(step=0, data={"tacotron2_items_per_sec": tacotron2_infer_perf})
    #DLLogger.log(step=0, data={"tacotron2_latency": measurements['tacotron2_time']})
    #DLLogger.log(step=0, data={"waveglow_items_per_sec": waveglow_infer_perf})
    #DLLogger.log(step=0, data={"waveglow_latency": measurements['waveglow_time']})
    #DLLogger.log(step=0, data={"denoiser_latency": measurements['denoiser_time']})
    #DLLogger.log(step=0, data={"latency": (measurements['tacotron2_time']+measurements['waveglow_time']+measurements['denoiser_time'])})

    #for i, audio in enumerate(audios):

        #plt.imshow(alignments[i].float().data.cpu().numpy().T, aspect="auto", origin="lower")
        #figure_path = os.path.join(args.output,"alignment_"+str(i)+args.suffix+".png")
        #plt.savefig(figure_path)

        #audio = audio[:mel_lengths[i]*args.stft_hop_length]
        #audio = audio/torch.max(torch.abs(audio))
        #audio_path = os.path.join(args.output,"audio_"+str(i)+args.suffix+".wav")
        #write(audio_path, args.sampling_rate, audio.cpu().numpy())

    #DLLogger.flush()

if __name__ == '__main__':
    main()
