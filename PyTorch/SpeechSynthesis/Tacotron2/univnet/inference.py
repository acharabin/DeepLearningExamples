import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
from omegaconf import OmegaConf

from model.generator import Generator
import re

def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = OmegaConf.load(args.config)
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

    with torch.no_grad():
        #for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.mel'))):
        for i, melfile in enumerate(os.listdir(args.input_folder),start=1):
            filenumber=int(re.search('_(.*).pt', melfile).group(1))
            if filenumber >= args.file_index_min and filenumber <= args.file_index_max:
                melpath=os.path.join(args.input_folder,melfile)
                mel = torch.load(melpath)
                if len(mel.shape) == 2:
                    mel = mel.unsqueeze(0)
                mel = mel.cuda()

                audio = model.inference(mel)
                audio = audio.cpu().detach().numpy()

                if args.output_folder is None:  # if output folder is not defined, audio samples are saved in input folder
                    out_path = melpath.replace('.pt', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
                else:
                    basename = os.path.basename(melpath)
                    basename = basename.replace('.pt', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
                    out_path = os.path.join(args.output_folder, basename)
                write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio")
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help="directory to output inferences")
    parser.add_argument('-min', '--file-index-min', type=int, default=1,
                        help="first file index in input directory to use")
    parser.add_argument('-max', '--file-index-max', type=int, default=1000000,
                        help="last file index in input directory to use")
    args = parser.parse_args()

    main(args)
