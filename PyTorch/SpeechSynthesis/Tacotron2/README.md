# Tacotron 2 And WaveGlow For PyTorch

This repository provides additions to NVIDIA/DeepLearningExamples scripts to further optimize training and inferences with Tacotron 2 and WaveGlow v1.6 models. More fundamentally, this repository is designed to provide a recipe for not just how to train a TTS model on an existing open source dataset (i.e. LJ voice), but to be able to easily scale application to new voices through tools for script management, audio editing, transfer learning, and more. As such, the repo comes equipped with a new open source voice dataset (AC voice) and associated models/inferences as a proof of concept.

## Source Repo

[Source README.md](https://github.com/acharabin/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/README_original.md)

DeepLearningExamples/.../Tacotron2 is the latest Pytorch implementation of Tacotron2 managed by NVIDIA and has several benefits over NVIDIA's independant Tacotron2 and WaveGlow repositories. 
- Common code for Tacotron2 and WaveGlow is consolidated
   * This reduces administration time for updates to common code and allows additional functionality (i.e. warm start) to be made available to both models simultaneously. 
- Command line arguments
   * In contrast to [NVIDIA/Tacotron2](https://github.com/NVIDIA/tacotron2) which uses a separate configuration file for model parameters, this repo offers the flexibility to change hyperparameters and other configurations by passing arguments when running a module. 
   * Recommended hyperparameters are set as default if no respective argument is passed.
- Switching costs are avoided
   * Both the TTS feature predictor (Tacotron2) and neural vocoder (WaveGlow) can be trained in the same repo eliminating time resulting from file transfer and otherwise switching across repos. 
   * An inference module is provided that manages the hand-off between the Tacotron2 and WaveGlow models to get inferences with your trained model quickly.

For those new to TTS and Tacotron2/WaveGlow, it's recommended to read the initial context provided in the source repo. 

## Additions to Source Repo

- [Analytics tracking](#analytics-tracking)
- [Padding adjusted loss](#padding-adjusted-loss)
- [Warm start](#warm-start)
- [Inference using ground truth mels](#inferece-using-ground-truth-mels)
- [Data Acquisition](#data-aquisition)
   * [Script files](#epoch-training-loss)
   * [Audio editing](#audio-editing)
- [Training WaveGlow with predicted mels](#predicted-mels)

## Table of Contents

- [Example model inference](#example-model-inference)
- [Getting Started](#getting-started)
   * [Requirements](#requirements)
   * [Quick start guide](#quick-start-guide)
   * [Downloading existing models](#downloading-existing-models)
   * [Recording](#recording)
   * [Training commands](#training-commands)
- [Performance](#performance)
   * [Example learning curve](#example-learning-curve)
- [Additions to source repo](#additions-to-source-repo)
- [Release notes](#release-notes)
   * [Changelog](#changelog)
   * [Known issues](#known-issues)

## Example model inference

[seashells_AC.wav](https://github.com/acharabin/DeepLearningExamples/tree/temp/PyTorch/SpeechSynthesis/Tacotron2/audio/seashells_AC.wav)

## Getting Started
   * Requirements
   * Quick start guide
   * Downloading existing models
   * Recording
   * Training commands

## Performance
   * Example learning curve

## Additions to source repo

### Analytics tracking

Having an accurate, ongoing, and easily accessible log of model training loss as Tacotron2 and WaveGlow models train is essential to properly supervise model training and ensure efficient use of GPUs. If training loss has converged, has hit a spike, or isn't meeting expectations/requirements, the training supervisor needs to be aware of this and take action by adjusting hyperparameters, changing the model setup, or halting training. 

The source repo provides logging of the training loss for each batch, and at the last batch of an epoch. While validation loss across all validation samples is logged at the end of each epoch, due to the limitations of validation loss (i.e. properly reconciling actual and predicted frames over time for loss computation), monitoring Tacotron2 training relies primarily on training loss; validation loss isn't referenced once in either the official Tacotron2 paper or the source repo. But as the batch size decreases, batch training loss deviates further from the signal the supervisor needs - how closely the model fits the data on average across all training samples. The problem is exacerbated in WaveGlow where an additional stochastic element is added to the batching process; for each epoch/passage, a sample of frames of a fixed 'segment length' (typically between 4K and 16K) are taken from each passage and used training. Note that with a sampling rate of 22050 audio frames per segment, a segment length of 4K would result in passage segments that are less than 1/5th of a second. While this added stochastic element defends against overfitting and supports model generalization, it further degrades the usefulness of batch training loss to guage model learning. 

As a result, an additional step was added after all batches in an epoch are completed to compute the training loss across a random sample of (or all) training passages based on the --epoch-loss-samples argument. Furthermore, results are appended to a file epochtrainingloss{modelname}.csv in the selected output directory at the end of each epoch. If --upload-to-s3 argument is used, the compute instance is associated with an AWS account, and associated s3 bucket and key information is provided in the Tacotron2/.env file, epoch loss will automatically be uploaded to s3 after each epoch so it can be connected to analytics tools for learning curve visualization (i.e. Tableau, Jupyter Notebook). Ammendments can be found in the following file: [tacotron2/train.py](https://github.com/acharabin/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/train.py)

### Padding adjusted loss

A requirement of Tacotron2 & WaveGlow training is for all passages in a batch to have a consistent size. To accomplish this, WaveGlow uses a stochastic approach whereby an audio segment of a fixed segment length (i.e. 8000 frames) is randomly sampled from each passage, afterwhich it's mel spectrogram pair is derived. Since the fixed segment length must be less than or equal to the passage audio length for the passage to be included, no 'padding' is required for WaveGlow. Tacotron2 on the other hand is an autoregressive model which requires the full chronology of audio frames in a passage for good performance. As a result, instead of a stochastic segment sampling, Tacotron2's data loader uses a custom collate (passage combination) function that adjusts all passages in a batch to have the same length as that of its longest passage. This is performed separately for both passage input characters and mel spectrograms, the inputs and outputs of the Tacotron2 model. Length is added to passages using zero padding, which fills character embedding vectors, and/or audio amplitudes at a time step, with values of 0. 

Tacotron2 uses mean squared error (MSE) loss out of box whereby the squared error is computed between actual and predicted mel spectrograms at each frequency bin and time step, then averaged. The values for some of these time steps (for passages that aren't the longest) will all be zero because of the zero padding. As the batch size increases, the longest passage length in the batch is expected to increase, and as a result more time steps become zero padded. Predictions of 0 across all time steps will perform well for the shortest passages that contain almost all zero-padded time steps, so loss decreases. This makes loss incomparable across different batch sizes and different passages in a batch. Most critically, it encourages the Tacotron2 model to focus on learning when the passage has stopped vs. the mel features of the passage. This makes training Tacotron2 highly inefficient and prevents convergence when using large batch sizes. At small batch sizes, the signal at each update step is diluted from the padding. 

Fortunately, padding can be 'excluded' from loss computation with minimal intervention to vanilla MSE loss. Instead of taking the mean squared loss, a sum of squared loss is taken across all frequency bins and time steps, and divided by the number of frequency bins in time steps that aren't zero padded (don't contain all zeros). The numerator is still burdened by padded time steps to encourage accurate stopping prediction, but now the denominator reflects the true passage lenght. The end result is an improved loss calculation that makes loss comparable across different passages and batch sizes, and more importantly provides drastic improvements in the productivity of each learning step when medium to large batch sizes are used. Ammendments can be found in the following file: [tacotron2/loss_function.py](https://github.com/acharabin/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/loss_function.py)

### Warm start

Warm start code from [NVIDIA/Tacotron2](https://github.com/NVIDIA/tacotron2) integrated but now using command line arguments to choose a model to warm start from and decide which layers to ignore. Warm start is also available for WaveGlow training. NVIDIA's [LJ voice WaveGlow checkpoint](https://catalog.ngc.nvidia.com/orgs/nvidia/models/waveglow256pyt_fp16/files) provides recognizable inferences of new Voices without additional training, and can be used with warm start to achieve high quality audio substantially quicker than training from scratch.       

### Inference using ground truth mels

To guage whether Tacotron2 or WaveGlow is the bottleneck for improved inference quality, your WaveGlow model can be run on ground truth mels for a selected passage. The difference between inferences obtained using ground truth mels and those obtained using Tacotron2 infered mels tells you the inference quality improvement that would be achieved if Tacotron2 was trained to perfectly match the actual audio. Furthermore, the difference between the inferences obtained using the ground truth mels and the ground truth audio tells you the inference quality improvement that would be achieved if the neural vocoder (in this case WaveGlow) was trained to perfectly match the actual audio. Whichever model is farther away from it's 'ideal' model based on an opinion score should be the focus for additional tuning efforts.  

See below for an example inference command using ground truth mels. 

```bash
python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> --wn-channels 256 -o output/inference/groundtruth/ --include-warmup -i AC-Voice-Cloning-Data/filelists/audio/acs_audio_text_train_filelist.txt --suffix _train1 --use-ground-truth-mels --cpu --gtm-index 0 --fp16
```

In this case, the --gtm-index argument indicates to use the first passage in the filelist provided (-i) to get inferences. Ground truth audio is then obtained from the path in the filelist. 

### Data Acquisition
#### Script files
TBD

#### Audio editing
TBD

### Training WaveGlow with predicted mels

In the original [Tacotron2 paper](https://arxiv.org/abs/1712.05884), the best performance was achieved when WaveNet was trained on Tacotron2 predicted mels vs. ground truth mels. The official [WaveGlow paper](https://arxiv.org/pdf/1811.00002v1.pdf) specifies application using ground mels. Existing NVIDIA WaveGlow repositories don't offer tools to train WaveGlow using predicted mels. It's expected that the inference quality of Tacotron2/WaveGlow models may be improved by training WaveGlow on Tacotron2 predicted mels, although there are technical limitations in doing so. Firstly, given WaveGlow isn't autoregressive and only uses information at the current time step to predict respective audio, properly reconciling the time steps of predicted mels and ground truth audio is essential. Secondly, the stochastic audio segment sampling used in WaveGlow further complicates properly mapping predicted mels with ground truth audio by time step. Thirdly, predicted mels and actual audio can have a different length, so clipping or padding would be required for WaveGlow to be trained on the last segment of a passage. Finally, since predicted mel spectrograms are 'reduced' - by preset the mels are an average over a window of 1024 frames of audio and hop by 256 frames per time step - there are additional challenges to align reduced mels with frames of audio. 

As an initial attempt inspired by the improvements noted in the Tacotron2 paper, functionality to save and use predicted mels in training was developed but remains in an experimental stage. The bottleneck for inference quality of the AC voice checkpoints provided was determined to be WaveGlow's ability to predict actual audio using ground truth mels (see []), and hence additional work using predicted mels was paused. 

The approach developed to save predicted mel/audition segments for a given passage is as follows: 
```bash
1. Obtain Tacotron2 predicted mels from the passage text.
2. Take the first segment of the mel and associated audio based on the segment length and save them to the output directories.
3. Create and append to a new filelist with the passage's text and link to the audio segment.
4. Repeat steps 2 & 3 with subsequent mel/audio segments until there are no additional audio segments for the passage.
5. For the last audio segment, add zero padding if needed to complete the full segment if needed.   
6. Continue to the next passage.
```
The execution can be found in the files below
[get_predicted_mels.py](https://github.com/acharabin/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/save_predicted_mels.py)
[waveglow/loss_function.py](https://github.com/acharabin/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/waveglow/data_function.py)

The following commands can be used to save predicted mel/audio segments and train WaveGlow using them. 

Save predicted mel/audio segments for train/validation
```bash
python save_predicted_mels.py -i AC-Voice-Cloning-Data/filelists/audio/acs_audio_text_train_filelist.txt -t --tacotron2 <Tacotron2_checkpoint> --split-segments --segment-length 32768 --no-padding --reset-filelist --empty-output-path

python save_predicted_mels.py -i AC-Voice-Cloning-Data/filelists/audio/acs_audio_text_validation_filelist.txt --filelist-output-path AC-Voice-Cloning-Data/filelists/audio/acs_audio_segment_text_validation_filelist.txt -v --tacotron2 <Tacotron2_checkpoint> --split-segments --segment-length 32768 --no-padding --reset-filelist --empty-output-path
```

Train WaveGlow with predicted mel/audio segments
```bash
python -m multiproc train.py -m WaveGlow -o output/ -lr 1e-4 --epochs 1001 --epochs-per-checkpoint 50 -bs 3 --segment-length 32768 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-enabled --cudnn-benchmark --log-file waveglowlog.json --training-files AC-Voice-Cloning-Data/filelists/audio/acs_audio_text_train_filelist.txt --validation-files AC-Voice-Cloning-Data/filelists/audio/acs_audio_segment_text_validation_filelist.txt --amp --upload-epoch-loss-to-s3 --warm-start --ignore-layers [] --checkpoint-path <WaveGlow_checkpoint> --wn-channels 256 --use-predicted-mels```

## Release notes
   * Changelog
   * Known issues
