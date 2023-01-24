# Tacotron 2 And WaveGlow For PyTorch

This repository provides additions to NVIDIA/DeepLearningExamples scripts to further optimize training and inferences with Tacotron 2 and WaveGlow v1.6 models. More fundamentally, this repository is designed to provide a recipe for not just how to train a TTS model on an existing open source dataset (i.e. LJ voice), but to be able to easily scale application to new voices through tools for script management, audio editing, transfer learning, and more. As such, the repo comes equipped with a new open source voice dataset (AC voice) and associated models/inferences as a proof of concept.

## Additional Context from Source Repo

[Source README.md](https://github.com/acharabin/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/README_original.md)

## Additions to Source Repo

- [Analytics tracking](#analytics-tracking)
   * [Epoch training loss](#epoch-training-loss)
   * [Auto upload to s3](#auto-upload-to-s3)
- [Warm start](#warm-start)
- [Padding adjusted loss](#padding-adjusted-loss)
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
#### Epoch training loss
#### Auto upload to s3
### Warm start
### Padding adjusted loss
### Inference using ground truth mels
### Data Acquisition
#### Script files
#### Audio editing
### Training WaveGlow with predicted mels

## Release notes
   * Changelog
   * Known issues
