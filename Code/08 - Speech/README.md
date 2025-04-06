# Speech Recognition and Synthesis (ASR & TTS)

This repository contains resources and insights from a lecture on Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) technologies. It covers various tasks related to speech processing and their application in NLP.

## Table of Contents
- [Speech Recognition and Synthesis (ASR \& TTS)](#speech-recognition-and-synthesis-asr--tts)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Concepts](#key-concepts)
    - [Automatic Speech Recognition (ASR)](#automatic-speech-recognition-asr)
    - [Text-to-Speech (TTS)](#text-to-speech-tts)
  - [Speech-Related Tasks](#speech-related-tasks)
  - [Evaluation Metrics](#evaluation-metrics)
  - [State-of-the-Art Models](#state-of-the-art-models)
  - [Good to Read](#good-to-read)
  - [Reference Resource \& Code](#reference-resource--code)

## Overview
This section covers the essentials of ASR and TTS, key components in speech processing, and their role in natural language understanding. It introduces various models and evaluation metrics used in speech tasks like transcription, synthesis, and translation.

## Key Concepts

### Automatic Speech Recognition (ASR)
- ASR is a process that converts spoken language into text.
- The **RNN-T** (Recurrent Neural Network-Transducer) model is essential for ASR, combining acoustic and language models.
- **Word Error Rate (WER)** is used to evaluate ASR systems.

### Text-to-Speech (TTS)
- TTS systems convert text into spoken words.
- **WaveNet** and other encoder-decoder models are used for synthesizing speech.
- **Mean Opinion Score (MOS)** is used to evaluate TTS quality.

## Speech-Related Tasks
- **Wake Word Detection**: Detects specific trigger words (e.g., "Hey Siri").
- **Speaker Diarization**: Identifies individual speakers in multi-speaker recordings.
- **Voice Cloning**: Synthetically recreates voices.

## Evaluation Metrics
- **TTS**: Mean Opinion Score (MOS) for assessing synthetic speech.
- **ASR**: Word Error Rate (WER) for transcription accuracy.

## State-of-the-Art Models
- **Whisper**: A robust ASR model by OpenAI.
- **Tacotron**: A deep-learning TTS model for natural speech synthesis.


## Good to Read
- [Latif et al. (2023)](https://arxiv.org/pdf/2308.12792)Sparks of Large Audio Models: A Survey and Outlook


## Reference Resource & Code
- https://github.com/MajoRoth/ASR