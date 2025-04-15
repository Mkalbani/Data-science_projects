"""
Text to speech or Speech Sysnthesis(ss) --is a componanent of conversational AI
Vocal clarity is achieved deep neaural networks that produce human-like intonation and
clear articulation of words

Models: FastPitch, Deeo voice 1, Nemo Riva

Nemo Riva - is a GPU-accelerated SDK and it's available as pretrained models

Modern TTS systems are fairly complex, with an end to end pipeline consisting of several components that each require their own model or heuristics.

A standard pipeline for TTS might look like:

Text Normalization: Converting raw text to spoken text (eg. "Mr." → "mister").
Grapheme to Phoneme conversion (G2P): Convert basic units of text (ie. graphemes/characters) to basic units of spoken language (ie. phonemes).
Spectrogram Synthesis: Convert text/phonemes into a spectrogram.
Audio Synthesis: Convert spectrogram into audio. Also known as spectrogram inversion. Models which do this are called vocoders.

"""
"""
Text Normalization

types: abbreviations, dates, numbers, roman numerals accronyms, url, cardinal directions

"""
#set-up
## Install NeMo library. this for Google Colab
BRANCH = 'r1.22.0'
!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]

from nemo_text_processing.text_normalization import Normalizer

text_normalizer = Normalizer(input_case='cased', lang='en')
txt = "Mr. Johnson is turning 35 years old on 04-15-2023."

normalized_text = text_normalizer.normalize(txt)