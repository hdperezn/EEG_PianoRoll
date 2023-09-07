

"""
some functions are taken from "https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb"

Copyright 2021 The TensorFlow Authors.
"""

import collections
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

def display_audio(pm: pretty_midi.PrettyMIDI, _SAMPLING_RATE, seconds=60):
  waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
  # Take a sample of the generated waveform to mitigate kernel resets
  waveform_short = waveform[:seconds*_SAMPLING_RATE]
  return display.Audio(waveform_short, rate=_SAMPLING_RATE)



"""
function "piano_roll_to_pretty_midi" taken from https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py
"""
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
        with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

"""
function specific for aplication "autoencoder piano roll"
""" 
def cut_midi_secTrial(X, wind=10):
    """
    X: piano roll array
    N_windows: N° of seconds
    """
    Xtrial = []
    step = int(X.shape[1]/wind)
    for i in range(wind):
    XX = X[:,int(i*step):(i+1)*step]
    Xtrial.append(XX)
    return np.asarray(Xtrial)

def pading_midis(Prolls_windowed, Prolls_trial_len = 128):
  """
  function to cut or padding the midi arrays to equal lengs
  """
  Prolls_cut = []
  for i in range(len(Prolls_windowed)):
    if Prolls_windowed[i].shape[2] >= Prolls_trial_len:
      x = Prolls_windowed[i][:,:,0:Prolls_trial_len]
      Prolls_cut.append(x)
    else:
      dif = Prolls_trial_len - Prolls_windowed[i].shape[2]
      x = np.concatenate( (Prolls_windowed[i][:,:,:],Prolls_windowed[i][:,:,-dif::] ), axis = 2)
      Prolls_cut.append(x)
  return np.concatenate(np.asarray(Prolls_cut), axis = 0)


def rebuilPianoRolls(X):
  """
  this function takes an output array from the vae net and returns
  a piano roll array ready to be converted into MIDI file
  input
  X: array with dimensions [samples, time, pitch, 1]
  output
  proll_padding: array with dimension [sample x pitch x time ]
  """
  #trasnpose to [sample x time x pitch]
  proll = X[:,:,:,0].transpose(0,2,1)
  proll_padding =np.pad(proll, ((0,0), (24,40), (0, 0)), 'constant')
  return proll_padding