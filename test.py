#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
from random import randrange
from ale_python_interface import ALEInterface
from skimage import io
import numpy as np 

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    print "!!!"
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)
ale.setInt('frame_skip',1)
# Load the ROM file
ale.loadROM('roms/breakout.bin')
ale.setInt('max_num_frames',1)
# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()
a1,a2 =  ale.getScreenDims()
cnt = 0
# Play 10 episodes
import  numpy as  np
d = np.empty((a1,a2),dtype=np.uint8)
for episode in xrange(10):
  total_reward = 0
  while not ale.game_over():
    a = legal_actions[randrange(len(legal_actions))]
    #print legal_actions
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    ale.getScreenGrayscale(d)

    io.imshow(d)
    io.show()
    #print reward
    #print ale.getScreenRGB()

    #total_reward += reward
  #print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()