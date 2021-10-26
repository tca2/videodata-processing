#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import allprocesses

parser = argparse.ArgumentParser()
parser.add_argument('directory_name', help='Directory of video files')
#parser.add_argument('--filename', help='Process one video file')
args = parser.parse_args()

allprocesses.run_allprocesses(args.directory_name)