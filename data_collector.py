#!/usr/bin/env python

# Copyright 2019 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import legion
from legion import index_launch, task, MustEpochLaunch, R, RW, Tunable
import numpy
from numpy import fft
import os
import threading

###
### Data Loading
###


data_store = []
n_events_ready = 0
n_runs_complete = 0
data_lock = threading.Lock()


def load_event_data(event, det):
    global n_events_ready
    image = event._dgrams[0].pnccd[0].raw.image
    orientation = event._dgrams[0].pnccd[0].raw.orientation

    with data_lock:
        data_store.append((image, orientation))
        n_events_ready += 1


@task(leaf=True)
def mark_completion(_):
    print('Completed load data for run')
    global n_runs_complete
    with data_lock:
        n_runs_complete += 1


def load_run_data(run):
    det = run.Detector('pnccd')

    # Hack: psana tries to register top-level task when not in script mode
    old_is_script = legion.is_script
    legion.is_script = True
    f = run.analyze(event_fn=load_event_data, det=det)
    legion.is_script = old_is_script

    n_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    with MustEpochLaunch([n_procs]):
        index_launch([n_procs], mark_completion, f)


def reset_data():
    global data_store, n_events_ready
    with data_lock:
        data_store = []
        n_events_ready = 0


@task(privileges=[RW, RW, RW], leaf=True)
def fill_data_region(images, orientations, active, limit=None):
    global data_store, n_events_ready
    with data_lock:
        taken = min(n_events_ready, limit) if limit is not None else n_events_ready
        assert taken <= len(data_store)
        raw = data_store
        data_store = raw[taken:]
        n_events_ready -= taken

    for idx in range(taken):
        numpy.copyto(images.image[idx,:,:,:], raw[idx][0], casting='no')
        numpy.copyto(orientations.orientation[idx,:], raw[idx][1], casting='no')
    active.active[0] = taken

    if taken > 0:
        print(f"Filled {taken} new events.")


def get_num_runs_complete():
    with data_lock:
        return n_runs_complete


@task(privileges=[R], return_type=legion.int64, leaf=True)
def get_num_events_ready(active):
    with data_lock:
        return n_events_ready
