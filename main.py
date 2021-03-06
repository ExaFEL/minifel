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
from legion import task

# Hack: psana tries to register top-level task when not in script mode
old_is_script = legion.is_script
legion.is_script = True
from psana import DataSource
legion.is_script = old_is_script

import os

import native_tasks
import data_collector
import solver


@task(top_level=True, replicable=True)
def main():
    limit = int(os.environ['LIMIT']) if 'LIMIT' in os.environ else None

    data_dir = os.environ['DATA_DIR']
    ds = DataSource(exp='junk', run=1, dir=data_dir, max_events=limit,
                    det_name='spi_cspad')
    # Note: DataSource doesn't seem to care about max_events when given
    # a filename.

    n_runs = 0
    runs = []
    for run in ds.runs():
        # FIXME: must epoch launch
        data_collector.load_run_data(run)
        # Right now, we assume one run or a serie of runs with the same
        # experimental configuration.

        n_runs += 1

        runs.append(run) # Keep run around to avoid having it be garbage collected.

    solver.solve(n_runs)

    legion.execution_fence(block=True) # Block to keep runs in scope until solve completes.
