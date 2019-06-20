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

import h5py as h5
import legion
from legion import task, R, RW
import numpy
from numpy import fft
import os

import data_collector

from phaseret import InitialState, Phaser
from phaseret.generator3D import Projection

###
### Solver
###

# Oversimplified solve on realistic XPP data.
# Somewhat realistic solve on the generated 3D data.
# See user.py for details.


N_POINTS = 64

root_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(root_dir, 'data', 'wangzy')


@task(privileges=[RW], leaf=True)
def generate_data(data):
    cutoff = 2
    spacing = numpy.linspace(-cutoff, cutoff, N_POINTS)

    H, K, L = numpy.meshgrid(spacing, spacing, spacing)

    caffeine_pbd = os.path.join(data_dir, "caffeine.pdb")
    caffeine = Projection.Molecule(caffeine_pbd)

    atomsf_lib = os.path.join(data_dir, "atomsf.lib")
    caffeine_trans = Projection.moltrans(caffeine, H, K, L, atomsf_lib)
    caffeine_trans_ = fft.ifftshift(caffeine_trans)

    amplitudes = numpy.absolute(caffeine_trans)

    numpy.copyto(data.amplitudes, amplitudes)


@task(privileges=[R, RW], leaf=True)
def preprocess(data_in, data_out):
    print("Preprocessing")
    pass # pretend to build/refine data_out (3D) out of data_in (set of 2D)


@task(privileges=[RW], leaf=True)
def solve_step(data, rank, iteration):
    initial_state = InitialState(data.amplitudes, data.support, data.rho, True)

    if iteration == 0:
        print(f"Initializing rank #{rank}")
        initial_state.generate_support_from_autocorrelation()
        initial_state.generate_random_rho()

    phaser = Phaser(initial_state, monitor='last')
    phaser.HIO_loop(2, .1)
    phaser.ER_loop(2)
    phaser.shrink_wrap(.01)

    err_Fourier = phaser.get_reciprocal_errs()[-1]
    err_real = phaser.get_real_errs()[-1]
    print(f"Errors: {err_Fourier:.5f}, {err_real:.5f}")

    numpy.copyto(data.support, phaser.get_support(True), casting='no')
    numpy.copyto(data.rho, phaser.get_rho(True), casting='no')


@task(privileges=[R], leaf=True)
def save(data, shape, idx):
    print("Saving data...")
    with h5.File(os.environ['OUT_DIR'] + f'/run-{idx}.hdf5', 'w') as f:
        f.create_dataset("images", shape=shape, data=data.image,
                         dtype=legion.float32)


@task(privileges=[RW], replicable=True)
def solve(n_runs):
    n_procs = legion.Tunable.select(legion.Tunable.GLOBAL_PYS).get()
    print(f"Working with {n_procs} processes\n")

    # Allocate data structures.
    n_events_per_node = 100
    event_raw_shape = (4, 512, 512)
    images = legion.Region.create(
        (n_events_per_node * n_procs,) + event_raw_shape, {'image': legion.float32})
    orientations = legion.Region.create(
        (n_events_per_node * n_procs, 4), {'orientation': legion.float32})
    legion.fill(images, 'image', 0)
    legion.fill(orientations, 'orientation', 0)
    images_part = legion.Partition.create_by_restriction(
        images, [n_procs], numpy.ones([4, 1]), (n_events_per_node,) + event_raw_shape)
    orient_part = legion.Partition.create_by_restriction(
        orientations, [n_procs], [[n_events_per_node], [1]], (n_events_per_node, 4))

    gen_data_shape = (N_POINTS,) * 3
    data = legion.Region.create(gen_data_shape, {
        'amplitudes': legion.float32,
        'support': legion.bool_,
        'rho': legion.complex64})

    legion.fill(data, 'amplitudes', 0.)
    legion.fill(data, 'support', 0)
    legion.fill(data, 'rho', 0.)

    complete = False
    iteration = 0
    fences = []
    while not complete or iteration < 10:
        if not complete:
            # Obtain the newest copy of the data.
            with legion.MustEpochLaunch([n_procs]):
                for idx in range(n_procs): # legion.IndexLaunch([n_procs]): # FIXME: index launch
                    data_collector.fill_data_region(
                        images_part[idx], orient_part[idx], point=idx)

        # Preprocess data.
        for idx in range(n_procs): # legion.IndexLaunch([n_procs]): # FIXME: index launch
            preprocess(images_part, data)

        # Generate data on first run
        if not iteration:
            generate_data(data)

        # Run solver.
        solve_step(data, 0, iteration)

        if not complete:
            # Make sure we don't run more than 2 iterations ahead.
            fences.append(legion.execution_fence(future=True))
            if iteration - 2 >= 0:
                fences[iteration - 2].get()

            # Check that all runs have been read.
            complete = data_collector.get_num_runs_complete() == n_runs

        iteration += 1

    for idx in range(n_procs):
        save(images_part[idx], (n_events_per_node,) + event_raw_shape,
             idx, point=idx)
