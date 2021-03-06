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
from legion import index_launch, task, ID, MustEpochLaunch, Partition, R, Reduce, Region, RW, Tunable
import numpy
from numpy import fft
import os

import data_collector

from phaseret import InitialState, Phaser
from phaseret.generator3D import Projection
import pysingfel as ps


###
### Solver
###


N_POINTS = 201


@task(privileges=[R, R, R, R, Reduce('+')], leaf=True)
def preprocess(images, orientations, active, pixels, diffraction, voxel_length):
    print(f"Aligning {active.active[0]} events")
    for l in range(active.active[0]):
        ps.merge_slice(
            images.image[l], pixels.momentum, orientations.orientation[l],
            diffraction.accumulator, diffraction.weight, voxel_length,
            inverse=False)


@task(privileges=[R, RW], leaf=True)
def solve_step(diffraction, reconstruction, rank, iteration,
               hio_iter, hio_beta, er_iter, sw_thresh):
    numpy.seterr(invalid='ignore', divide='ignore')
    amplitude = numpy.nan_to_num(fft.ifftshift(
        (diffraction.accumulator + diffraction.accumulator[::-1,::-1,::-1])
        / (diffraction.weight + diffraction.weight[::-1,::-1,::-1])))
    initial_state = InitialState(amplitude, reconstruction.support,
                                 reconstruction.rho, True)

    if iteration == 0:
        print(f"Initializing rank #{rank}")
        initial_state.generate_support_from_autocorrelation(rel_threshold=0.25)
        initial_state.generate_random_rho()

    phaser = Phaser(initial_state, monitor='last')
    phaser.ER_loop(er_iter)
    phaser.HIO_loop(hio_iter, hio_beta)
    phaser.ER_loop(er_iter)
    phaser.shrink_wrap(sw_thresh)

    err_Fourier = phaser.get_reciprocal_errs()[-1]
    err_real = phaser.get_real_errs()[-1]
    print(f"Rank #{rank}, iter #{iteration} -- "
          f"errors: {err_Fourier:.5f}, {err_real:.5f}")

    numpy.copyto(reconstruction.support, phaser.get_support(True),
                 casting='no')
    numpy.copyto(reconstruction.rho, phaser.get_rho(True), casting='no')


@task(privileges=[R], leaf=True)
def save_diffraction(diffraction, idx):
    print("Saving diffraction...")
    with h5.File(os.environ['OUT_DIR'] + f'/amplitude-{idx}.hdf5', 'w') as f:
        f.create_dataset("accumulator", shape=diffraction.accumulator.shape,
                         data=diffraction.accumulator,
                         dtype=diffraction.accumulator.dtype)
        f.create_dataset("weight", shape=diffraction.weight.shape,
                         data=diffraction.weight,
                         dtype=diffraction.weight.dtype)


@task(privileges=[R], leaf=True)
def save_rho(data, idx):
    print("Saving density...")
    with h5.File(os.environ['OUT_DIR'] + f'/rho-{idx}.hdf5', 'w') as f:
        f.create_dataset("rho", shape=data.rho.shape,
                         data=fft.fftshift(data.rho), dtype=data.rho.dtype)


@task(privileges=[R], leaf=True)
def save_images(data, idx):
    print("Saving images...")
    with h5.File(os.environ['OUT_DIR'] + f'/images-{idx}.hdf5', 'w') as f:
        f.create_dataset("images", shape=data.image.shape, data=data.image,
                         dtype=data.image.dtype)


@task(privileges=[RW], leaf=True)
def load_pixels(pixels):
    beam = ps.Beam('data/exp_chuck.beam')
    det = ps.PnccdDetector(
        geom='data/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/'
             'geometry/0-end.data',
        beam=beam)
    pixel_momentum = det.pixel_position_reciprocal
    numpy.copyto(pixels.momentum, det.pixel_position_reciprocal, casting='no')
    max_pixel_dist = numpy.max(det.pixel_distance_reciprocal)
    return max_pixel_dist


@task(inner=True) # replicable=True, # FIXME: Can't replicate both this and main
def solve(n_runs):
    n_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    print(f"Working with {n_procs} processes\n")

    # Allocate data structures.
    n_events_per_node = 100
    event_raw_shape = (4, 512, 512)
    images = Region(
        (n_events_per_node * n_procs,) + event_raw_shape, {'image': legion.float64})
    orientations = Region(
        (n_events_per_node * n_procs, 4), {'orientation': legion.float32})
    active = Region((n_procs,), {'active': legion.uint32})
    legion.fill(images, 'image', 0)
    legion.fill(orientations, 'orientation', 0)
    legion.fill(active, 'active', 0)
    images_part = Partition.restrict(
        images, [n_procs], numpy.eye(4, 1) * n_events_per_node, (n_events_per_node,) + event_raw_shape)
    orient_part = Partition.restrict(
        orientations, [n_procs], numpy.eye(2, 1) * n_events_per_node, (n_events_per_node, 4))
    active_part = Partition.restrict(
        active, [n_procs], numpy.eye(1, 1), (1,))

    volume_shape = (N_POINTS,) * 3
    diffraction = Region(volume_shape, {
        'accumulator': legion.float32,
        'weight': legion.float32})
    legion.fill(diffraction, 'accumulator', 0.)
    legion.fill(diffraction, 'weight', 0.)

    n_reconstructions = 4
    reconstructions = []
    for i in range(n_reconstructions):
        reconstruction = Region(volume_shape, {
            'support': legion.bool_,
            'rho': legion.complex64})
        legion.fill(reconstruction, 'support', False)
        legion.fill(reconstruction, 'rho', 0.)
        reconstructions.append(reconstruction)

    # Load pixel momentum
    pixels = Region(event_raw_shape + (3,), {'momentum': legion.float64})
    legion.fill(pixels, 'momentum', 0.)
    max_pixel_dist = load_pixels(pixels).get()
    voxel_length = 2 * max_pixel_dist / (N_POINTS - 1)

    images_per_solve = n_events_per_node
    iterations_ahead = 2

    complete = False
    iteration = 0
    fences = []
    n_events_ready = []
    while not complete or iteration < 50:
        if not complete:
            # Obtain the newest copy of the data.
            with MustEpochLaunch([n_procs]):
                index_launch(
                    [n_procs], data_collector.fill_data_region,
                    images_part[ID], orient_part[ID], active_part[ID], images_per_solve)

            # Preprocess data.
            index_launch(
                [n_procs], preprocess,
                images_part[ID], orient_part[ID], active_part[ID], pixels, diffraction,
                voxel_length)

        # Run solver.
        assert n_reconstructions == 4
        hio_loop = 100
        er_loop = hio_loop // 2
        solve_step(diffraction, reconstructions[0], 0, iteration,
                   hio_loop, .1, er_loop, .14)
        solve_step(diffraction, reconstructions[1], 1, iteration,
                   hio_loop, .05, er_loop, .14)
        solve_step(diffraction, reconstructions[2], 2, iteration,
                   hio_loop, .1, er_loop, .16)
        solve_step(diffraction, reconstructions[3], 3, iteration,
                   hio_loop, .05, er_loop, .16)

        if not complete:
            # Make sure we don't run more than N iterations ahead.
            fences.append(legion.execution_fence(future=True))
            if iteration - iterations_ahead >= 0:
                fences[iteration - iterations_ahead].get()

            # Check that all runs have been read and that all events have been consumed.
            if data_collector.get_num_runs_complete() == n_runs:
                n_events_ready.append(index_launch([n_procs], data_collector.get_num_events_ready, active_part[ID], reduce='+'))
                if iteration - iterations_ahead >= 0:
                    ready = n_events_ready[iteration - iterations_ahead].get()
                    print(f'All runs complete, {ready} events remaining', flush=True)
                    complete = ready == 0

        iteration += 1

    ##### -------------------------------------------------------------- #####

    # for idx in range(n_procs):
    #     save_images(images_part[idx], idx, point=idx)
    for i in range(n_reconstructions):
        save_rho(reconstructions[i], i)
    save_diffraction(diffraction, 0)
