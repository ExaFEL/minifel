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

from collections import OrderedDict
import h5py as h5
import itertools
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

use_cpu = False
use_gpu = True

gpu_no_overlap = False

N_POINTS = 201

gpu_phaser_0 = legion.extern_task(
    task_id=101,
    argument_types=[Region, Region, legion.int32, legion.float32, legion.int32, legion.int32],
    privileges=[R('amplitude_0'), RW],
    return_type=legion.void,
    calling_convention='regent')

gpu_phaser_1 = legion.extern_task(
    task_id=101,
    argument_types=[Region, Region, legion.int32, legion.float32, legion.int32, legion.int32],
    privileges=[R('amplitude_1'), RW],
    return_type=legion.void,
    calling_convention='regent')

gpu_phaser_2 = legion.extern_task(
    task_id=101,
    argument_types=[Region, Region, legion.int32, legion.float32, legion.int32, legion.int32],
    privileges=[R('amplitude_2'), RW],
    return_type=legion.void,
    calling_convention='regent')

gpu_phaser_3 = legion.extern_task(
    task_id=101,
    argument_types=[Region, Region, legion.int32, legion.float32, legion.int32, legion.int32],
    privileges=[R('amplitude_3'), RW],
    return_type=legion.void,
    calling_convention='regent')

gpu_phaser_4 = legion.extern_task(
    task_id=101,
    argument_types=[Region, Region, legion.int32, legion.float32, legion.int32, legion.int32],
    privileges=[R('amplitude_4'), RW],
    return_type=legion.void,
    calling_convention='regent')

gpu_phaser_tasks = [gpu_phaser_0, gpu_phaser_1, gpu_phaser_2, gpu_phaser_3, gpu_phaser_4]

@task(privileges=[R, R, R, R, Reduce('+', 'accumulator', 'weight')], leaf=True)
def preprocess(images, orientations, active, pixels, diffraction, voxel_length):
    print(f"Aligning {active.active[0]} events")
    for l in range(active.active[0]):
        ps.merge_slice(
            images.image[l], pixels.momentum, orientations.orientation[l],
            diffraction.accumulator, diffraction.weight, voxel_length,
            inverse=False)

@task(privileges=[R, R, R, R, Reduce('+', 'accumulator', 'weight')], leaf=True)
def preprocess_0(*args):
    return preprocess.body(*args)

@task(privileges=[R, R, R, R, Reduce('+', 'accumulator', 'weight')], leaf=True)
def preprocess_1(*args):
    return preprocess.body(*args)

@task(privileges=[R, R, R, R, Reduce('+', 'accumulator', 'weight')], leaf=True)
def preprocess_2(*args):
    return preprocess.body(*args)

@task(privileges=[R, R, R, R, Reduce('+', 'accumulator', 'weight')], leaf=True)
def preprocess_3(*args):
    return preprocess.body(*args)

@task(privileges=[R, R, R, R, Reduce('+', 'accumulator', 'weight')], leaf=True)
def preprocess_4(*args):
    return preprocess.body(*args)

preprocess_tasks = [preprocess_0, preprocess_1, preprocess_2, preprocess_3, preprocess_4]

def merge(diffraction, field):
    numpy.seterr(invalid='ignore', divide='ignore')
    amplitude = numpy.nan_to_num(fft.ifftshift(
        (diffraction.accumulator + diffraction.accumulator[::-1,::-1,::-1])
        / (diffraction.weight + diffraction.weight[::-1,::-1,::-1])))
    numpy.copyto(getattr(diffraction, 'amplitude_%s' % field), amplitude, casting='no')

@task(privileges=[RW('amplitude_0') + R('accumulator', 'weight')], leaf=True)
def merge_0(diffraction):
    return merge(diffraction, 0)

@task(privileges=[RW('amplitude_1') + R('accumulator', 'weight')], leaf=True)
def merge_1(diffraction):
    return merge(diffraction, 1)

@task(privileges=[RW('amplitude_2') + R('accumulator', 'weight')], leaf=True)
def merge_2(diffraction):
    return merge(diffraction, 2)

@task(privileges=[RW('amplitude_3') + R('accumulator', 'weight')], leaf=True)
def merge_3(diffraction):
    return merge(diffraction, 3)

@task(privileges=[RW('amplitude_4') + R('accumulator', 'weight')], leaf=True)
def merge_4(diffraction):
    return merge(diffraction, 4)

merge_tasks = [merge_0, merge_1, merge_2, merge_3, merge_4]

def solve_step(field, diffraction, reconstruction, rank, iteration,
               # hio_iter, hio_betas, er_iter, sw_threshes):
               hio_iter, hio_beta, er_iter, sw_thresh):
    # hio_beta = hio_betas[int(rank) % hio_betas.size]
    # sw_thresh = sw_threshes[int(rank) // hio_betas.size]

    numpy.seterr(invalid='ignore', divide='ignore')
    initial_state = InitialState(
        getattr(diffraction, 'amplitude_%s' % field),
        reconstruction.support, #.reshape(reconstruction.support.shape[1:]),
        reconstruction.rho, #.reshape(reconstruction.rho.shape[1:]),
        True)

    if iteration == 0:
        print(f"Initializing rank #{rank}")
        initial_state.generate_support_from_autocorrelation(rel_threshold=0.25)
        initial_state.generate_random_rho()

    phaser = Phaser(initial_state, monitor='last')
    if use_cpu:
        phaser.ER_loop(er_iter)
        phaser.HIO_loop(hio_iter, hio_beta)
        phaser.ER_loop(er_iter)
    if use_gpu:
        if gpu_no_overlap:
            legion.c.legion_runtime_enable_scheduler_lock()
        gpu_phaser_tasks[field](diffraction, reconstruction, hio_iter, hio_beta, er_iter, field)
        if gpu_no_overlap:
            legion.c.legion_runtime_disable_scheduler_lock()
    if use_cpu and use_gpu:
        assert numpy.array_equal(reconstruction.support, phaser.get_support(True))
        gpu_rho = reconstruction.rho
        cpu_rho = phaser.get_rho(True)
        match = numpy.all(numpy.logical_or(numpy.logical_and(numpy.isnan(cpu_rho), numpy.isnan(gpu_rho)), cpu_rho == gpu_rho))
        if not match:
            import tempfile
            tmp_dir = os.path.join(os.environ['MEMBERWORK'], 'chm137', 'minifel_output')
            with tempfile.NamedTemporaryFile(dir=tmp_dir, prefix='rho_cpu_', delete=False) as f:
                numpy.save(f, cpu_rho)
            with tempfile.NamedTemporaryFile(dir=tmp_dir, prefix='rho_gpu_', delete=False) as f:
                numpy.save(f, gpu_rho)
        assert match
        print('validated GPU solver results')
    if use_gpu and not use_cpu:
        numpy.copyto(phaser._support_, reconstruction.support, casting='no')
        numpy.copyto(phaser._rho_, reconstruction.rho, casting='no')
    phaser.shrink_wrap(sw_thresh)

    if use_cpu:
        err_Fourier = phaser.get_reciprocal_errs()[-1]
        err_real = phaser.get_real_errs()[-1]
        print(f"Rank #{rank}, iter #{iteration} -- "
              f"errors: {err_Fourier:.5f}, {err_real:.5f}")

    numpy.copyto(reconstruction.support, phaser.get_support(True),
                 casting='no')
    numpy.copyto(reconstruction.rho, phaser.get_rho(True), casting='no')


@task(privileges=[R('amplitude_0'), RW])
def solve_step_0(*args):
    return solve_step(0, *args)

@task(privileges=[R('amplitude_1'), RW])
def solve_step_1(*args):
    return solve_step(1, *args)

@task(privileges=[R('amplitude_2'), RW])
def solve_step_2(*args):
    return solve_step(2, *args)

@task(privileges=[R('amplitude_3'), RW])
def solve_step_3(*args):
    return solve_step(3, *args)

@task(privileges=[R('amplitude_4'), RW])
def solve_step_4(*args):
    return solve_step(4, *args)

solve_step_tasks = [solve_step_0, solve_step_1, solve_step_2, solve_step_3, solve_step_4]

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
        (n_events_per_node * n_procs,) + event_raw_shape, OrderedDict([('image', legion.float64)]))
    orientations = Region(
        (n_events_per_node * n_procs, 4), OrderedDict([('orientation', legion.float32)]))
    active = Region((n_procs,), OrderedDict([('active', legion.uint32)]))
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
    diffraction = Region(volume_shape, OrderedDict([
        ('accumulator', legion.float32),
        ('weight', legion.float32),
        ('amplitude_0', legion.float32),
        ('amplitude_1', legion.float32),
        ('amplitude_2', legion.float32),
        ('amplitude_3', legion.float32),
        ('amplitude_4', legion.float32)]))
    legion.fill(diffraction, 'accumulator', 0.)
    legion.fill(diffraction, 'weight', 0.)
    legion.fill(diffraction, 'amplitude_0', 0.)
    legion.fill(diffraction, 'amplitude_1', 0.)
    legion.fill(diffraction, 'amplitude_2', 0.)
    legion.fill(diffraction, 'amplitude_3', 0.)
    legion.fill(diffraction, 'amplitude_4', 0.)

    hio_betas = numpy.linspace(.05, .1, 5)
    sw_threshes = numpy.linspace(.14, .16, 2)

    n_reconstructions = hio_betas.size * sw_threshes.size

    # reconstructions = Region((n_reconstructions,) + volume_shape, OrderedDict([
    #     ('support', legion.bool_),
    #     ('rho', legion.complex64)]))
    # reconstructions_part = Partition.restrict(
    #     reconstructions, [n_reconstructions], numpy.eye(4, 1), (1,) + volume_shape)
    # legion.fill(reconstructions, 'support', False)
    # legion.fill(reconstructions, 'rho', 0.)

    reconstructions = []
    for i in range(n_reconstructions):
        reconstruction = Region(volume_shape, OrderedDict([
            ('support', legion.bool_),
            ('rho', legion.complex64)]))
        legion.fill(reconstruction, 'support', False)
        legion.fill(reconstruction, 'rho', 0.)
        reconstructions.append(reconstruction)

    # Load pixel momentum
    pixels = Region(event_raw_shape + (3,), OrderedDict([('momentum', legion.float64)]))
    legion.fill(pixels, 'momentum', 0.)
    max_pixel_dist = load_pixels(pixels).get()
    voxel_length = 2 * max_pixel_dist / (N_POINTS - 1)

    images_per_solve = 4
    iterations_ahead = 5

    n_fields = 5

    complete = False
    iteration = 0
    futures = []
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
                [n_procs], preprocess_tasks[iteration % n_fields],
                images_part[ID], orient_part[ID], active_part[ID], pixels, diffraction,
                voxel_length)

            futures.append(merge_tasks[iteration % n_fields](diffraction))

        # Run solver.
        hio_loop = 100
        er_loop = hio_loop // 2
        for i, (hio_beta, sw_thresh) in enumerate(itertools.product(hio_betas, sw_threshes)):
            solve_step_tasks[iteration % n_fields](
                diffraction, reconstructions[i], i, iteration,
                hio_loop, hio_beta, er_loop, sw_thresh)
        # index_launch(
        #     [n_reconstructions], solve_step_tasks[iteration % n_fields],
        #     diffraction, reconstructions_part[ID], ID, iteration,
        #     hio_loop, hio_betas, er_loop, sw_threshes)

        if not complete:
            # Make sure we don't run more than K iterations ahead.
            if iteration - iterations_ahead >= 0:
                futures[iteration - iterations_ahead].get()

            # Check that all runs have been read and that all events have been consumed.
            if data_collector.get_num_runs_complete() == n_runs:
                n_events_ready.append(index_launch([n_procs], data_collector.get_num_events_ready, active_part[ID], reduce='+'))
                if iteration - iterations_ahead >= 0:
                    ready = n_events_ready[iteration - iterations_ahead].get()
                    print(f'All runs complete, {ready} events remaining', flush=True)
                    complete = ready == 0

            if complete:
                for field in range(1, n_fields):
                    merge_tasks[(iteration + field) % n_fields](diffraction)


        iteration += 1

    ##### -------------------------------------------------------------- #####

    # for idx in range(n_procs):
    #     save_images(images_part[idx], idx, point=idx)
    for i in range(n_reconstructions):
        save_rho(reconstructions[i], i)
    # index_launch([n_reconstructions], save_rho, reconstructions_part[ID], ID)
    save_diffraction(diffraction, 0)
