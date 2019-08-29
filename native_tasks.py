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

import cffi
import legion
import os
import subprocess

root_dir = os.path.dirname(os.path.realpath(__file__))
simple_mapper_h_path = os.path.join(root_dir, 'native_kernels', 'simple_mapper.h')
native_kernels_so_path = os.path.join(root_dir, 'native_kernels', 'build', 'libnative_kernels.so')
simple_mapper_header = subprocess.check_output(['gcc', '-E', '-P', simple_mapper_h_path]).decode('utf-8')

ffi = cffi.FFI()
ffi.cdef(simple_mapper_header)
c = ffi.dlopen(native_kernels_so_path)

if legion.is_script:
    print('WARNING: unable to set mapper in script mode')
else:
    c.register_simple_mapper()

c.register_phaser_tasks(101)
