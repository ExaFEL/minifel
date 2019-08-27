/* Copyright 2019 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "phaser_tasks.h"

#include "legion.h"

#include <cstdint>
// FIXME: results in build failure on Summit with <cinttypes>
#include <inttypes.h>

using namespace Legion;

#ifdef USE_CUDA
int64_t gpu_phaser_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
#endif

void preregister_phaser_tasks(int phaser_task_id)
{
#ifdef USE_CUDA
  {
    TaskVariantRegistrar registrar(phaser_task_id, "phaser_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<int64_t, gpu_phaser_task>(registrar, "phaser_task");
  }
#endif
}

void register_phaser_tasks(int phaser_task_id)
{
  Runtime *runtime = Runtime::get_runtime();

#ifdef USE_CUDA
  {
    TaskVariantRegistrar registrar(phaser_task_id, "phaser_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    runtime->register_task_variant<int64_t, gpu_phaser_task>(registrar);
  }
#endif
}
