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

#include "simple_mapper.h"

#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class SimpleMapper : public DefaultMapper
{
public:
  SimpleMapper(MapperRuntime *rt, Machine machine, Processor local,
              const char *mapper_name);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
};

SimpleMapper::SimpleMapper(MapperRuntime *rt, Machine machine, Processor local,
                         const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

Processor SimpleMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task)
{
  // If this is an individual task with a point assigned, round robin
  // around the machine.
  if (!task.is_index_space && !task.index_point.is_null()) {
    VariantInfo info = 
      default_find_preferred_variant(task, ctx, false/*needs tight*/);
    switch (info.proc_kind)
    {
      case Processor::LOC_PROC:
        return default_get_next_global_cpu();
      case Processor::TOC_PROC:
        return default_get_next_global_gpu();
      case Processor::IO_PROC:
        return default_get_next_global_io();
      case Processor::OMP_PROC:
        return default_get_next_global_omp();
      case Processor::PY_PROC:
        return default_get_next_global_py();
      default:
        assert(false);
    }
  }

  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    SimpleMapper* mapper = new SimpleMapper(runtime->get_mapper_runtime(),
                                          machine, *it, "simple_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void preregister_simple_mapper()
{
  Runtime::add_registration_callback(create_mappers);
}

void register_simple_mapper()
{
  Runtime *runtime = Runtime::get_runtime();
  Machine machine = Machine::get_machine();
  Machine::ProcessorQuery query(machine);
  query.local_address_space();
  std::set<Processor> local_procs(query.begin(), query.end());
  for(auto it = local_procs.begin(); it != local_procs.end(); ) {
    if (it->kind() == Processor::UTIL_PROC) {
      it = local_procs.erase(it);
    } else {
      ++it;
    }
  }
  create_mappers(machine, runtime, local_procs);
}
