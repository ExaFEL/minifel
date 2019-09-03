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

#include "cufft.h"

#include "legion.h"

using namespace Legion;

#if 0
__global__
void gpu_phaser_kernel(Rect<3> rect,
                    const FieldAccessor<READ_ONLY, int16_t, 3, coord_t, Realm::AffineAccessor<int16_t, 3, coord_t> > x,
                    unsigned long long *result)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p(rect.lo.x + idx, rect.lo.y + idy, rect.lo.z + idz);

  // WARNING: This kernel is really, really inefficient. Please don't
  // use this in any context where performance is important!!!

  // FIXME: CUDA only supports atomicAdd on unsigned. Hopefully this
  // cast does sign extension???
  unsigned long long value = x[p];
  atomicAdd(result, value);
}
#endif

__global__
void phaser_kernel(cufftComplex *rho_hat, const float *amplitudes)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  cufftComplex rhat = rho_hat[idx];
  float amplitude = amplitudes[idx];

  float phase = atan2(rhat.x, rhat.y);

  // compute the complex exponent:
  // https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html
  cufftComplex exp_phase = { .x = amplitude * cos(phase), .y = amplitude * sin(phase) };

  bool amp_mask = true; // FIXME
  rho_hat[idx] = amp_mask ? exp_phase : rhat;
}

__global__
void ER_update_kernel(cufftComplex *rho, const cufftComplex *rho_mod, const bool *support)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  rho[idx] = support ? rho_mod[idx] : cufftComplex { .x = 0, .y = 0 };
}

__global__
void HIO_update_kernel(cufftComplex *rho, const cufftComplex *rho_mod, const bool *support, float beta)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  cufftComplex rmod = rho_mod[idx];
  rho[idx] = support ? rmod : cufftComplex { .x = rho[idx].x - beta * rmod.x, .y = rho[idx].y - beta * rmod.y };
}

void run_phaser_kernel(cufftComplex *rho_hat, const float *amplitudes, Rect<3> rect)
{
  const dim3 block(256, 1, 1);
  const dim3 grid(rect.volume() / block.x, 1, 1);

  phaser_kernel<<<grid, block>>>(rho_hat, amplitudes);
}

void run_ER_update_kernel(cufftComplex *rho, const cufftComplex *rho_mod, const bool *support, Rect<3> rect)
{
  const dim3 block(256, 1, 1);
  const dim3 grid(rect.volume() / block.x, 1, 1);

  ER_update_kernel<<<grid, block>>>(rho, rho_mod, support);
}

void run_HIO_update_kernel(cufftComplex *rho, const cufftComplex *rho_mod, const bool *support, Rect<3> rect, float beta)
{
  const dim3 block(256, 1, 1);
  const dim3 grid(rect.volume() / block.x, 1, 1);

  HIO_update_kernel<<<grid, block>>>(rho, rho_mod, support, beta);
}

class FFT {
public:
  FFT(Rect<3> rect, const size_t *strides)
    : rect(rect)
    , strides(strides)
  {
    int n[3] = {int(rect.hi.x - rect.lo.x + 1), int(rect.hi.y - rect.lo.y + 1), int(rect.hi.z - rect.lo.z + 1)};

    if (cufftPlanMany(&plan, 3, n,
                      NULL, 1, rect.volume(),
                      NULL, 1, rect.volume(),
                      CUFFT_C2C, 1) != CUFFT_SUCCESS) {
      assert(false &&"cuFFT error: Plan creation failed");
    }
  }

  ~FFT()
  {
    cufftDestroy(plan);
  }

  void run(cufftComplex *input, cufftComplex *output, int direction)
  {
    if (cufftExecC2C(plan, input, output, direction) != CUFFT_SUCCESS) {
      assert(false && "cuFFT error: ExecC2C Forward failed");
    }

    if (cudaDeviceSynchronize() != cudaSuccess){
      assert(false && "CUDA error: Failed to synchronize");
    }
  }

private:
  Rect<3> rect;
  const size_t *strides;
  cufftHandle plan;
};

class Phaser {
public:
  Phaser(long er_iter, long hio_iter, double hio_beta,
         const float *amplitudes,
         cufftComplex *rho, bool *support, Rect<3> rect, const size_t *strides)
    : er_iter(er_iter)
    , hio_iter(hio_iter)
    , hio_beta(hio_beta)
    , amplitudes(amplitudes)
    , rho(rho)
    , support(support)
    , rect(rect)
    , strides(strides)
    , rho_fft(rect, strides)
  {
    cudaMalloc((void**)&rho_hat, sizeof(cufftComplex) * rect.volume());
    if (cudaGetLastError() != cudaSuccess) {
      assert(false && "CUDA error: Failed to allocate");
    }
  }

  ~Phaser()
  {
    cudaFree(rho_hat);
  }

  void run()
  {
    ER_loop();
    HIO_loop();
    ER_loop();
    // shrink_wrap();
  }

private:
  void ER_loop()
  {
    for (long k = 0; k < er_iter; ++k) {
      ER();
    }
  }

  void ER()
  {
    phase();
    run_ER_update_kernel(rho, rho_hat, support, rect);
  }

  void HIO_loop()
  {
    for (long k = 0; k < hio_iter; ++k) {
      HIO();
    }
  }

  void HIO()
  {
    phase();
    run_HIO_update_kernel(rho, rho_hat, support, rect, hio_beta);
  }

  void phase() // updates rho_hat
  {
    rho_fft.run(rho, rho_hat, CUFFT_FORWARD);
    run_phaser_kernel(rho_hat, amplitudes, rect);
    rho_fft.run(rho_hat, rho_hat, CUFFT_INVERSE);
  }

private:
  long er_iter;
  long hio_iter;
  double hio_beta;

  const float *amplitudes;

  cufftComplex *rho;
  bool *support;
  cufftComplex *rho_hat;

  Rect<3> rect;
  const size_t *strides;
  FFT rho_fft;
};

struct gpu_phaser_task_args {
  int64_t map[1];
  LogicalRegion diffraction;
  LogicalRegion reconstruction;
  int32_t hio_iter;
  float hio_beta;
  int32_t er_iter;
  FieldID diffraction_fields[3];
  FieldID reconstruction_fields[2];
};

__host__
int64_t gpu_phaser_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(gpu_phaser_task_args));
  gpu_phaser_task_args args = *(gpu_phaser_task_args *)(task->args);

  assert(regions.size() == 2);

  const FieldAccessor<READ_ONLY, float, 3, coord_t, Realm::AffineAccessor<float, 3, coord_t> > amplitude(regions[0], args.diffraction_fields[2]);
  Rect<3> diffraction_rect = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  size_t diffraction_strides[3];
  const float *amplitude_origin = amplitude.ptr(diffraction_rect, diffraction_strides);

  const FieldAccessor<READ_WRITE, bool, 3, coord_t, Realm::AffineAccessor<bool, 3, coord_t> > support(regions[1], args.reconstruction_fields[0]);
  const FieldAccessor<READ_WRITE, cufftComplex, 3, coord_t, Realm::AffineAccessor<cufftComplex, 3, coord_t> > rho(regions[1], args.reconstruction_fields[1]);
  Rect<3> rho_rect = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
  size_t rho_strides[3];
  size_t support_strides[3];
  cufftComplex *rho_origin = rho.ptr(rho_rect, rho_strides);
  bool *support_origin = support.ptr(rho_rect, support_strides);

  assert(diffraction_rect == rho_rect);
  assert(diffraction_strides[0] == rho_strides[0]);
  assert(diffraction_strides[1] == rho_strides[1]);
  assert(diffraction_strides[2] == rho_strides[2]);
  assert(diffraction_strides[0] == support_strides[0]);
  assert(diffraction_strides[1] == support_strides[1]);
  assert(diffraction_strides[2] == support_strides[2]);

  long hio_iter = args.hio_iter;
  double hio_beta = args.hio_beta;
  long er_iter = args.er_iter;

  Phaser phaser(er_iter, hio_iter, hio_beta,
                amplitude_origin,
                rho_origin, support_origin, rho_rect, rho_strides);
  phaser.run();

#if 0
  const dim3 block(8, 8, 4);
  const dim3 grid(
    ((rect.hi.x - rect.lo.x + 1) + (block.x-1)) / block.x,
    ((rect.hi.y - rect.lo.y + 1) + (block.y-1)) / block.y,
    ((rect.hi.z - rect.lo.z + 1) + (block.z-1)) / block.z);

  unsigned long long result = 0;

  unsigned long long *gpu_result;
  if (cudaMalloc(&gpu_result, sizeof(unsigned long long)) != cudaSuccess) {
    abort();
  }

  if (cudaMemcpy(gpu_result, &result, sizeof(unsigned long long), cudaMemcpyHostToDevice) != cudaSuccess) {
    abort();
  }

  gpu_phaser_kernel<<<grid, block>>>(rect, x, gpu_result);

  if (cudaMemcpy(&result, gpu_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
    abort();
  }

  int64_t sum = result;
  // printf("gpu sum is %" PRId64 "\n", sum);
  return sum;
#else
  return 0;
#endif
}
