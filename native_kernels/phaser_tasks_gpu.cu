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

enum FieldIDs {
  FID_RHO = 1,
};

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

class Phaser {
public:
  __host__ Phaser(long er_iter, long hio_iter, double hio_beta,
                  cufftComplex *rho, Rect<3> rho_rect, const size_t *rho_strides)
    : er_iter(er_iter)
    , hio_iter(hio_iter)
    , hio_beta(hio_beta)
    , rho(rho)
    , rho_rect(rho_rect)
    , rho_strides(rho_strides)
  {}

  __host__ void run()
  {
    ER_loop();
    HIO_loop();
    ER_loop();
    // shrink_wrap();
  }

private:
  __host__ void ER_loop()
  {
    for (long k = 0; k < er_iter; ++k) {
      ER();
    }
  }

  __host__ void ER()
  {
    phase();
  }

  __host__ void HIO_loop()
  {
    for (long k = 0; k < hio_iter; ++k) {
      HIO();
    }
  }

  __host__ void HIO()
  {
  }

  __host__ void phase()
  {
    cufftComplex *rho_hat = fft(rho, rho_rect, rho_strides);
  }

  __host__ cufftComplex *fft(cufftComplex *data, Rect<3> rect, const size_t *strides)
  {
    cufftHandle plan;
    int n[3] = {int(rect.hi.x - rect.lo.x + 1), int(rect.hi.y - rect.lo.y + 1), int(rect.hi.z - rect.lo.z + 1)};

    cufftComplex *result;
    cudaMalloc((void**)&result, sizeof(cufftComplex) * rect.volume());
    if (cudaGetLastError() != cudaSuccess) {
      assert(false && "CUDA error: Failed to allocate");
    }

    if (cufftPlanMany(&plan, 3, n,
                      NULL, 1, rect.volume(),
                      NULL, 1, rect.volume(),
                      CUFFT_C2C, 1) != CUFFT_SUCCESS) {
      assert(false &&"cuFFT error: Plan creation failed");
    }

    if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
      assert(false && "cuFFT error: ExecC2C Forward failed");
    }

    if (cudaDeviceSynchronize() != cudaSuccess){
      assert(false && "CUDA error: Failed to synchronize");
    }

    cufftDestroy(plan);
  }


private:
  long er_iter;
  long hio_iter;
  double hio_beta;

  cufftComplex *rho;
  Rect<3> rho_rect;
  const size_t *rho_strides;
};

__host__
void ER_loop(long er_iter)
{
  for (long k = 0; k < er_iter; ++k) {
  }
}

__host__
int64_t gpu_phaser_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);

  const FieldAccessor<READ_WRITE, cufftComplex, 3, coord_t, Realm::AffineAccessor<cufftComplex, 3, coord_t> > rho(regions[0], FID_RHO);
  Rect<3> rho_rect = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  size_t rho_strides[3];
  cufftComplex *rho_origin = rho.ptr(rho_rect, rho_strides);

  long hio_iter = 100;
  double hio_beta = 0.1;
  long er_iter = hio_iter / 2;

  Phaser phaser(er_iter, hio_iter, hio_beta, rho_origin, rho_rect, rho_strides);

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
