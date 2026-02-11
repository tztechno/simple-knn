/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group
 */

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <vector>
#include <limits>
#include <cmath>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

struct CustomMin
{
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return make_float3(
            fminf(a.x, b.x),
            fminf(a.y, b.y),
            fminf(a.z, b.z)
        );
    }
};

struct CustomMax
{
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return make_float3(
            fmaxf(a.x, b.x),
            fmaxf(a.y, b.y),
            fmaxf(a.z, b.z)
        );
    }
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
    float invx = 1.0f / (maxx.x - minn.x + 1e-12f);
    float invy = 1.0f / (maxx.y - minn.y + 1e-12f);
    float invz = 1.0f / (maxx.z - minn.z + 1e-12f);

    uint32_t x = prepMorton((uint32_t)(((coord.x - minn.x) * invx) * 1023.0f));
    uint32_t y = prepMorton((uint32_t)(((coord.y - minn.y) * invy) * 1023.0f));
    uint32_t z = prepMorton((uint32_t)(((coord.z - minn.z) * invz) * 1023.0f));

    return x | (y << 1) | (z << 2);
}

__global__ void coord2MortonKernel(int P, const float3* points,
                                   float3 minn, float3 maxx,
                                   uint32_t* codes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
    float3 minn;
    float3 maxx;
};

__global__ void boxMinMax(uint32_t P,
                          float3* points,
                          uint32_t* indices,
                          MinMax* boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float inf = std::numeric_limits<float>::max();

    MinMax me;

    if (idx < P)
    {
        me.minn = points[indices[idx]];
        me.maxx = points[indices[idx]];
    }
    else
    {
        me.minn = make_float3(inf, inf, inf);
        me.maxx = make_float3(-inf, -inf, -inf);
    }

    __shared__ MinMax redResult[BOX_SIZE];

    redResult[threadIdx.x] = me;
    __syncthreads();

    for (int off = BOX_SIZE / 2; off >= 1; off >>= 1)
    {
        if (threadIdx.x < off)
        {
            MinMax other = redResult[threadIdx.x + off];

            redResult[threadIdx.x].minn.x =
                fminf(redResult[threadIdx.x].minn.x, other.minn.x);
            redResult[threadIdx.x].minn.y =
                fminf(redResult[threadIdx.x].minn.y, other.minn.y);
            redResult[threadIdx.x].minn.z =
                fminf(redResult[threadIdx.x].minn.z, other.minn.z);

            redResult[threadIdx.x].maxx.x =
                fmaxf(redResult[threadIdx.x].maxx.x, other.maxx.x);
            redResult[threadIdx.x].maxx.y =
                fmaxf(redResult[threadIdx.x].maxx.y, other.maxx.y);
            redResult[threadIdx.x].maxx.z =
                fmaxf(redResult[threadIdx.x].maxx.z, other.maxx.z);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        boxes[blockIdx.x] = redResult[0];
}

__device__ float distBoxPoint(const MinMax& box, const float3& p)
{
    float3 diff = make_float3(0.f, 0.f, 0.f);

    if (p.x < box.minn.x || p.x > box.maxx.x)
        diff.x = fminf(fabsf(p.x - box.minn.x),
                       fabsf(p.x - box.maxx.x));

    if (p.y < box.minn.y || p.y > box.maxx.y)
        diff.y = fminf(fabsf(p.y - box.minn.y),
                       fabsf(p.y - box.maxx.y));

    if (p.z < box.minn.z || p.z > box.maxx.z)
        diff.z = fminf(fabsf(p.z - box.minn.z),
                       fabsf(p.z - box.maxx.z));

    return diff.x * diff.x +
           diff.y * diff.y +
           diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref,
                            const float3& point,
                            float* knn)
{
    float dx = point.x - ref.x;
    float dy = point.y - ref.y;
    float dz = point.z - ref.z;

    float dist = dx*dx + dy*dy + dz*dz;

    for (int j = 0; j < K; j++)
    {
        if (knn[j] > dist)
        {
            float tmp = knn[j];
            knn[j] = dist;
            dist = tmp;
        }
    }
}

__global__ void boxMeanDist(uint32_t P,
                            float3* points,
                            uint32_t* indices,
                            MinMax* boxes,
                            float* dists)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    float inf = std::numeric_limits<float>::max();

    float3 point = points[indices[idx]];
    float best[3] = { inf, inf, inf };

    for (int i = max(0, idx - 3);
         i <= min((int)P - 1, idx + 3); i++)
    {
        if (i == idx) continue;
        updateKBest<3>(point, points[indices[i]], best);
    }

    float reject = best[2];

    best[0] = best[1] = best[2] = inf;

    int num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;

    for (int b = 0; b < num_boxes; b++)
    {
        MinMax box = boxes[b];
        float dist = distBoxPoint(box, point);

        if (dist > reject || dist > best[2])
            continue;

        int start = b * BOX_SIZE;
        int end = min((int)P, start + BOX_SIZE);

        for (int i = start; i < end; i++)
        {
            if (i == idx) continue;
            updateKBest<3>(point, points[indices[i]], best);
        }
    }

    dists[indices[idx]] =
        (best[0] + best[1] + best[2]) / 3.0f;
}

void SimpleKNN::knn(int P,
                    float3* points,
                    float* meanDists)
{
    float3* result;
    cudaMalloc(&result, sizeof(float3));

    size_t temp_storage_bytes = 0;

    float3 init = make_float3(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );

    cub::DeviceReduce::Reduce(nullptr,
        temp_storage_bytes,
        points,
        result,
        P,
        CustomMin(),
        init);

    thrust::device_vector<char> temp_storage(temp_storage_bytes);

    cub::DeviceReduce::Reduce(
        temp_storage.data().get(),
        temp_storage_bytes,
        points,
        result,
        P,
        CustomMin(),
        init);

    float3 minn;
    cudaMemcpy(&minn, result,
               sizeof(float3),
               cudaMemcpyDeviceToHost);

    cub::DeviceReduce::Reduce(
        temp_storage.data().get(),
        temp_storage_bytes,
        points,
        result,
        P,
        CustomMax(),
        make_float3(-init.x,-init.y,-init.z));

    float3 maxx;
    cudaMemcpy(&maxx, result,
               sizeof(float3),
               cudaMemcpyDeviceToHost);

    thrust::device_vector<uint32_t> morton(P);
    thrust::device_vector<uint32_t> morton_sorted(P);

    coord2MortonKernel<<<(P + 255) / 256, 256>>>(
        P, points, minn, maxx,
        morton.data().get());

    thrust::device_vector<uint32_t> indices(P);
    thrust::sequence(indices.begin(), indices.end());

    thrust::device_vector<uint32_t> indices_sorted(P);

    cub::DeviceRadixSort::SortPairs(
        nullptr,
        temp_storage_bytes,
        morton.data().get(),
        morton_sorted.data().get(),
        indices.data().get(),
        indices_sorted.data().get(),
        P);

    temp_storage.resize(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_storage.data().get(),
        temp_storage_bytes,
        morton.data().get(),
        morton_sorted.data().get(),
        indices.data().get(),
        indices_sorted.data().get(),
        P);

    uint32_t num_boxes =
        (P + BOX_SIZE - 1) / BOX_SIZE;

    thrust::device_vector<MinMax> boxes(num_boxes);

    boxMinMax<<<num_boxes, BOX_SIZE>>>(
        P,
        points,
        indices_sorted.data().get(),
        boxes.data().get());

    boxMeanDist<<<num_boxes, BOX_SIZE>>>(
        P,
        points,
        indices_sorted.data().get(),
        boxes.data().get(),
        meanDists);

    cudaFree(result);
}

