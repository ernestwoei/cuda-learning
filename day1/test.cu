#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <memory>
using namespace std;

__global__ void gpuHello();
__global__ void gpuVectorAdd(float *a, float *b, float *c, size_t N);

#define TILE_SIZE 16
__global__ void blur3x3_tiled(  const float* input,
                                float* output,
                                int width,
                                int height)
{
    // 16x16 output tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global coordinates of this thread's output pixel
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 18x18 shared-memory tile:
    // 16x16 center + 1-pixel halo on each side
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];

    // ------------------------------------------------
    // 1. Load the center 16x16 pixels
    // ------------------------------------------------
    if (x < width && y < height)
        tile[ty+1][tx+1] = input[y*width + x];
    else
        tile[ty+1][tx+1] = 0.0f;

    // ------------------------------------------------
    // 2. Load top and bottom halo
    // ------------------------------------------------
    if (ty == 0)
    {
        // Top halo
        int halo_top_y = y-1;

        if (halo_top_y >= 0 && x < width)
            tile[0][tx+1] = input[halo_top_y * width+x];
        else
            tile[0][tx+1] = 0.0f;

        // Bottom halo
        int halo_bottom_y = y + TILE_SIZE;

        if (halo_bottom_y < height && x < width)
            tile[TILE_SIZE+1][tx+1] = input[halo_bottom_y * width+x];
        else
            tile[TILE_SIZE+1][tx+1] = 0.0f;
    }

    // ------------------------------------------------
    // 3. Load left and right halo
    // ------------------------------------------------
    if (tx == 0)
    {
        // left halo
        int halo_left_x = x-1;

        if (halo_left_x >= 0 && y < height)
            tile[ty+1][0] = input[y*width + halo_left_x];
        else
            tile[ty+1][0] = 0.0f;

        // right halo
        int halo_right_x = x + TILE_SIZE;

        if (halo_right_x < width && y < height)
            tile[ty+1][TILE_SIZE+1] = input[y*width + halo_right_x];
        else
            tile[ty+1][TILE_SIZE+1] = 0.0f;
    }

    // ------------------------------------------------
    // 4. Load the four corners
    // ------------------------------------------------
    if (tx == 0 && ty == 0)
    {
        // Top-left
        if (x > 0 && y > 0)
            tile[0][0] = input[(y-1)*width + (x-1)];
        else
            tile[0][0] = 0.0f;

        // Top-right
        if (x + TILE_SIZE < width && y > 0)
            tile[0][TILE_SIZE + 1] = input[(y-1)*width + (x+TILE_SIZE)];
        else
            tile[0][TILE_SIZE + 1] = 0.0f;

        // Bottom-left
        if (x > 0 && y + TILE_SIZE < height)
            tile[TILE_SIZE + 1][0] = input[(y+TILE_SIZE)*width + (x-1)];
        else
            tile[TILE_SIZE + 1][0] = 0.0f;

        // Bottom-right
        if (x + TILE_SIZE < width && y + TILE_SIZE < height)
            tile[TILE_SIZE + 1][TILE_SIZE + 1] = input[(y+TILE_SIZE)*width + (x+TILE_SIZE)];
        else
            tile[TILE_SIZE + 1][TILE_SIZE + 1] = 0.0f;
    }

    // ------------------------------------------------
    // 5. Wait until the entire is ready
    // ------------------------------------------------
    __syncthreads();

    // ------------------------------------------------
    // 6. Perform the 3x3 blur using shared memory
    // ------------------------------------------------
    if (x < width && y < height)
    {
        float sum = 0.0f;

        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                sum += tile[ty+1+dy][tx+1+dx];
            }
        }
        output[y*width + x] = sum/9.0f;
    }
}

// Custom deleter functor (callable object)
// empty struct 0 bytes of memory inside unique_ptr due to C++ optimization Empty Base Optimization (EBO)
// better than standard C-style function pointer example
// rawptr size is 8 bytes
// compile time inlining, compiler knows what code inside operator() needs to run at compile time, inline cudaFree into the destructor eliminating CPU overhead of a function lookup
struct CudaDeleter {
    void operator()(void* ptr) const { // void* can be reused for any data type
        if (ptr) {
            cout << "[Deleter] Automatically freeing GPU memory via cudaFree\n";
        }
    }
};

// smart pointer type alias
template<typename T>
using CudaUniquePtr = unique_ptr<T, CudaDeleter>;

// allocation helper function
template<typename T>
CudaUniquePtr<T> make_cuda_unique(size_t N) {
    T* rawptr = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&rawptr), N * sizeof(T));

    if (err != cudaSuccess) {
        cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << endl;
        return nullptr;
    }
}

template<class T>
void randomlyFillVector(vector<T> &v)
{
    random_device rd;   // create random number tool to get a unique seed every time it's run
    mt19937 gen(rd());  // create a fast pseudo-random number generator, Mersenne Twister
    uniform_real_distribution<T> dis(0.0f,1.0f);    // shape input to uniform distribution [0.0,1.0)

    generate(v.begin(), v.end(), [&](){ return dis(gen); });    // lambda captures external variables by reference
}

template<class T>
void allocateDeviceMemory(T* &d_data, size_t N)
{
    cudaError_t err = cudaMalloc( reinterpret_cast<void**>(&d_data), N*sizeof(T));
    if (err != cudaSuccess)
    {
        cerr << "CUDA Malloc failed:" << cudaGetErrorString(err) << endl;
        return;
    }
}

template<class T>
void copyToDeviceMemory(T *h_data, T *d_data, size_t N)
{
    cudaMemcpy(d_data, h_data, N * sizeof(T), cudaMemcpyHostToDevice);
}

template<class T>
void copyToHostMemory(T *h_data, T *d_data, size_t N)
{
    cudaMemcpy(h_data, d_data, N * sizeof(T), cudaMemcpyDeviceToHost);
}

template<class T>
void cpuVectorAdd(const vector<T> &v_a, const vector<T> &v_b, vector<T> &v_c)
{
    transform(v_a.begin(), v_a.end(), v_b.begin(), v_c.begin(), std::plus<>{});
}

template<class T>
T sumOfArrayAbsoluteDifferences(T *left, T *right, size_t N)
{
    return transform_reduce(left, left+N, right, 0.0f, std::plus<>(), [](T x, T y) { return std::abs(x - y); });
}

int main()
{
    cout << "Hello from the CPU!" << endl;

    int numBlocks = 2, numThreadsPerBlocks = 4;
    gpuHello<<<numBlocks, numThreadsPerBlocks>>>();

    int N = 1000;
    numThreadsPerBlocks = 256;

    // Allocate vector, going to use v.data()
    vector<float> h_v_a(N, 0.0f), h_v_b(N, 0.0f), h_v_c(N, 0.0f);
    randomlyFillVector(h_v_a);
    randomlyFillVector(h_v_b);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr; // device pointer
    allocateDeviceMemory(d_A, N);
    allocateDeviceMemory(d_B, N);
    allocateDeviceMemory(d_C, N);

    copyToDeviceMemory(h_v_a.data(), d_A, N);
    copyToDeviceMemory(h_v_b.data(), d_B, N);

    // numBlocks = ceil(static_cast<float>(N)/static_cast<float>(numThreadsPerBlocks)); // don't use C-style cast as they are unsafe, i.e., could cast away const or incompatible pointer conversion
    // nicer way
    numBlocks = (N + numThreadsPerBlocks - 1)/numThreadsPerBlocks;
    
    gpuVectorAdd<<<numBlocks, numThreadsPerBlocks>>>(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();

    cpuVectorAdd(h_v_a, h_v_b, h_v_c);

    vector<float> d2h_v_C(N, 0.0);
    copyToHostMemory(d2h_v_C.data(), d_C, N);
    cout << "Sum of Absolute Differences of Array: " << sumOfArrayAbsoluteDifferences(h_v_c.data(), d2h_v_C.data(), N) << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

// CUDA kernels
// print hello kernel
__global__ void gpuHello()
{
    printf("Hello from the GPU!\n");
}

// vector add kernel
__global__ void gpuVectorAdd(float *a, float *b, float *c, size_t N)
{
    // 1. Figure out which element this thread owns
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. Make sure that element actually exists
    if (i < N)
    {
        // 3. Add the corresponding elements
        c[i] = a[i] + b[i];
    }
}