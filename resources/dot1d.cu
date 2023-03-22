extern "C" __global__ void dot(float *A, float *B, float *C, int N, int M, int P, int loop) {
    int i = blockDim.x * threadIdx.x + blockIdx.x;
    if (i < N * P)
        C[i] = i + loop;
}