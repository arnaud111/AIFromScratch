extern "C" __global__ void dot(float *A, float *B, float *C, int N, int M, int P, int loop) {
    int c_i = blockDim.x * threadIdx.x + blockIdx.x;
    int real_i = c_i + loop;
    if (real_i < N * P + loop) {
        int col = real_i % P;
        int row = real_i % P;
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += A[row * M + i] * B[i * P + col];
        }
        C[c_i] = sum;
    }
}