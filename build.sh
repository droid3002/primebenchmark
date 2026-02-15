#!/bin/bash

echo "===== MAX STRESS BENCH BUILDER ====="
echo "Build target:"
echo "1 = Linux"
echo "2 = Windows (MinGW cross-compile)"
read TARGET

echo "Generating source files..."

############################
# main.c
############################
cat << 'EOF' > main.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <gmp.h>

extern void gpu_sha256_stress();

int throttle = 0;

double read_cmd(const char* cmd){
    FILE* f=popen(cmd,"r");
    if(!f) return 0;
    double t=0;
    fscanf(f,"%lf",&t);
    pclose(f);
    return t;
}

double cpu_temp(){
    return read_cmd("sensors | grep 'Tctl' | awk '{print $2}' | tr -d '+°C'");
}

double gpu_temp(){
    return read_cmd("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits");
}

void maybe_throttle(){
    if(!throttle) return;
    if(cpu_temp()>85 || gpu_temp()>83)
        usleep(500000);
}

void cpu_prime(){
    while(1){
        #pragma omp parallel for schedule(dynamic)
        for(int i=2;i<5000000;i++)
            for(int j=i*i;j<5000000;j+=i);
        maybe_throttle();
    }
}

void mersenne(){
    unsigned long p=10000;
    while(1){
        mpz_t s,M,tmp;
        mpz_init(s); mpz_init(M); mpz_init(tmp);
        mpz_set_ui(s,4);
        mpz_ui_pow_ui(M,2,p);
        mpz_sub_ui(M,M,1);
        for(unsigned long i=0;i<p-2;i++){
            mpz_mul(tmp,s,s);
            mpz_sub_ui(tmp,tmp,2);
            mpz_mod(s,tmp,M);
        }
        mpz_clear(s); mpz_clear(M); mpz_clear(tmp);
        p++;
        maybe_throttle();
    }
}

int main(){

    printf("Enable thermal protection? (y/n): ");
    char c;
    scanf(" %c",&c);
    if(c=='y'||c=='Y') throttle=1;

    printf("\nSelect mode:\n");
    printf("1 = CPU Prime AVX2\n");
    printf("2 = GMP Mersenne\n");
    printf("3 = GPU SHA256\n");
    printf("4 = FULL SYSTEM\n");
    printf("> ");

    int mode;
    scanf("%d",&mode);

    if(mode==1) cpu_prime();
    if(mode==2) mersenne();
    if(mode==3) gpu_sha256_stress();
    if(mode==4){
        #pragma omp parallel sections
        {
            #pragma omp section
            cpu_prime();
            #pragma omp section
            gpu_sha256_stress();
        }
    }
}
EOF

############################
# gpu.cu
############################
cat << 'EOF' > gpu.cu
#include <cuda.h>
#include <stdint.h>

__device__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_round(uint32_t &a,uint32_t &b,uint32_t &c,uint32_t &d,
                             uint32_t &e,uint32_t &f,uint32_t &g,uint32_t &h,
                             uint32_t k,uint32_t w) {

    uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t temp1 = h + S1 + ch + k + w;
    uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temp2 = S0 + maj;

    h=g; g=f; f=e;
    e=d+temp1;
    d=c; c=b; b=a;
    a=temp1+temp2;
}

__global__ void sha256_kernel(uint32_t *data) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = data[idx];

    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a;
    uint32_t e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;

    #pragma unroll 64
    for(int i=0;i<1000;i++)
        sha256_round(a,b,c,d,e,f,g,h,0x428a2f98,x);

    data[idx]=a^b^c^d^e^f^g^h;
}

extern "C"
void gpu_sha256_stress() {

    const int N = 1<<20;
    uint32_t *d;
    cudaMalloc(&d,N*sizeof(uint32_t));

    int threads=256;
    int blocks=N/threads;

    while(1)
        sha256_kernel<<<blocks,threads>>>(d);
}
EOF

#################################
# Install + Build
#################################

if [ "$TARGET" == "1" ]; then
    echo "Installing dependencies..."
    sudo apt update
    sudo apt install -y build-essential libgmp-dev nvidia-cuda-toolkit lm-sensors

    echo "Compiling..."
    nvcc -c gpu.cu -O3 -o gpu.o
    gcc main.c gpu.o -O3 -march=native -mavx2 -fopenmp -lgmp -lcudart -L/usr/local/cuda/lib64 -o benchmark

    echo "Build complete: ./benchmark"
fi

if [ "$TARGET" == "2" ]; then
    sudo apt install -y mingw-w64
    nvcc -c gpu.cu -O3 -o gpu.o
    x86_64-w64-mingw32-gcc main.c gpu.o -O3 -lgmp -o benchmark.exe
    echo "Build complete: benchmark.exe"
fi

echo "Done."

