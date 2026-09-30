// Diagnostic C2R -> compact material tangent -> R2C fusion on 240-point lines.
// Compiled only by the isolated benchmark; not linked into production.
#include <cuda_runtime.h>
#include <cufftdx.hpp>
#include <algorithm>
#include <cstdint>

#ifndef FFT_EPT
#define FFT_EPT 30
#endif
#ifndef FFT_FPB
#define FFT_FPB 4
#endif
using namespace cufftdx;
using Base = decltype(Block()+Size<240>()+Precision<float>()+
    ElementsPerThread<FFT_EPT>()+FFTsPerBlock<FFT_FPB>()+
    RealFFTOptions<complex_layout::natural, real_mode::normal>()+SM<1200>());
using Inverse = decltype(Base()+Type<fft_type::c2r>()+Direction<fft_direction::inverse>());
using Forward = decltype(Base()+Type<fft_type::r2c>()+Direction<fft_direction::forward>());
using Complex = Inverse::value_type;
static_assert(!Inverse::requires_workspace && !Forward::requires_workspace);
static_assert(Inverse::stride==Forward::stride);
static_assert(Inverse::output_ept==Forward::input_ept);
static_assert(Inverse::storage_size>=Forward::storage_size);
static constexpr int fft_shared_bytes = std::max(Inverse::shared_memory_size, Forward::shared_memory_size);
#ifdef FFT_SHARED
static constexpr int shared_bytes = fft_shared_bytes+6*240*FFT_FPB*sizeof(float);
#else
static constexpr int shared_bytes = fft_shared_bytes;
#endif

__global__ void line_tangent(const Complex* input, Complex* output,
    const float* u, const float* a, const float* b, const int* ids,
    const float* table, float bulk, int lines) {
    extern __shared__ __align__(16) unsigned char shared[];
    auto smem = reinterpret_cast<Complex*>(shared);
    const int line = blockIdx.x*FFT_FPB+threadIdx.y;
    const long long count = (long long)lines*121;
    const long long n = (long long)lines*240;
#ifdef FFT_SHARED
    Complex scratch[Inverse::storage_size];
    auto local = reinterpret_cast<float*>(shared+fft_shared_bytes);
#else
    Complex values[6][Inverse::storage_size];
#endif
    #pragma unroll
    for(int c=0;c<6;c++) {
#ifdef FFT_SHARED
        auto data = scratch;
#else
        auto data = values[c];
#endif
        #pragma unroll
        for(int j=0;j<Inverse::storage_size;j++)data[j]=Complex{0,0};
        if(Inverse::working_group::is_thread_active()) {
            #pragma unroll
            for(int j=0;j<Inverse::input_ept;j++) {
                const int k=threadIdx.x+j*Inverse::stride;
                if(line<lines && k<121)data[j]=input[c*count+(long long)line*121+k];
            }
        }
        Inverse().execute(data,smem);
#ifdef FFT_SHARED
        if(Inverse::working_group::is_thread_active()) {
            #pragma unroll
            for(int j=0;j<Inverse::output_ept;j++) {
                const int k=threadIdx.x+j*Inverse::stride;
                if(k<240)local[(c*FFT_FPB+threadIdx.y)*240+k]=reinterpret_cast<float*>(data)[j];
            }
        }
#endif
        __syncthreads(); // shared workspace is reused by next component
    }
    if(Inverse::working_group::is_thread_active()) {
        #pragma unroll
        for(int j=0;j<Inverse::output_ept;j++) {
            const int k=threadIdx.x+j*Inverse::stride;
            if(line<lines && k<240) {
                const long long voxel=(long long)line*240+k;
                float v[6],w[6];
                #pragma unroll
                for(int c=0;c<6;c++) {
#ifdef FFT_SHARED
                    v[c]=local[(c*FFT_FPB+threadIdx.y)*240+k];
#else
                    v[c]=reinterpret_cast<float*>(values[c])[j];
#endif
                }
                if(ids[voxel]) {
                    #pragma unroll
                    for(int c=0;c<6;c++) {
                        w[c]=0;
                        #pragma unroll
                        for(int d=0;d<6;d++)w[c]+=table[(long long)ids[voxel]*36+c*6+d]*v[d];
                    }
                } else {
                    float dot=0, tr=v[0]+v[1]+v[2];
                    #pragma unroll
                    for(int c=0;c<6;c++)dot+=u[c*n+voxel]*v[c];
                    #pragma unroll
                    for(int c=0;c<6;c++)w[c]=a[voxel]*v[c]+b[voxel]*u[c*n+voxel]*dot+
                        (c<3?(bulk-a[voxel]/3)*tr:0);
                }
                #pragma unroll
                for(int c=0;c<6;c++) {
#ifdef FFT_SHARED
                    local[(c*FFT_FPB+threadIdx.y)*240+k]=w[c];
#else
                    reinterpret_cast<float*>(values[c])[j]=w[c];
#endif
                }
            }
        }
    }
    #pragma unroll
    for(int c=0;c<6;c++) {
#ifdef FFT_SHARED
        auto data = scratch;
        if(Forward::working_group::is_thread_active()) {
            #pragma unroll
            for(int j=0;j<Forward::input_ept;j++) {
                const int k=threadIdx.x+j*Forward::stride;
                if(k<240)reinterpret_cast<float*>(data)[j]=local[(c*FFT_FPB+threadIdx.y)*240+k];
            }
        }
#else
        auto data = values[c];
#endif
        Forward().execute(data,smem);
        __syncthreads();
        if(Forward::working_group::is_thread_active()) {
            #pragma unroll
            for(int j=0;j<Forward::output_ept;j++) {
                const int k=threadIdx.x+j*Forward::stride;
                if(line<lines && k<121)output[c*count+(long long)line*121+k]=data[j];
            }
        }
    }
}

extern "C" int candidate_info(int* out) {
    out[0]=Inverse::block_dim.x; out[1]=Inverse::block_dim.y;
    out[2]=shared_bytes; out[3]=Inverse::storage_size;
    cudaFuncAttributes attributes;
    auto err=cudaFuncGetAttributes(&attributes,line_tangent);
    if(err!=cudaSuccess)return int(err);
    out[4]=attributes.numRegs; out[5]=attributes.localSizeBytes;
    return int(cudaFuncSetAttribute(line_tangent,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
}
extern "C" int candidate_launch(void* input, void* output, void* u, void* a,
    void* b, void* ids, void* table, float bulk, int lines, uintptr_t stream) {
    line_tangent<<<(lines+FFT_FPB-1)/FFT_FPB,Inverse::block_dim,shared_bytes,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<Complex*>(input),static_cast<Complex*>(output),
        static_cast<float*>(u),static_cast<float*>(a),static_cast<float*>(b),
        static_cast<int*>(ids),static_cast<float*>(table),bulk,lines);
    return int(cudaGetLastError());
}
