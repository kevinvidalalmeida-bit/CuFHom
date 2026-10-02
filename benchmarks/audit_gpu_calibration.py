"""Empirical, sustained roofs; not vendor peaks or hardware counters."""
from pathlib import Path
import argparse,json,time
import numpy as np
import cupy as cp

def timed(fn,repeats=9,batch=20):
    for _ in range(4):fn()
    cp.cuda.get_current_stream().synchronize()
    rows=[]
    for _ in range(repeats):
        e0,e1=cp.cuda.Event(),cp.cuda.Event();e0.record();t=time.perf_counter()
        for j in range(batch):fn()
        e1.record();e1.synchronize()
        rows.append(dict(wall_s=(time.perf_counter()-t)/batch,event_s=cp.cuda.get_elapsed_time(e0,e1)/1000/batch))
    return dict(rows=rows,median_s=float(np.median([r['event_s'] for r in rows])),
                median_wall_s=float(np.median([r['wall_s'] for r in rows])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new output path')
    n=64*1024*1024
    x=cp.full(n,1.001,cp.float32);y=cp.full(n,1.003,cp.float32);z=cp.empty_like(x)
    triad=cp.RawKernel(r'''extern "C" __global__ void triad(const float* x,const float* y,float* z,long n){
      for(long i=(long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x)z[i]=x[i]+1.23f*y[i];}''','triad')
    r=timed(lambda:triad((16384,),(256,),(x,y,z,np.int64(n))))
    r.update(logical_bytes=12*n,GB_s=12*n/r['median_s']/1e9,arithmetic_intensity=2/12)
    copy=timed(lambda:cp.copyto(z,x));copy.update(logical_bytes=8*n,GB_s=8*n/copy['median_s']/1e9)
    nl2=1024*1024;xl2=cp.full(nl2,1.001,cp.float32);yl2=cp.full(nl2,1.003,cp.float32);zl2=cp.empty_like(xl2)
    l2=timed(lambda:triad((4096,),(256,),(xl2,yl2,zl2,np.int64(nl2))),batch=100)
    l2.update(logical_bytes=12*nl2,GB_s=12*nl2/l2['median_s']/1e9)
    roofs={}
    for dtype,ctype in ((cp.float32,'float'),(cp.float64,'double')):
        out=cp.empty(262144,dtype)
        # 16 independent accumulator chains, 256 steps, counted FMA=2 FLOPs.
        code='''extern "C" __global__ void peak(TYPE* out){int i=blockIdx.x*blockDim.x+threadIdx.x;
          TYPE a[16];
          #pragma unroll
          for(int j=0;j<16;j++)a[j]=(TYPE)(1.0+0.001*j+0.0000001*i);
          #pragma unroll 1
          for(int k=0;k<256;k++){
          #pragma unroll
          for(int j=0;j<16;j++)a[j]=FMA(a[j],(TYPE)1.0000001,(TYPE)0.0000001);}
          TYPE sum=0;
          #pragma unroll
          for(int j=0;j<16;j++)sum+=a[j];out[i]=sum;}'''.replace('TYPE',ctype).replace('FMA','fmaf' if ctype=='float' else 'fma')
        kernel=cp.RawKernel(code,'peak');v=timed(lambda:kernel((1024,),(256,),(out,)),batch=8)
        v.update(FLOPs=262144*16*256*2,TFLOP_s=262144*16*256*2/v['median_s']/1e12,
                 kernel_attributes=kernel.attributes)
        assert bool(cp.all(cp.isfinite(out)))
        roofs[ctype]=v
    tiny=cp.RawKernel('extern "C" __global__ void empty(){}','empty')
    noop=timed(lambda:tiny((1,),(32,),()),batch=1000)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        tiny((1,),(32,),());stream.synchronize();stream.begin_capture()
        for i in range(1000):tiny((1,),(32,),())
        graph=stream.end_capture()
    g=timed(lambda:graph.launch(),batch=10)
    g['per_kernel_s']=g['median_s']/1000
    pinned=cp.cuda.alloc_pinned_memory(8);host=np.frombuffer(pinned,dtype=np.float64,count=1);scalar=cp.ones(1,cp.float64)
    roundtrip=timed(lambda:scalar.get(out=host),batch=100)
    pool=cp.get_default_memory_pool();tmp=cp.empty(1024,cp.float32);del tmp
    allocation_wall=[]
    for _ in range(9):
        t=time.perf_counter()
        for j in range(10000):tmp=cp.empty(1024,cp.float32);del tmp
        allocation_wall.append((time.perf_counter()-t)/10000)
    report=dict(scope='Independent empirical calibration in this GPU session; data arrays exceed L2. '
        'Logical bytes, not DRAM counters. Empty graph estimates an optimistic device launch floor. '
        'Arithmetic roofs apply to independent FMA chains, not divisions, sqrt, complex shuffles or FP64 reductions.',
        bandwidth_triad=r,bandwidth_copy=copy,bandwidth_cache_resident=l2,arithmetic=roofs,empty_launch=noop,empty_graph=g,
        scalar_roundtrip=roundtrip,pool_allocate_free_s=allocation_wall,
        device_properties={k:v for k,v in cp.cuda.runtime.getDeviceProperties(0).items() if k in
            ('l2CacheSize','multiProcessorCount','memoryClockRate','memoryBusWidth','clockRate',
             'regsPerMultiprocessor','maxThreadsPerMultiProcessor','sharedMemPerMultiprocessor')},
        gpu=str(cp.cuda.runtime.getDeviceProperties(0)['name']))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(triad_GB_s=r['GB_s'],copy_GB_s=copy['GB_s'],
        TFLOP_s={k:v['TFLOP_s'] for k,v in roofs.items()},launch_us=noop['median_s']*1e6,
        device_launch_us=g['per_kernel_s']*1e6,scalar_roundtrip_us=roundtrip['median_s']*1e6)))
if __name__=='__main__':main()
