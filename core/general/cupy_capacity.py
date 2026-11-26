"""Pinned storage for capacity-limited library-transform solves."""
import numpy as np
from contextlib import contextmanager
import threading
from functools import lru_cache
from cufhom_core.tensors.fft import cp

_LOCAL=threading.local()
_CAPACITY_POOLS={}


class CapacityMemoryPool:
    """Separate allocation size classes so tiny arrays cannot pin field blocks.

    CuPy pools still provide ordinary stream-safe reuse. On allocation failure
    cached, unused blocks in every class are released before one retry. Live
    allocations, numeric values and field precision are never changed.
    """
    def __init__(self):
        self.pools={}
        self.used_peak=0
        self.reserved_peak=0
        self.retries=0

    def malloc(self,size):
        key=max(9,(int(size)-1).bit_length())
        pool=self.pools.get(key)
        if pool is None:
            pool=cp.cuda.MemoryPool()
            self.pools[key]=pool
        try:
            value=pool.malloc(size)
        except cp.cuda.memory.OutOfMemoryError:
            self.retries+=1
            self.free_all_blocks()
            value=pool.malloc(size)
        self.used_peak=max(self.used_peak,self.used_bytes())
        self.reserved_peak=max(self.reserved_peak,self.total_bytes())
        return value

    def used_bytes(self):return sum(p.used_bytes() for p in self.pools.values())
    def total_bytes(self):return sum(p.total_bytes() for p in self.pools.values())
    def free_all_blocks(self):
        for pool in self.pools.values():
            pool.free_all_blocks()

    def record(self):
        return dict(strategy='allocation_size_classes',classes=len(self.pools),
                    pool_used_peak_gib=self.used_peak/2**30,pool_reserved_peak_gib=self.reserved_peak/2**30,
                    cache_trim_retries=self.retries)


def current_memory_pool():
    return getattr(_LOCAL,'pool',None) or cp.get_default_memory_pool()


@contextmanager
def capacity_memory_pool():
    # Completed solver contexts can contain unreachable Python closures.
    # Reclaim only dead objects before reserving another large working set.
    import gc
    gc.collect()
    previous=getattr(_LOCAL,'pool',None)
    device=cp.cuda.runtime.getDevice()
    pool=_CAPACITY_POOLS.get(device)
    if pool is None:
        pool=CapacityMemoryPool()
        _CAPACITY_POOLS[device]=pool
    # Generated-kernel twiddles and other process caches can own allocations
    # beyond one solve. Their owning pools must outlive those arrays, just as
    # the default CuPy pool does. Reuse the per-device pool across calls.
    pool.free_all_blocks()
    pool.used_peak=pool.used_bytes()
    pool.reserved_peak=pool.total_bytes()
    pool.retries=0
    cp.cuda.get_current_stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    _LOCAL.pool=pool
    try:
        with cp.cuda.using_allocator(pool.malloc):
            yield pool
    finally:
        cp.cuda.get_current_stream().synchronize()
        pool.free_all_blocks()
        _LOCAL.pool=previous


@lru_cache(maxsize=None)
def _device_capacity(device):
    # Physical capacity is constant for a device; free/reusable memory is
    # deliberately queried afresh by the residency and assembly selectors.
    return int(cp.cuda.runtime.getDeviceProperties(device)['totalGlobalMem'])


def library_capacity_required(shape, dtype=np.float32):
    """Eight full Mandel fields exceed capacity minus a transient reserve."""
    itemsize=np.dtype(dtype).itemsize
    return 48*itemsize*int(np.prod(shape)) > _device_capacity(cp.cuda.runtime.getDevice())-512*2**20


def host_iterate_required(shape, dtype=np.float32):
    """Six full Mandel fields exceed capacity minus the same reserve."""
    return 36*np.dtype(dtype).itemsize*int(np.prod(shape)) > _device_capacity(cp.cuda.runtime.getDevice())-512*2**20


def pinned_array(shape,dtype):
    dtype=np.dtype(dtype)
    count=int(np.prod(shape))
    memory=cp.cuda.alloc_pinned_memory(count*dtype.itemsize)
    return np.frombuffer(memory,dtype,count).reshape(shape)


def pinned_copy(value):
    host=pinned_array(value.shape,value.dtype)
    value.get(out=host)
    return host
