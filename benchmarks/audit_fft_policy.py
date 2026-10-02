"""Compare existing cuFFT component/full policies without changing defaults."""
from contextlib import nullcontext
from profile_g09_linear import parser, run

if __name__ == '__main__':
    run(parser().parse_args(), variants={
        'auto': ({}, nullcontext),
        'full': ({'cupy_rfft_batch_mode': 'full'}, nullcontext),
        'component': ({'cupy_rfft_batch_mode': 'component'}, nullcontext),
    })
