"""Freeze a control's next proposal and compare gradients to finite differences.

Saved trial tables are diagnostic, unaccepted states; they are not generator
checkpoints and never replace the independently certified geometry.
"""
import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np

from cufhom.pipeline.sam_paper_reference import (
    objective, relax_paper_reference, rotational_inertia, sphere_exp,
)
from cufhom.pipeline.sam_staged_insertion import Settings, propose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint_bytes = args.checkpoint.read_bytes()
    with np.load(BytesIO(checkpoint_bytes), allow_pickle=False) as saved:
        accepted = saved['state'].copy()
        settings = Settings(**json.loads(str(saved['settings_json'])))
        state = json.loads(str(saved['resume_json']))
        a2, a4 = saved['A2'].copy(), saved['A4'].copy()
    rng = np.random.default_rng()
    rng.bit_generator.state = state['rng_state']
    count = min(state['batch_size'], settings.target_count-len(accepted))
    trial = propose(accepted, count, a2, settings, rng, a4)
    started = time.perf_counter()
    relaxed, info = relax_paper_reference(
        trial, settings.box, a2, target_a4=a4, optimizer=settings.optimizer,
        maxiter=args.iterations, min_gap=settings.min_gap,
        orientation_weight=settings.orientation_weight,
        overlap_tolerance=settings.overlap_tolerance,
        a2_tolerance=settings.a2_tolerance, a4_tolerance=settings.a4_tolerance)
    elapsed = time.perf_counter()-started
    c, u = relaxed[:, :3]/settings.diameter, relaxed[:, 3:6]
    domain = np.full(3, settings.box/settings.diameter)
    length = settings.length/settings.diameter
    diameter = 1+settings.min_gap/settings.diameter
    kwargs = dict(weight=settings.orientation_weight,target_a4=a4,contact_diameter=diameter)
    value, gc, gu, _, _ = objective(c, u, length, domain, **kwargs)
    eps = rotational_inertia(length, 1.)
    projected = gu-u*np.sum(gu*u,axis=1)[:,None]
    probes = []
    random = np.random.default_rng(920071)
    for name in ('negative_gradient','random_0','random_1','random_2'):
        dc = -gc if name=='negative_gradient' else random.normal(size=c.shape)
        du = -projected/eps if name=='negative_gradient' else random.normal(size=u.shape)
        du -= u*np.sum(u*du,axis=1)[:,None]
        norm = np.sqrt(np.sum(dc*dc)+eps*np.sum(du*du))
        dc, du = dc/norm, du/norm
        derivative = float(np.sum(gc*dc)+np.sum(gu*du))
        comparisons = []
        for h in (1e-3,1e-4,1e-5,1e-6,1e-7):
            plus = objective(c+h*dc,sphere_exp(u,h*du),length,domain,**kwargs)[0]
            minus = objective(c-h*dc,sphere_exp(u,-h*du),length,domain,**kwargs)[0]
            fd = (plus-minus)/(2*h)
            comparisons.append(dict(h=h,finite_difference=fd,
                                    difference=fd-derivative,
                                    forward_slope=(plus-value)/h,
                                    backward_slope=(value-minus)/h))
        probes.append(dict(direction=name,analytic_derivative=derivative,comparisons=comparisons))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), proposed_fibers=trial,
                        relaxed_fibers=relaxed, isotropic_packing_accepted=False,
                        purpose='unaccepted diagnostic trial',settings_json=json.dumps(asdict(settings)),
                        A2=a2,A4=a4,source_rng_json=json.dumps(state))
    result = dict(source_checkpoint=str(args.checkpoint), source_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                  accepted_count=len(accepted),trial_count=len(trial),settings=asdict(settings),
                  relaxation_wall_seconds=elapsed,optimization=info,
                  raw_position_gradient_norm=float(np.linalg.norm(gc)),
                  raw_tangent_gradient_norm=float(np.linalg.norm(projected)),
                  inertia_scaled_gradient_norm=float(np.sqrt(np.sum(gc*gc)+np.sum(projected*projected)/eps)),
                  probes=probes)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['accepted_count','trial_count',
        'relaxation_wall_seconds','raw_position_gradient_norm',
        'raw_tangent_gradient_norm','inertia_scaled_gradient_norm','probes']},indent=2))


if __name__=='__main__':
    main()
