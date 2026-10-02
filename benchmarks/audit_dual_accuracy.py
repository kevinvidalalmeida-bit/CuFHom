"""Paired control of dual-norm CG, including a stricter physical stopping norm."""
from contextlib import nullcontext
from profile_g09_linear import parser,run
from cufhom_dual_metric import trial
from cufhom_dual_metric_strict import trial as strict


if __name__=='__main__':
    run(parser().parse_args(),variants={'current':({},nullcontext),
        'dual_metric':({},trial),'dual_strict_half_rtol':({},strict)})
