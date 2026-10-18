"""Opt-in source perturbation; strengths are not measured acoustic percentages."""
import math

DEFAULT_PERTURBATION={'jitter':10.,'shimmer':0.,'seed':1}
PERTURBATION_METHOD='m10/source-1'


def validate_perturbation(value=None):
    if value is None:return None
    if not isinstance(value,dict) or set(value)-set(DEFAULT_PERTURBATION):
        raise ValueError('无效的声源抖动设置')
    result={}
    for key in ('jitter','shimmer'):
        number=value.get(key,DEFAULT_PERTURBATION[key])
        if type(number) not in (int,float) or not math.isfinite(number) or not 0<=number<=100:
            raise ValueError('声源抖动强度应为 0–100')
        result[key]=float(number)
    seed=value.get('seed',1)
    if type(seed) is not int or not 0<=seed<=4294967295:raise ValueError('随机种子应为 0–4294967295 的整数')
    return {**result,'seed':seed}
