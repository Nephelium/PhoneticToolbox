"""Explicit, validated alias/order configuration; original rows are immutable."""
from .rules import PhonologyRules


def default_config(analysis):
    return dict(tone_map={v:v for v in analysis.unique_tones},tone_order=analysis.unique_tones[:],
        initial_order=analysis.unique_initials[:],final_order=sorted((v for v in analysis.unique_finals if v),key=lambda v:(v[0],len(v),v))+([''] if '' in analysis.unique_finals else []),initial_map={},final_map={})


def configure(analysis,config):
    def validate_map(mapping,values):
        if not isinstance(mapping,dict) or any(k not in values or v not in values for k,v in mapping.items()):raise ValueError('m14_unknown_symbol')
        for source in mapping:
            seen=set();v=source
            while v in mapping:
                if v in seen:raise ValueError('m14_cyclic_merge')
                seen.add(v);v=mapping[v]
    validate_map(config['initial_map'],analysis.unique_initials);validate_map(config['final_map'],analysis.unique_finals)
    result=PhonologyRules().apply_symbol_aliases(analysis,config['initial_map'],config['final_map'])
    for key,values in [('initial_order',result.unique_initials),('final_order',result.unique_finals),('tone_order',result.unique_tones)]:
        if len(config[key])!=len(values) or set(config[key])!=set(values):raise ValueError('m14_order_mismatch')
    if set(config['tone_map'])!=set(result.unique_tones):raise ValueError('m14_tone_mismatch')
    if any(not isinstance(v,str) or len(v)>100 or any(ord(c)<32 for c in v) for v in config['tone_map'].values()):raise ValueError('m14_invalid_tone_name')
    tone_map={k:v.strip() or k for k,v in config['tone_map'].items()}
    return result,tone_map
