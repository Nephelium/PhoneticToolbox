"""M14 fixed handler. Input bytes are already owner/hash/expiry-authorized."""
from dataclasses import asdict
import hashlib
import json
from .m14_import import load
from phonetic_core.transcription.phonology import PhonologyRules
from phonetic_core.transcription.phonology.config import default_config,configure
from phonetic_core.transcription.phonology.export import export
from phonetic_core.transcription.phonology.presentation import WorkbenchRenderer


def execute(raw,name,config,cancelled=lambda:False):
    if cancelled():raise InterruptedError('m14_cancelled')
    revision=config.get('computation_revision','m14/1')
    if config['action']=='inspect':
        from .m14_table import inspect
        return {'m14-preview.json':json.dumps(inspect(raw,name,config),ensure_ascii=False).encode()}
    if revision=='m14/2':
        from .m14_table import load_v2
        rows,diagnostics=load_v2(raw,name,config)
    else:rows,diagnostics=load(raw,name,config['skip_first_row'])
    rules=PhonologyRules(computation_revision=revision);analysis=rules.analyze(rows,config['consonant_only_as_zero_initial'])
    data=dict(schema_version='m14/1',input_sha256=hashlib.sha256(raw).hexdigest(),analysis=asdict(analysis),diagnostics=diagnostics,
        single_consonants=[asdict(r) for r in rules.find_single_consonant_rows(rows)],config=default_config(analysis),source_ids=['PENDING-PHONOLOGY'])
    data['computation_revision']=revision
    if config['action']=='preview':return {'m14-preview.json':json.dumps(data,ensure_ascii=False).encode()}
    remapped,tone_map=configure(analysis,config['settings'])
    # All three are generated in memory before the common writer receives any result.
    return export(remapped,tone_map,config['settings']['tone_order'],config['settings']['initial_order'],config['settings']['final_order'],renderer=WorkbenchRenderer(config['font']),cancelled=cancelled)
