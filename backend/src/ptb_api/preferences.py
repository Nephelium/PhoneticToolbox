"""Account-owned settings. No workspace lease, file contents, or automatic DDL."""
from datetime import datetime
import json
import math
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field, JsonValue
from psycopg.types.json import Jsonb
from .models import WireModel

ModuleId = Literal['APP','M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15','M16','M17','M18']
FIELDS = {
 'APP': 'theme palette collapsed buttonAppearance waveformAppearance pageScale backgroundAppearance customPaletteColor homeIconScale fonts panelWidths',
 'M01': 'settings sameDirectory recursive parameters showBoth showSpectrogram max_duration_s audio_channel egg panelWidths',
 'M02': 'groups reaper correction recursive showBoth showSpectrogram panelWidths',
 'M03': 'autoDbEnabled order signal_mode flip_channels micro_width_ms gci_method goi_method peak_prominence valley_prominence auto_prominence highpass_cutoff lowpass_cutoff spec_window_ms spec_vmin spec_vmax keep_praat_f0 keep_gci_f0 keep_reaper_f0 f0_min_hz f0_max_hz silence_threshold generate_images batchConfig showBoth showSpectrogram panelWidths',
 'M04': 'draft showBoth showSpectrogram panelWidths',
 'M05': 'camera microphone filter cutoff fps delegate mirror quality saveAll saveVideo saveAnimation panelWidths',
 'M06': 'duration sample_rate fade_in fade_out smooth f0_range f0_method f0_transform selectedPreset windowMs curveOverrides panelWidths',
 'M07': 'target_sample_rate f0_backend min_f0_hz max_f0_hz f0_frame_interval_ms frame_length frame_shift lpc_order preemphasis window_name negative_peak_threshold pulse_inner_periods pulse_outer_periods trim_silence silence_threshold_db silence_padding_ms voiced_margin_ms step_count energy_match normalize_to_source output_peak_limit alignment count kind reverse axisRange panelWidths',
 'M08': 'speed pitch_ratio pitch_hz offset minInput maxInput ymin ymax refInput refs directPlayback panelWidths',
 'M09': 'invert_image vocoder algorithm reconstruction_profile inference_device phase_initialization frequency_axis auto_parameters dynamic_range min_db max_db win_length_ms n_iter target_sr seed durationInput freqStartInput freqEndInput audioFft channel panelWidths',
 'M10': 'frame panelWidths webNoticeSeen',
 'M11': 'config transcriptSource panelWidths',
 'M12': 'suffix showOpen showWidth sidebar settingsOpen phoneSplit durationInput nudgeStep showBoth showSpectrogram panelWidths',
 'M13': 'standard display layout hanziSize ipaSize ipaSizeUserSet gap lineHeight bold italic underline showTone ipaColor hanziColor hanziFont equalWidth panelWidths',
 'M14': 'options panelWidths',
 'M15': 'paradigm title introText interTrialText introLargeText showInterTrialText largeText advanceMode autoPlay showFilename interTrialInterval isi useBeep validKeys seed resultFormat panelWidths',
 'M16': 'config outputDevice playChannel playVolume showSpectrum showAllChannels displayScale strength gainDb subtype rawExport exportMode leftWidth panelWidths',
 'M17': 'system introductions textSize textSizeVersion editorHeight playOnClick panelWidths',
 'M18': 'citeStyle language details showObtained readerZoom readerFit readerText readerSidebar annotationColor exportLanguage exportAnnotated paperAutoReceive paperInterval panelWidths',
}
FIELDS = {module: text.split() for module, text in FIELDS.items()}
NESTED = {
 ('M01','settings'): 'silence_threshold energy_window_ms frameshift_ms windowsize_ms smooth_win_size lip_smooth_win_size only_voiced n_periods num_formants max_formant min_f0 max_f0 reaper_hilbert reaper_no_highpass',
 ('M01','egg'): 'enabled values',
 ('M01','egg.values'): 'egg_channel storage smooth_ms max_gap_ms derived highpass_cutoff lowpass_cutoff gci_method goi_method auto_prominence peak_prominence valley_prominence silence_threshold',
 ('M02','groups[]'): 'id title parameters',
 ('M03','batchConfig'): 'mode signal_mode flip_channels roi_start roi_end micro_center micro_width_ms gci_method goi_method peak_prominence valley_prominence auto_prominence highpass_cutoff lowpass_cutoff spec_window_ms spec_vmin spec_vmax keep_praat_f0 keep_gci_f0 keep_reaper_f0 f0_policy f0_min_hz f0_max_hz glottal_movement silence_threshold generate_images lp_order export_policy',
 ('M04','draft'): 'order freq_max_hz amp_min_db amp_max_db dynamic_y',
 ('M06','f0_transform'): 'preset offset_hz',
 ('M07','axisRange'): 'minimum maximum',
 ('M10','frame'): 'view audio sectionCount f0 source keepVowel pose',
 ('M10','frame.view'): 'mode teeth head nose labels fullModel',
 ('M10','frame.audio'): 'device volume gain_db',
 ('M10','frame.pose'): 'params manualRoot larynxHeight tongueShape lipWidth',
 ('M11','config'): 'beam retry_beam',
 ('M14','options'): 'skip_first_row consonant_only_as_zero_initial computation_revision character_column ipa_column note_column start_row table_index encoding delimiter',
 ('M15','validKeys[]'): 'key label',
 ('M16','config'): 'device sample_rate channels roles gain_db',
 ('APP','fonts'): 'version zh latin mono ipa bodySize figure',
 ('APP','fonts.figure'): 'follow zh latin size',
}
# Appearance normalizers own their exact schemas; keep bounded primitive fields.
PRIMITIVE_MAPS = {'curveOverrides','panelWidths','frame.source','buttonAppearance','waveformAppearance','backgroundAppearance'}
NESTED = {key:text.split() for key,text in NESTED.items()}

def validate_values(module, values):
    def walk(value, path='', depth=0):
        if depth > 7: raise ValueError('settings_too_large')
        if isinstance(value, dict):
            allowed = FIELDS[module] if not path else NESTED.get((module,path))
            if allowed is not None and set(value)-set(allowed): raise ValueError('invalid_settings_field')
            if allowed is None and path not in PRIMITIVE_MAPS: raise ValueError('invalid_settings_field')
            for key, item in value.items():
                if not isinstance(key,str) or len(key)>96 or key in ('__proto__','constructor','prototype'): raise ValueError('invalid_settings_field')
                if path in PRIMITIVE_MAPS and not (item is None or isinstance(item,(str,int,float,bool))): raise ValueError('invalid_settings_field')
                walk(item,key if not path else path+'.'+key,depth+1)
        elif isinstance(value,list):
            if len(value)>2048: raise ValueError('settings_too_large')
            for item in value: walk(item,path+'[]',depth+1)
        elif isinstance(value,str):
            if len(value)>8192 or value.startswith(('data:','blob:')): raise ValueError('settings_too_large')
        elif value is not None and not isinstance(value,(bool,int,float)): raise ValueError('invalid_settings_field')
        elif isinstance(value,float) and not math.isfinite(value): raise ValueError('invalid_settings_field')
    walk(values)
    if len(json.dumps(values,ensure_ascii=False).encode('utf-8'))>120000: raise ValueError('settings_too_large')
    return values

def merge_values(base, patch):
    result = dict(base)
    for key,value in patch.items():
        result[key] = merge_values(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else value
    return result

class PreferencePatch(WireModel):
    expected_revision: Annotated[int, Field(ge=0, le=9007199254740991)]
    changes: dict[str, JsonValue]

class PreferenceView(WireModel):
    module: ModuleId
    revision: int
    values: dict[str, JsonValue]
    updated_at: datetime

class PreferencesView(WireModel):
    modules: list[PreferenceView]
    allowed_fields: dict[str,list[str]]

class PreferenceConflict(Exception): pass

def list_preferences(store, owner):
    with store.connection() as c:
        return c.execute('SELECT module_id AS module,revision,settings AS "values",updated_at FROM ptb_accounts.module_preferences WHERE owner_id=%s ORDER BY module_id',(owner,)).fetchall()

def patch_preferences(store, owner, module, expected, changes):
    validate_values(module,changes)
    with store.connection() as c:
        if not c.execute('SELECT id FROM ptb_accounts.users WHERE id=%s AND active FOR UPDATE',(owner,)).fetchone():
            raise HTTPException(401,'authentication_required')
        current=c.execute('SELECT revision,settings AS "values" FROM ptb_accounts.module_preferences WHERE owner_id=%s AND module_id=%s',(owner,module)).fetchone()
        if (current['revision'] if current else 0)!=expected: raise PreferenceConflict()
        values=validate_values(module,merge_values(current['values'] if current else {},changes))
        return c.execute('''INSERT INTO ptb_accounts.module_preferences(owner_id,module_id,revision,settings) VALUES(%s,%s,1,%s)
            ON CONFLICT(owner_id,module_id) DO UPDATE SET revision=module_preferences.revision+1,settings=excluded.settings,updated_at=clock_timestamp()
            RETURNING module_id AS module,revision,settings AS "values",updated_at''',(owner,module,Jsonb(values))).fetchone()

def create_preferences_router(ctx):
    router=APIRouter(prefix='/api/v1/preferences',tags=['accounts'])
    def available():
        if not callable(getattr(ctx.store,'connection',None)): raise HTTPException(503,'settings_service_unavailable')
    @router.get('',response_model=PreferencesView,operation_id='get_account_preferences')
    def listing(session=Depends(ctx.session)):
        available()
        return PreferencesView(modules=list_preferences(ctx.store,session['id']),allowed_fields=FIELDS)
    @router.patch('/{module}',response_model=PreferenceView,operation_id='patch_account_preferences')
    def patch(module:ModuleId,body:PreferencePatch,session=Depends(ctx.mutation)):
        available()
        try:return patch_preferences(ctx.store,session['id'],module,body.expected_revision,body.changes)
        except PreferenceConflict:raise HTTPException(409,'settings_revision_conflict') from None
        except ValueError as error:raise HTTPException(422,str(error)) from None
    return router
