import {msg,tr,join,bindText,appendText,bindAttribute,textNode,setHTML,onLocaleChange,parameterLabel,uiError} from './i18n.mjs';
export function articulationParameters(metadata){
 return [{name:'larynx_height',label:msg("modules.vocal-tract-frame.text.laryngeal_height_100"),min:-1,max:1,unit:'cm'},
 {name:'lip_width',label:msg("modules.vocal-tract-frame.text.lip_width_ratio_101"),min:.55,max:1.6,unit:msg("modules.vocal-tract-frame.text.label_102")},
 {name:'f0',label:msg("modules.vocal-tract-frame.text.fundamental_frequency_103"),min:60,max:350,unit:'Hz'},
 {name:'pressure_pa',label:msg("modules.vocal-tract-frame.text.driving_pressure_104"),min:0,max:1600,unit:'Pa'},
 {name:'jitter',label:msg("modules.vocal-tract-frame.text.jitter_period_perturbation_105"),min:0,max:100,unit:msg("modules.vocal-tract-frame.text.strength_106"),hint:msg("modules.vocal-tract-frame.text.source_control_strength_0_100_using_the_sequence_seed_not_a_measured_percent_107")},
 {name:'shimmer',label:msg("modules.vocal-tract-frame.text.shimmer_amplitude_perturbation_108"),min:0,max:100,unit:msg("modules.vocal-tract-frame.text.strength_106"),hint:msg("modules.vocal-tract-frame.text.source_control_strength_0_100_using_the_sequence_seed_not_a_measured_percent_107")},
 ...['upper_lip_x','upper_lip_y','lower_lip_x','lower_lip_y'].map((name,i)=>({name,label:[msg("modules.vocal-tract-frame.text.upper_lip_forward_back_28"),msg("modules.vocal-tract-frame.text.upper_lip_height_29"),msg("modules.vocal-tract-frame.text.lower_lip_forward_back_30"),msg("modules.vocal-tract-frame.text.lower_lip_height_31")][i],min:-.8,max:.8,unit:'cm'})),
 ...metadata.parameters.map(p=>({...p,label:parameterLabel(p),unit:p.name==='VO'?'cm²':''}))];
}
