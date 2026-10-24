import {ref} from 'vue';
import {projects} from '../platform/browser.ts';
export const automaticAudioGain=ref(true),auditionSettingsError=ref('');
export function loadAuditionSettings(){const saved=projects.read<unknown>('automaticAudioGain',true);automaticAudioGain.value=typeof saved==='boolean'?saved:true;auditionSettingsError.value='';}
export function setAutomaticAudioGain(value:boolean){automaticAudioGain.value=value;auditionSettingsError.value=projects.write('automaticAudioGain',value)?'':'音频设置未能保存，请重试。';}
