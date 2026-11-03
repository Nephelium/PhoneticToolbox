import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {ref} from 'vue';
import {projects} from '../platform/browser.ts';
export const automaticAudioGain=ref(true),auditionSettingsError=messageRef();
export function loadAuditionSettings(){const saved=projects.read<unknown>('automaticAudioGain',true);automaticAudioGain.value=typeof saved==='boolean'?saved:true;auditionSettingsError.value='';}
export function setAutomaticAudioGain(value:boolean){automaticAudioGain.value=value;auditionSettingsError.value=projects.write('automaticAudioGain',value)?'':uiMessage('state.the_audio_settings_could_not_be_saved_try_again');}
