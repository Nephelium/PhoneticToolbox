<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr,locale} from '../i18n/index.ts';
import {ref,computed,watch} from 'vue';
import {preferences,setFonts,prepareFonts,fontError} from '../state/fonts.ts';
import {defaults,normalizeFonts,candidates,japaneseCandidates} from '../design/fonts.ts';
import {desktopFontFamilies} from '../platform/desktop.ts';
import FontFamilySelect from './FontFamilySelect.vue';
const emit=defineEmits<{dirty:[value:boolean]}>();
const draft=ref(normalizeFonts(preferences.value)),message=messageRef(),busy=ref(false),extra=ref<string[]>([]),preview=ref<Record<string,string>>({});
watch(preferences,value=>{draft.value=normalizeFonts(value);});
const fontLabels:Record<string,string>=localized(()=>({SimSun:tr('components.simsun'),KaiTi:tr('components.kaiti'),'Source Han Serif SC':tr('components.source_han_serif'),'Microsoft YaHei':tr('components.microsoft_yahei')}));
const names=computed(()=>[...new Set([...candidates.zh,...japaneseCandidates,...candidates.latin,...candidates.mono,...extra.value])].sort((a,b)=>a.localeCompare(b)));
const resetToken=ref(0);
const monoNames=computed(()=>[...new Set([...candidates.mono,...(draft.value.mono?[draft.value.mono]:[]),...extra.value])]);
async function list(){busy.value=true;message.value='';try{
 if(desktopFontFamilies)extra.value=await desktopFontFamilies();
 else{const api=(window as unknown as {queryLocalFonts?:()=>Promise<{family:string}[]>}).queryLocalFonts;if(!api)throw uiError(uiMessage('components.this_browser_cannot_list_fonts_enter_the_name_of'));extra.value=(await api()).map(f=>f.family);}
 message.value=uiMessage('components.loaded_value_font_names_the_selected_font_will_be',{v0:(extra.value.length)});
}catch(e){message.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.the_font_list_could_not_be_read');}finally{busy.value=false;}}
async function show(){busy.value=true;message.value='';try{const p=await prepareFonts(normalizeFonts(draft.value));preview.value={fontFamily:p.ui,fontSize:p.bodySize+'px','--preview-mono':p.mono,'--preview-figure':p.figure,'--preview-size':p.size+'px'};message.value=uiMessage('components.preview_updated_click_apply_fonts_to_use_it');}catch(e){message.value=uiErrorMessage((e as Error));}finally{busy.value=false;}}
async function apply(){if(busy.value)return false;busy.value=true;message.value='';try{if(await setFonts(draft.value)){draft.value=normalizeFonts(preferences.value);resetToken.value++;preview.value={};message.value=uiMessage('components.fonts_were_applied_to_the_workspace_and_figures_and');return true;}return false;}catch(e){message.value=uiErrorMessage((e as Error));return false;}finally{busy.value=false;}}
function cancel(){draft.value=normalizeFonts(preferences.value);resetToken.value++;preview.value={};message.value=uiMessage('components.font_editing_cancelled');}
watch(()=>busy.value||JSON.stringify(normalizeFonts(draft.value))!==JSON.stringify(normalizeFonts(preferences.value)),value=>emit('dirty',value),{immediate:true});
defineExpose({save:apply});
</script>
<template><section class="font-settings settings-card" :aria-label="tr('components.font_settings')">
<div class="section-title settings-heading"><h3>{{tr('components.fonts')}}</h3><button @click="list" :disabled="busy">{{tr('components.read_installed_font_list')}}</button></div>
<p class="settings-note">{{tr('components.the_defaults_are_simsun_and_times_new_roman_choose')}}</p>
<p v-if="locale==='ja'" class="settings-note">{{tr('components.japanese_font_default')}}</p>
<div class="settings-columns"><div class="settings-column">
<div class="settings-section"><h4>{{tr('components.interface_fonts')}}</h4><div class="font-fields">
<FontFamilySelect v-model="draft.zh" :label="tr('components.chinese_font')" :names="names" :labels="fontLabels" :reset-token="resetToken"/>
<FontFamilySelect v-model="draft.latin" :label="tr('components.english_and_numeric_font')" :names="names" :labels="fontLabels" :reset-token="resetToken"/>
<FontFamilySelect id="mono-choice" v-model="draft.mono" :label="tr('components.code_and_monospace_font')" :custom-label="tr('components.custom_code_font')" :names="monoNames" :labels="{'JetBrains Mono':tr('components.built_in')}" :reset-token="resetToken"/>
<label class="fixed-font"><span>{{tr('components.all_ipa_symbols')}}</span><input :value="tr('components.doulos_sil_fixed')" :aria-label="tr('components.ipa_font')" readonly/></label>
</div><p class="settings-note">{{tr('components.jetbrains_mono_and_the_ipa_font_are_bundled_no')}}</p></div>
<label class="setting-row body-size-row">{{tr('components.base_body_font_size_px')}}<input v-model.number="draft.bodySize" type="number" min="10" max="24" step="1" :aria-label="tr('components.base_body_font_size')"/></label>
<div class="settings-section"><div class="font-chart-heading"><h4>{{tr('components.figures_and_exports')}}</h4><label class="font-follow"><input v-model="draft.figure.follow" type="checkbox" :aria-label="tr('components.use_global_fonts_for_figures_and_exports')"/>{{tr('components.use_global_fonts')}}</label></div>
<div v-if="!draft.figure.follow" class="font-fields"><FontFamilySelect v-model="draft.figure.zh" :label="tr('components.chinese_font_for_figures')" :names="names" :labels="fontLabels" :reset-token="resetToken"/><FontFamilySelect v-model="draft.figure.latin" :label="tr('components.english_font_for_figures')" :names="names" :labels="fontLabels" :reset-token="resetToken"/></div>
<label class="setting-row">{{tr('components.base_figure_font_size_px')}}<input v-model.number="draft.figure.size" type="number" min="10" max="24" step="1" :aria-label="tr('components.base_figure_font_size')"/></label>
<p class="settings-note">{{tr('components.exports_use_the_selected_font_where_available_otherwise_a')}}</p></div>
</div><div class="settings-column">
<div class="settings-section font-preview-section"><h4>{{tr('components.font_preview')}}</h4>
<div class="font-preview" :style="preview"><span>{{tr('components.acoustic_parameters_chinese_font_preview')}}</span><span>PhoneticToolbox 0123456789 −12.5 Hz</span><span class="ipa-text">[aː tʰ ɕ ŋ ə ã n̩ ɑ²]</span><code class="font-code-preview" :style="{fontFamily:preview['--preview-mono']}">f0 = signal[0:10]  # Hz != ms</code><svg viewBox="0 0 420 72" role="img" :aria-label="tr('components.figure_font_preview')" :style="{fontFamily:preview['--preview-figure']||'var(--font-figure)',fontSize:preview['--preview-size']||'var(--figure-size)'}"><path d="M40 5V42H405" fill="none" stroke="currentColor"/><text x="48" y="24" fill="currentColor">{{tr('components.fundamental_frequency_f0_12_5_hz')}}</text><text x="240" y="64" fill="currentColor">{{tr('components.time_s')}}</text></svg></div>
<p class="settings-note">{{tr('components.preview_then_click_apply_fonts_the_settings_will_be')}}</p></div>
</div></div>
<p v-if="message||fontError" role="status">{{message||fontError}}</p>
<div class="font-actions"><button @click="draft=defaults();resetToken++;message=tr('components.default_settings_filled_in_click_apply_fonts_to_use')" :disabled="busy">{{tr('components.restore_defaults')}}</button><button @click="cancel" :disabled="busy">{{tr('components.cancel_font_editing')}}</button><button @click="show" :disabled="busy">{{tr('components.preview_fonts')}}</button><button class="primary" @click="apply" :disabled="busy">{{busy?tr('components.checking_fonts'):tr('components.apply_fonts')}}</button></div>
</section></template>
<style scoped>
.font-settings{display:flex;flex-direction:column;gap:14px;align-self:start}.font-settings .section-title{margin:0}.font-settings .section-title button{font-size:0.857143rem;min-height:28px;padding:4px 8px}.font-fields{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}.fixed-font{display:flex;flex-direction:column;gap:7px;min-width:0}.fixed-font input{width:100%;height:36px;background:var(--app);color:var(--muted)}.font-chart-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.font-follow{display:flex;align-items:center;gap:7px;font-size:0.857143rem}.font-preview{display:grid;grid-template-columns:1fr;gap:6px;padding:12px 14px;background:var(--app);border:1px solid var(--border);border-radius:7px;line-height:1.65;overflow-wrap:anywhere}.font-preview svg{width:100%;height:65px;max-height:65px}.font-code-preview{font-family:var(--preview-mono,var(--font-mono));font-size:0.857143rem;font-variant-ligatures:none}.font-actions{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;border-top:1px solid var(--border);padding-top:14px}.font-actions button{font-size:0.857143rem;padding:5px 9px}.font-settings .setting-row{display:flex;align-items:center;justify-content:space-between;margin:0;gap:12px}.font-settings .setting-row input{width:80px}.font-settings [role=status]{font-size:0.857143rem;color:var(--accent)}@container module (max-width:760px){.font-fields{grid-template-columns:1fr}}
</style>
