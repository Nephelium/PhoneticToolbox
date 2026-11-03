<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {computed} from 'vue';
import PalettePicker from './PalettePicker.vue';
import CustomPaletteSettings from './CustomPaletteSettings.vue';
import BackgroundSettings from './BackgroundSettings.vue';
import ThemeModeCards from './ThemeModeCards.vue';
import {paletteRevision} from '../state/appearance.ts';
import {palettes,normalizePalette,type ThemeMode} from '../design/themes.ts';
import {pageScale,setPageScale} from '../state/pageZoom.ts';
import {buttonAppearance,setButtonAppearance} from '../state/buttons.ts';
import type {ButtonMode,ButtonMaterial} from '../design/buttons.ts';
import {waveformAppearance,setWaveformAppearance} from '../state/waveform-color.ts';
import {normalizeHex,type WaveformColorMode} from '../design/waveform-color.ts';
import {ref} from 'vue';
const buttonError=messageRef();
function setButtons(value:Parameters<typeof setButtonAppearance>[0]){buttonError.value=setButtonAppearance(value)?'':uiMessage('components.the_button_appearance_was_applied_but_local_settings_could');}
const waveDraft=ref(waveformAppearance.value.custom),waveError=messageRef();
function setWave(value:Parameters<typeof setWaveformAppearance>[0]){waveError.value=setWaveformAppearance(value)?'':uiMessage('components.the_waveform_color_was_applied_but_local_settings_could');waveDraft.value=waveformAppearance.value.custom;}
function customWave(value:string){const color=normalizeHex(value);if(!color){waveError.value=uiMessage('components.enter_a_valid_hex_color_such_as_2463eb');return;}setWave({custom:color});}
const props=defineProps<{mode:ThemeMode;palette:string}>();
const emit=defineEmits<{'update:mode':[value:ThemeMode];'update:palette':[value:string]}>();
const current=computed(()=>{paletteRevision.value;return palettes.find(p=>p.id===normalizePalette(props.palette))!;});
</script>
<template><section class="appearance-settings settings-card" :aria-label="tr('components.appearance_settings')">
 <div class="settings-heading"><h3>{{tr('components.appearance')}}</h3><span class="muted">{{tr('components.applied_immediately')}}</span></div>
 <div class="settings-columns"><div class="settings-column">
 <div class="setting-block"><label for="palette-choice">{{tr('components.color_scheme')}}</label><PalettePicker :model-value="palette" :mode="mode" @update:model-value="emit('update:palette',$event)"/></div>
 <CustomPaletteSettings @select="emit('update:palette','custom')"/>
 <details open class="palette-details"><summary>{{tr('components.about_these_color_schemes')}}</summary><p class="settings-note">{{tr('components.these_palettes_are_adapted_from_the_codex_themes_of')}}</p></details>
 <p class="settings-note">{{tr('components.open_the_list_then_hover_or_use_the_arrow')}}</p>
 <ThemeModeCards :mode="mode" :palette="current" @select="emit('update:mode',$event)"/>
 <p class="settings-note">{{tr('components.light_and_dark_variants_use_the_same_scheme_follow')}}</p>
 <div class="appearance-divider"/>
 <BackgroundSettings/>
 </div><div class="settings-column">
 <div class="setting-block"><span>{{tr('components.page_zoom')}}</span><div class="page-zoom-controls"><button :aria-label="tr('components.zoom_out_page')" :disabled="pageScale<=70" @click="setPageScale(pageScale-10)">−</button><output :aria-label="tr('components.page_zoom_level')">{{pageScale}}%</output><button :aria-label="tr('components.zoom_in_page')" :disabled="pageScale>=150" @click="setPageScale(pageScale+10)">+</button><button @click="setPageScale(100)">{{tr('components.reset_to_100')}}</button></div></div>
 <p class="settings-note">{{tr('components.ctrl_wheel_zooms_within_figures_adjust_the_page_size')}}</p>
 <div class="appearance-divider"/>
 <div class="setting-block"><label for="button-style-choice">{{tr('components.button_highlighting')}}</label><select id="button-style-choice" :value="buttonAppearance.mode" @change="setButtons({mode:($event.target as HTMLSelectElement).value as ButtonMode})"><option value="auto">{{tr('components.highlight_by_importance_recommended')}}</option><option value="all">{{tr('components.highlight_all')}}</option><option value="plain">{{tr('components.use_plain_style_for_all')}}</option></select></div>
 <div class="setting-block"><label for="button-material-choice">{{tr('components.button_material')}}</label><select id="button-material-choice" :value="buttonAppearance.material" @change="setButtons({material:($event.target as HTMLSelectElement).value as ButtonMaterial})"><option value="solid">{{tr('components.plain')}}</option><option value="glass">{{tr('components.frosted_glass')}}</option></select></div>
 <p class="settings-note">{{tr('components.plain_buttons_retain_their_background_color_frosted_glass_buttons')}}</p>
 <div v-if="buttonAppearance.material==='glass'" class="glass-controls">
  <label for="button-glass-opacity">{{tr('components.glass_opacity')}} <output>{{buttonAppearance.glassOpacity}}%</output></label><input id="button-glass-opacity" type="range" min="0" max="100" :value="buttonAppearance.glassOpacity" @input="setButtons({glassOpacity:Number(($event.target as HTMLInputElement).value)})"/>
  <label for="button-glass-blur">{{tr('components.glass_blur')}} <output>{{buttonAppearance.glassBlur}} px</output></label><input id="button-glass-blur" type="range" min="0" max="40" :value="buttonAppearance.glassBlur" @input="setButtons({glassBlur:Number(($event.target as HTMLInputElement).value)})"/>
  <p class="settings-note">{{tr('components.lower_opacity_makes_the_button_background_lighter_and_the')}}</p>
 </div>
 <label class="button-effects-toggle"><input type="checkbox" :checked="buttonAppearance.effects" @change="setButtons({effects:($event.target as HTMLInputElement).checked})"/>{{tr('components.button_shadows_and_hover_glow')}}</label>
 <p class="settings-note">{{tr('components.important_actions_are_highlighted_by_default_shadows_and_glow')}}</p>
 <p v-if="buttonError" class="settings-note" role="status">{{buttonError}}</p>
 <div class="appearance-divider"/>
 <div class="setting-block"><label for="waveform-color-choice">{{tr('components.waveform_line_color')}}</label><select id="waveform-color-choice" :value="waveformAppearance.mode" @change="setWave({mode:($event.target as HTMLSelectElement).value as WaveformColorMode})"><option value="theme">{{tr('components.follow_theme_color_default')}}</option><option value="blue">{{tr('components.blue')}}</option><option value="custom">{{tr('components.custom_color')}}</option></select></div>
 <div v-if="waveformAppearance.mode==='custom'" class="wave-color-controls"><label>{{tr('components.color_picker')}}<input type="color" :value="waveformAppearance.custom" :aria-label="tr('components.waveform_color_picker')" @input="customWave(($event.target as HTMLInputElement).value)"/></label><label>{{tr('components.hex_color')}}<input v-model="waveDraft" :aria-label="tr('components.waveform_hex_color')" maxlength="7" spellcheck="false" placeholder="#2463eb" @change="customWave(waveDraft)"/></label></div>
 <svg class="wave-color-preview" viewBox="0 0 220 32" role="img" :aria-label="tr('components.waveform_color_preview')"><path d="M0 16H12L18 10L24 22L30 4L36 28L42 8L48 24L54 13L60 19L66 16H88L94 7L100 25L106 2L112 30L118 5L124 27L130 12L136 20L142 16H166L172 8L178 24L184 4L190 28L196 12L202 20L208 16H220"/></svg>
 <p class="settings-note">{{tr('components.the_theme_color_follows_the_palette_and_light_dark')}}</p>
 <p v-if="waveError" class="settings-note" role="status">{{waveError}}</p>
 </div></div>
</section></template>
<style scoped>
.glass-controls{display:grid;gap:8px}.glass-controls label{display:flex;justify-content:space-between;gap:8px;font-size:var(--control-size)}.glass-controls input{width:100%}
.button-effects-toggle{display:flex;align-items:center;gap:8px;font-size:var(--control-size)}
.wave-color-controls{display:grid;grid-template-columns:64px minmax(0,1fr);gap:12px}.wave-color-controls label{display:grid;gap:6px;min-width:0;font-size:var(--control-size)}.wave-color-controls input{width:100%;min-width:0}.wave-color-controls input[type=color]{padding:3px;height:34px}.wave-color-preview{width:100%;height:40px;padding:4px 8px;background:var(--app);border:1px solid var(--border);border-radius:6px}.wave-color-preview path{fill:none;stroke:var(--waveform-color,var(--wave));stroke-width:1.5}
.appearance-settings{display:flex;flex-direction:column;gap:14px;align-self:start}.setting-block{display:grid;gap:8px}.setting-block>label,.setting-block>span{font-weight:550}.setting-block select{width:100%}.appearance-divider{border-top:1px solid var(--border)}.page-zoom-controls{gap:6px;flex-wrap:wrap}.page-zoom-controls button{padding:5px 10px}.palette-details{font-size:.857143rem;color:var(--muted)}.palette-details summary{cursor:pointer}.palette-details p{margin-top:8px}
</style>
