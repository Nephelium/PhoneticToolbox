<script setup lang="ts">
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
const buttonError=ref('');
function setButtons(value:Parameters<typeof setButtonAppearance>[0]){buttonError.value=setButtonAppearance(value)?'':'按钮外观已生效，但本机设置未能保存。';}
const waveDraft=ref(waveformAppearance.value.custom),waveError=ref('');
function setWave(value:Parameters<typeof setWaveformAppearance>[0]){waveError.value=setWaveformAppearance(value)?'':'波形颜色已生效，但本机设置未能保存。';waveDraft.value=waveformAppearance.value.custom;}
function customWave(value:string){const color=normalizeHex(value);if(!color){waveError.value='请输入有效的 HEX 颜色，如 #2463eb。';return;}setWave({custom:color});}
const props=defineProps<{mode:ThemeMode;palette:string}>();
const emit=defineEmits<{'update:mode':[value:ThemeMode];'update:palette':[value:string]}>();
const current=computed(()=>{paletteRevision.value;return palettes.find(p=>p.id===normalizePalette(props.palette))!;});
</script>
<template><section class="appearance-settings settings-card" aria-label="外观设置">
 <div class="settings-heading"><h3>外观</h3><span class="muted">即时生效</span></div>
 <div class="settings-columns"><div class="settings-column">
 <div class="setting-block"><label for="palette-choice">配色方案</label><PalettePicker :model-value="palette" :mode="mode" @update:model-value="emit('update:palette',$event)"/></div>
 <CustomPaletteSettings @select="emit('update:palette','custom')"/>
 <details open class="palette-details"><summary>关于配色</summary><p class="settings-note">这些配色参考 Codex 同名主题，由 PhoneticToolbox 适配并补齐深浅版本。各主题的来源与许可见关于页。科研曲线沿用各轨道的颜色含义。</p></details>
 <p class="settings-note">默认使用 Everforest。展开列表后，悬停或用方向键浏览配色，查看文字、按钮与波形的小窗预览。</p>
 <ThemeModeCards :mode="mode" :palette="current" @select="emit('update:mode',$event)"/>
 <p class="settings-note">同一方案的深浅配色同步切换。跟随系统会自动响应系统外观。</p>
 <div class="appearance-divider"/>
 <BackgroundSettings/>
 </div><div class="settings-column">
 <div class="setting-block"><span>页面缩放</span><div class="page-zoom-controls"><button aria-label="缩小页面" :disabled="pageScale<=70" @click="setPageScale(pageScale-10)">−</button><output aria-label="页面缩放比例">{{pageScale}}%</output><button aria-label="放大页面" :disabled="pageScale>=150" @click="setPageScale(pageScale+10)">+</button><button @click="setPageScale(100)">恢复 100%</button></div></div>
 <p class="settings-note">Ctrl＋滚轮用于图内缩放，页面大小在这里调整。</p>
 <div class="appearance-divider"/>
 <div class="setting-block"><label for="button-style-choice">按钮高亮</label><select id="button-style-choice" :value="buttonAppearance.mode" @change="setButtons({mode:($event.target as HTMLSelectElement).value as ButtonMode})"><option value="auto">按重要性高亮（推荐）</option><option value="all">全部高亮</option><option value="plain">全部普通</option></select></div>
 <div class="setting-block"><label for="button-material-choice">按钮材质</label><select id="button-material-choice" :value="buttonAppearance.material" @change="setButtons({material:($event.target as HTMLSelectElement).value as ButtonMaterial})"><option value="solid">普通</option><option value="glass">磨砂玻璃</option></select></div>
 <p class="settings-note">普通按钮保留底色，磨砂玻璃按钮透出背景并模糊按钮后的画面。</p>
 <div v-if="buttonAppearance.material==='glass'" class="glass-controls">
  <label for="button-glass-opacity">玻璃不透明度 <output>{{buttonAppearance.glassOpacity}}%</output></label><input id="button-glass-opacity" type="range" min="0" max="100" :value="buttonAppearance.glassOpacity" @input="setButtons({glassOpacity:Number(($event.target as HTMLInputElement).value)})"/>
  <label for="button-glass-blur">玻璃模糊强度 <output>{{buttonAppearance.glassBlur}} px</output></label><input id="button-glass-blur" type="range" min="0" max="40" :value="buttonAppearance.glassBlur" @input="setButtons({glassBlur:Number(($event.target as HTMLInputElement).value)})"/>
  <p class="settings-note">不透明度越低，按钮底色越淡、背景越明显。模糊强度只改变按钮后的画面清晰度。</p>
 </div>
 <label class="button-effects-toggle"><input type="checkbox" :checked="buttonAppearance.effects" @change="setButtons({effects:($event.target as HTMLInputElement).checked})"/>按钮阴影与悬停光效</label>
 <p class="settings-note">重要操作默认高亮。阴影与光效可独立关闭，选择状态和键盘焦点仍会保留。</p>
 <p v-if="buttonError" class="settings-note" role="status">{{buttonError}}</p>
 <div class="appearance-divider"/>
 <div class="setting-block"><label for="waveform-color-choice">波形线颜色</label><select id="waveform-color-choice" :value="waveformAppearance.mode" @change="setWave({mode:($event.target as HTMLSelectElement).value as WaveformColorMode})"><option value="theme">跟随主题色（默认）</option><option value="blue">蓝色</option><option value="custom">自定义颜色</option></select></div>
 <div v-if="waveformAppearance.mode==='custom'" class="wave-color-controls"><label>色盘<input type="color" :value="waveformAppearance.custom" aria-label="波形线颜色色盘" @input="customWave(($event.target as HTMLInputElement).value)"/></label><label>HEX 颜色<input v-model="waveDraft" aria-label="波形线 HEX 颜色" maxlength="7" spellcheck="false" placeholder="#2463eb" @change="customWave(waveDraft)"/></label></div>
 <svg class="wave-color-preview" viewBox="0 0 220 32" role="img" aria-label="波形线颜色预览"><path d="M0 16H12L18 10L24 22L30 4L36 28L42 8L48 24L54 13L60 19L66 16H88L94 7L100 25L106 2L112 30L118 5L124 27L130 12L136 20L142 16H166L172 8L178 24L184 4L190 28L196 12L202 20L208 16H220"/></svg>
 <p class="settings-note">主题色随配色与深浅模式变化，自定义颜色保持所选值。音频波形使用此颜色，其他参数曲线保留各自配色。</p>
 <p v-if="waveError" class="settings-note" role="status">{{waveError}}</p>
 </div></div>
</section></template>
<style scoped>
.glass-controls{display:grid;gap:8px}.glass-controls label{display:flex;justify-content:space-between;gap:8px;font-size:var(--control-size)}.glass-controls input{width:100%}
.button-effects-toggle{display:flex;align-items:center;gap:8px;font-size:var(--control-size)}
.wave-color-controls{display:grid;grid-template-columns:64px minmax(0,1fr);gap:12px}.wave-color-controls label{display:grid;gap:6px;min-width:0;font-size:var(--control-size)}.wave-color-controls input{width:100%;min-width:0}.wave-color-controls input[type=color]{padding:3px;height:34px}.wave-color-preview{width:100%;height:40px;padding:4px 8px;background:var(--app);border:1px solid var(--border);border-radius:6px}.wave-color-preview path{fill:none;stroke:var(--waveform-color,var(--wave));stroke-width:1.5}
.appearance-settings{display:flex;flex-direction:column;gap:14px;align-self:start}.setting-block{display:grid;gap:8px}.setting-block>label,.setting-block>span{font-weight:550}.setting-block select{width:100%}.appearance-divider{border-top:1px solid var(--border)}.page-zoom-controls{gap:6px;flex-wrap:wrap}.page-zoom-controls button{padding:5px 10px}.palette-details{font-size:.857143rem;color:var(--muted)}.palette-details summary{cursor:pointer}.palette-details p{margin-top:8px}
</style>
