import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {EggTaskConfig,FigureFontSnapshot,ResearchTasks} from '../../platform/research.ts';

export async function preflightExportFonts(config:EggTaskConfig,font:FigureFontSnapshot,check:ResearchTasks['eggFonts']){
  const frozen={...config,font:{...font}};
  if(config.mode!=='single'&&!(config.mode==='batch'&&config.generate_images))return frozen;
  if(!check)throw uiError(uiMessage("modules.egg-analysis.this_entry_cannot_check_export_fonts_restart_the_latest"));
  let result;
  try{result=await check(frozen.font);}catch{throw uiError(uiMessage("modules.egg-analysis.cannot_check_export_fonts_now_retry_later_if_this"));}
  if(!result.available){const roles=localized(()=>({zh:tr("modules.egg-analysis.chinese"),latin:tr("modules.egg-analysis.english_and_digits"),ipa:'IPA'}));
    throw uiError(joinMessages(joinMessages(uiMessage("modules.egg-analysis.fonts_missing_from_export_environment"),result.fonts.filter(f=>!f.available).map(f=>`${roles[f.role]} ${f.requested}`).join('、')),uiMessage("modules.egg-analysis.adjust_plot_fonts_in_font_settings_and_retry_ipa")));}
  return frozen;
}
