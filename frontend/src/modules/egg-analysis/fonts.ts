import type {EggTaskConfig,FigureFontSnapshot,ResearchTasks} from '../../platform/research.ts';

export async function preflightExportFonts(config:EggTaskConfig,font:FigureFontSnapshot,check:ResearchTasks['eggFonts']){
  const frozen={...config,font:{...font}};
  if(config.mode!=='single'&&!(config.mode==='batch'&&config.generate_images))return frozen;
  if(!check)throw Error('当前入口尚不能检查导出字体，请重新启动最新工作台。');
  let result;
  try{result=await check(frozen.font);}catch{throw Error('暂时无法检查导出字体，请稍后重试；若持续失败，请检查 EGG 计算环境。');}
  if(!result.available){const roles={zh:'中文',latin:'英文与数字',ipa:'IPA'};
    throw Error('导出环境缺少字体：'+result.fonts.filter(f=>!f.available).map(f=>`${roles[f.role]} ${f.requested}`).join('、')+'。请在字体设置中调整图表字体后重试；IPA 固定使用 Doulos SIL。');}
  return frozen;
}
