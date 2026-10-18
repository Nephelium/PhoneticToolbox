export function articulationParameters(metadata){
 return [{name:'larynx_height',label:'喉位高低',min:-1,max:1,unit:'cm'},
 {name:'lip_width',label:'唇宽比例',min:.55,max:1.6,unit:'倍'},
 {name:'f0',label:'基频',min:60,max:350,unit:'Hz'},
 {name:'pressure_pa',label:'驱动气压',min:0,max:1600,unit:'Pa'},
 {name:'jitter',label:'Jitter（周期抖动）',min:0,max:100,unit:'强度',hint:'0–100 声源调节强度，沿用序列随机种子，不等于测量百分比。'},
 {name:'shimmer',label:'Shimmer（振幅抖动）',min:0,max:100,unit:'强度',hint:'0–100 声源调节强度，沿用序列随机种子，不等于测量百分比。'},
 ...['upper_lip_x','upper_lip_y','lower_lip_x','lower_lip_y'].map((name,i)=>({name,label:['上唇前后','上唇高低','下唇前后','下唇高低'][i],min:-.8,max:.8,unit:'cm'})),
 ...metadata.parameters.map(p=>({...p,unit:p.name==='VO'?'cm²':''}))];
}
