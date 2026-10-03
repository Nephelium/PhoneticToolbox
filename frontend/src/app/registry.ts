export const groups = ['分析与采集', '合成与操控', '标注与实验'] as const;
export interface Module {id:string;title:string;description:string;icon:string;group:0|1|2}
// Persisted workspace keys use these IDs. Display order must never assign identity.
export const modules:Module[] = [
  {id:'M01',title:'参数估计',description:'批量提取声学与唇形参数',icon:'sliders',group:0},
  {id:'M02',title:'参数显示',description:'多轨参数对照与时间选区',icon:'chart',group:0},
  {id:'M03',title:'EGG 信号分析',description:'双声道信号与声门事件',icon:'wave',group:0},
  {id:'M04',title:'LPC 谱图',description:'联动波形与共振峰频谱',icon:'bars',group:0},
  {id:'M05',title:'唇形提取',description:'视频采集、识别与对齐',icon:'lip',group:0},
  {id:'M06',title:'语音合成',description:'Klatt 参数化语音合成',icon:'speaker',group:1},
  {id:'M07',title:'发声类型合成',description:'探索发声方式连续统',icon:'wave',group:1},
  {id:'M08',title:'变速变调',description:'独立调整语速与基频',icon:'sliders',group:1},
  {id:'M09',title:'语谱图转音频',description:'从谱图重建声音',icon:'bars',group:1},
  {id:'M10',title:'声道工作台',description:'声道建模与动态发音',icon:'tract',group:1},
  {id:'M11',title:'MFA 自动标注',description:'模型驱动的强制对齐',icon:'file',group:2},
  {id:'M12',title:'语音标注对齐',description:'TextGrid 编辑与唇形对齐',icon:'align',group:2},
  {id:'M13',title:'普通话转 IPA',description:'汉字与国际音标转换',icon:'ipa',group:2},
  {id:'M14',title:'音系归纳',description:'调查字表与同音字表',icon:'nodes',group:2},
  {id:'M15',title:'感知实验',description:'实验设计、配置与运行',icon:'headphones',group:2},
  {id:'M16',title:'录音',description:'本机采集、任务录音与可恢复剪辑',icon:'wave',group:0},
  {id:'M17',title:'国际音标 Plus',description:'IPA、extIPA 与 VoQS 全表输入',icon:'ipa-keyboard',group:2},
];
