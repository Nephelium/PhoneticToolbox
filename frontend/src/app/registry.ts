export const groups = ['分析与采集', '合成与操控', '标注与实验'] as const;
const entries = [
  ['参数估计','批量提取声学与唇形参数','sliders'], ['参数显示','多轨参数对照与时间选区','chart'],
  ['EGG 信号分析','双声道信号与声门事件','wave'], ['LPC 谱图','联动波形与共振峰频谱','bars'],
  ['唇形提取','视频采集、识别与对齐','lip'], ['语音合成','Klatt 参数化语音合成','speaker'],
  ['发声类型合成','探索发声方式连续统','wave'], ['变速变调','独立调整语速与基频','sliders'],
  ['语谱图转音频','从谱图重建声音','bars'], ['声道工作台','声道建模与动态发音','tract'],
  ['MFA 自动标注','模型驱动的强制对齐','file'], ['语音标注对齐','TextGrid 编辑与唇形对齐','align'],
  ['普通话转 IPA','汉字与国际音标转换','ipa'], ['音系归纳','调查字表与同音字表','nodes'],
  ['感知实验','实验设计、配置与运行','headphones'],
];
export const modules = entries.map(([title, description, icon], i) => ({
  id: `M${String(i + 1).padStart(2,'0')}`, title, description, icon, group: Math.floor(i / 5),
}));
export type Module = typeof modules[number];
