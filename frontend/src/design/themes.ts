// UI adaptations, not Codex application code. Provenance: docs/references/p19-appearance-sources.md.
export type ColorMode='light'|'dark';
export type ThemeMode=ColorMode|'system';
type Seed=readonly [background:string,foreground:string,accent:string];
export interface Palette {id:string;name:string;light:Seed;dark:Seed}
export const DEFAULT_PALETTE='codex';
export const palettes:Palette[]=[
 {id:'absolutely',name:'Absolutely',light:['#faf9f6','#34322f','#a94f31'],dark:['#202020','#e6e2dc','#e59b77']},
 {id:'ayu',name:'Ayu',light:['#fafafa','#575f66','#a76500'],dark:['#0b0e14','#bfbdb6','#e6b450']},
 {id:'catppuccin',name:'Catppuccin',light:['#eff1f5','#4c4f69','#8839ef'],dark:['#1e1e2e','#cdd6f4','#cba6f7']},
 {id:'codex',name:'Codex',light:['#f9f9f9','#202020','#0969da'],dark:['#171717','#ededed','#79aaff']},
 {id:'dracula',name:'Dracula',light:['#f7f5fb','#39354a','#8751a8'],dark:['#282a36','#f8f8f2','#ff79c6']},
 {id:'everforest',name:'Everforest',light:['#fdf6e3','#5c6a72','#657b35'],dark:['#2d353b','#d3c6aa','#a7c080']},
 {id:'github',name:'GitHub',light:['#f6f8fa','#24292f','#0969da'],dark:['#0d1117','#c9d1d9','#58a6ff']},
 {id:'gruvbox',name:'Gruvbox',light:['#fbf1c7','#3c3836','#427b58'],dark:['#282828','#ebdbb2','#8ec07c']},
 {id:'linear',name:'Linear',light:['#f8f9fc','#292a35','#5b56bd'],dark:['#17181c','#e6e6ed','#a6a0ff']},
 {id:'lobster',name:'Lobster',light:['#fff5f1','#472c29','#b52f28'],dark:['#141824','#f4ded7','#ff6f61']},
 {id:'material',name:'Material',light:['#fafafa','#455a64','#007d86'],dark:['#263238','#eeffff','#80cbc4']},
 {id:'matrix',name:'Matrix',light:['#f0f7ef','#233c27','#236e36'],dark:['#0d150f','#c0dfbd','#6cdb7b']},
 {id:'monokai',name:'Monokai',light:['#faf8ee','#49483e','#9b3864'],dark:['#272822','#f8f8f2','#a6e22e']},
 {id:'night-owl',name:'Night Owl',light:['#f7fafb','#403f53','#356c9c'],dark:['#011627','#d6deeb','#82aaff']},
 {id:'nord',name:'Nord',light:['#eceff4','#2e3440','#466680'],dark:['#2e3440','#eceff4','#88c0d0']},
 {id:'notion',name:'Notion',light:['#f7f7f5','#37352f','#8b5e3c'],dark:['#191919','#d4d4d4','#d8ad78']},
 {id:'og',name:'OG',light:['#f4f9f7','#243a34','#15765f'],dark:['#17211e','#e2eee7','#72c7a8']},
 {id:'oscurange',name:'Oscurange',light:['#fff5e9','#4c352b','#a84d15'],dark:['#211a17','#efdccc','#eea064']},
 {id:'one',name:'One',light:['#fafafa','#383a42','#4078b5'],dark:['#282c34','#abb2bf','#61afef']},
 {id:'proof',name:'Proof',light:['#f7f5ee','#39382f','#75642b'],dark:['#22221c','#e2dfce','#c5b675']},
 {id:'raycast',name:'Raycast',light:['#faf8f8','#302c2d','#ba3945'],dark:['#19191b','#e9e7e8','#ff7480']},
 {id:'rose-pine',name:'Rose Pine',light:['#faf4ed','#575279','#b4637a'],dark:['#191724','#e0def4','#ebbcba']},
 {id:'sentry',name:'Sentry',light:['#faf5fb','#463451','#7950a0'],dark:['#241c2f','#eee2f0','#cda7e6']},
 {id:'solarized',name:'Solarized',light:['#fdf6e3','#586e75','#006c91'],dark:['#002b36','#93a1a1','#60b3d1']},
 {id:'temple',name:'Temple',light:['#f8f4e9','#454739','#617649'],dark:['#22271e','#dbe2ce','#b5c48e']},
 {id:'tokyo-night',name:'Tokyo Night',light:['#e1e2e7','#3760bf','#5a4a78'],dark:['#1a1b26','#c0caf5','#7aa2f7']},
 {id:'vercel',name:'Vercel',light:['#fafafa','#171717','#404040'],dark:['#0a0a0a','#ededed','#bdbdbd']},
 {id:'vscode-plus',name:'VS Code Plus',light:['#f3f3f3','#333333','#005a9e'],dark:['#1e1e1e','#d4d4d4','#75beff']},
 {id:'xcode',name:'Xcode',light:['#f5f5f5','#292a30','#9b2393'],dark:['#292a30','#dfdfe0','#ff8ab0']},
];
export const normalizePalette=(value:unknown)=>palettes.some(p=>p.id===value)?value as string:DEFAULT_PALETTE;
export const normalizeMode=(value:unknown):ThemeMode=>value==='dark'||value==='light'?value:'system';
const rgb=(hex:string)=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
export function mix(a:string,b:string,amount:number){return '#'+rgb(a).map((x,i)=>Math.round(x*(1-amount)+rgb(b)[i]*amount).toString(16).padStart(2,'0')).join('');}
const luminance=(hex:string)=>rgb(hex).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}).reduce((s,x,i)=>s+x*[.2126,.7152,.0722][i],0);
export function contrast(a:string,b:string){const x=luminance(a),y=luminance(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);}
function readable(color:string,backgrounds:string[],minimum=4.5){
 const target=contrast('#000000',backgrounds[0])>contrast('#ffffff',backgrounds[0])?'#000000':'#ffffff';
 for(let step=0;step<=100;step++){const next=mix(color,target,step/100);if(backgrounds.every(bg=>contrast(next,bg)>=minimum))return next;}
 return target;
}
export function paletteTokens(id:string,mode:ColorMode):Record<string,string>{
 const seed=palettes.find(p=>p.id===normalizePalette(id))![mode],isDark=mode==='dark';
 const [app,ink,hue]=seed,panel=mix(app,'#ffffff',isDark?.035:.55),sidebar=mix(app,ink,isDark?.025:.035);
 const selected=mix(panel,hue,isDark?.16:.10),surfaces=[app,panel,sidebar,selected];
 const text=readable(ink,surfaces,7),muted=readable(mix(ink,app,.30),surfaces),accent=readable(hue,surfaces);
 return {'--app':app,'--panel':panel,'--sidebar':sidebar,'--text':text,'--muted':muted,'--border':mix(panel,ink,.20),'--accent':accent,'--selected':selected,
 '--on-accent':contrast('#ffffff',accent)>contrast('#111111',accent)?'#ffffff':'#111111',
 '--success':readable(isDark?'#86c99a':'#267544',surfaces),'--warning':readable(isDark?'#e9c17e':'#8a5a13',surfaces),'--danger':readable(isDark?'#f69ba1':'#b33448',surfaces),
 '--teal':readable(isDark?'#80c6c2':'#287d79',surfaces),'--teal-bg':mix(panel,'#398779',.10),'--violet':readable(isDark?'#c6b4ec':'#7557a1',surfaces),'--violet-bg':mix(panel,'#9975bc',.09),
 '--selection':`${accent}${isDark?'33':'22'}`};
}
