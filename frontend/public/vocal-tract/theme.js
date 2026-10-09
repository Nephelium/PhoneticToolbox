export const dark=()=>document.documentElement.dataset.theme==='dark';
export const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const colors={'#8a9489':'#aabaca','#e5e7dd':'#344858','#39877920':'#65c9bd28','#297c6b':'#75d8c5','#b48774':'#e4b394','#b17869':'#e3a99a','#62918227':'#68bbaf35','#518575':'#80cbbd','#92a08e':'#92b7ad','#7b897b':'#afc3bd','#e3e5da':'#344858','#7b877a':'#aabaca','#39877916':'#7bbfaa16','#39877908':'#7bbfaa08','#adbaaf':'#57736f','#51756b':'#bbd5cf','#226f65':'#80d9cd','#a88973':'#e3bba0'};
Object.assign(colors,{'#e0e4d9':'#344858','#386a5e':'#80cbbd','#839080':'#afc3bd'});
// Canvas plots and controls use semantic host colours. Tissue rendering in
// scene.js keeps its anatomical palette and does not call this function.
const roles={
 '#fffef9':'--panel','#8a9489':'--muted','#7b897b':'--muted','#7b877a':'--muted','#839080':'--muted','#51756b':'--ink',
 '#e5e7dd':'--line','#e3e5da':'--line','#e0e4d9':'--line','#adbaaf':'--line','#92a08e':'--muted',
 '#297c6b':'--accent','#b17869':'--accent','#518575':'--accent','#386a5e':'--accent','#226f65':'--accent',
 '#b48774':'--muted','#a88973':'--muted',
};
export const tint=color=>{
 if(['#39877920','#62918227','#39877916','#39877908'].includes(color))return (themeColor('--accent')||'#0969da')+color.slice(-2);
 if(roles[color])return themeColor(roles[color])||(dark()?colors[color]:color)||color;
 return dark()?(colors[color]||color):color;
};
