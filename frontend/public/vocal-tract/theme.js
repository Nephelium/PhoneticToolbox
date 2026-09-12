export const dark=()=>document.documentElement.dataset.theme==='dark';
const colors={'#8a9489':'#aabaca','#e5e7dd':'#344858','#39877920':'#65c9bd28','#297c6b':'#75d8c5','#b48774':'#e4b394','#b17869':'#e3a99a','#62918227':'#68bbaf35','#518575':'#80cbbd','#92a08e':'#92b7ad','#7b897b':'#afc3bd','#e3e5da':'#344858','#7b877a':'#aabaca','#39877916':'#7bbfaa16','#39877908':'#7bbfaa08','#adbaaf':'#57736f','#51756b':'#bbd5cf','#226f65':'#80d9cd','#a88973':'#e3bba0'};
Object.assign(colors,{'#fffef9':'#172a36','#e0e4d9':'#344858','#386a5e':'#80cbbd','#839080':'#afc3bd'});
export const tint=color=>dark()?(colors[color]||color):color;
