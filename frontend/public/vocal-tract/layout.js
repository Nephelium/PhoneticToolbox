const tabs=[...document.querySelectorAll('[data-panel]')];
function selectPanel(name){
  for(const tab of tabs){const active=tab.dataset.panel===name;tab.setAttribute('aria-selected',active);tab.tabIndex=active?0:-1;document.getElementById('panel-'+tab.dataset.panel).hidden=!active;}
}
for(const [i,tab] of tabs.entries()){
  tab.onclick=()=>selectPanel(tab.dataset.panel);
  tab.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;selectPanel(tabs[next].dataset.panel);tabs[next].focus();};
}
document.getElementById('showTimeline').onclick=()=>selectPanel('motion');
document.addEventListener('inspector-organ',()=>selectPanel('organs'));
selectPanel('organs');

const layout=document.getElementById('columnLayout');
try{const saved=localStorage.getItem('m10-columns');if(['auto','two','three'].includes(saved))layout.value=saved;}catch{}
function applyLayout(){document.documentElement.dataset.columns=layout.value;try{localStorage.setItem('m10-columns',layout.value);}catch{}window.dispatchEvent(new Event('resize'));}
layout.onchange=applyLayout;applyLayout();
