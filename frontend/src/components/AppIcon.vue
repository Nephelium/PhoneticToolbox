<script setup lang="ts">
defineProps<{name?:string}>();
const paths:Record<string,string>={
 home:'M3 10 12 3l9 7M5 9v12h5v-7h4v7h5V9', search:'M16 16l5 5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
 sliders:'M3 6h7m4 0h7M3 12h12m4 0h2M3 18h3m4 0h11M10 3v6m5 0v6M6 15v6',
 chart:'M3 3v18h18M6 14l4-5 4 7 6-11',wave:'M2 12h4l3-8 5 16 3-8h5',bars:'M3 13v5M7 7v11M12 3v15M17 7v11M21 13v5',
 microphone:'M9 5a3 3 0 0 1 6 0v7a3 3 0 0 1-6 0ZM5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8',
 glottis:'M6 3C3 6 3 17 6 21c1-4 4-6 4-9S7 7 6 3ZM18 3c3 3 3 14 0 18-1-4-4-6-4-9s3-5 4-9Z',
 'time-pitch':'M3 8a6 6 0 1 1 12 0 6 6 0 0 1-12 0ZM9 5v3l2 1M3 21h4l4-4 4 2 6-7M17 12h4v4',
 'spectrum-audio':'M3 3h12v11H3ZM7 3v11M11 3v11M3 7h12M3 10h12M3 19h7l3-3v6l-3-3M17 17q3 2 0 4M20 15q5 4 0 8',
 'forced-alignment':'M3 4h11M3 7h7M3 17h18v5H3ZM8 17v5M16 17v5M6 10v4m-2-2 2 2 2-2M18 3v10m-2-2 2 2 2-2',
 textgrid:'M3 3h18v18H3ZM3 9h18M3 15h18M9 3v6M16 3v6M6 9v6M14 9v6M11 15v6M18 15v6',
 lip:'M2 12q10-11 20 0-10 11-20 0Zm0 0h20',speaker:'M3 9h4l6-5v16l-6-5H3ZM17 8q5 4 0 8M20 5q7 7 0 14',
 tract:'M4 3v16q0 2 2 2h15M8 6v8q0 5 5 1t7 0',file:'M5 2h10l4 4v16H5ZM14 2v6h5M8 12h8m-8 4h6',
 align:'M8 5h13M8 12h10M8 19h13M3 5h1M3 12h1M3 19h1',nodes:'M9 2h6v6H9ZM2 16h6v6H2Zm14 0h6v6h-6ZM12 8v4M5 16v-4h14v4',
 'ipa-keyboard':'M4 5h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2ZM6 9h1m4 0h1m4 0h1M6 13h1m4 0h1m4 0h1M8 16h8',
 headphones:'M3 16v-5a9 9 0 0 1 18 0v5M3 12h4v9H3Zm14 0h4v9h-4Z',
 settings:'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2',
 info:'M12 8v.1M12 11v6M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
 folder:'M2 5h7l2 3h11l-2 12H2ZM2 8h20',arrow:'M9 5l7 7-7 7',close:'M6 6l12 12M6 18 18 6',menu:'M4 6h16M4 12h16M4 18h16',
 sun:'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 1v3m0 16v3M1 12h3m16 0h3M4 4l2 2m12 12 2 2M4 20l2-2M18 6l2-2',
 clock:'M12 6v6l4 3M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',play:'M7 3l14 9-14 9Z',pause:'M7 4v16M17 4v16',stop:'M5 5h14v14H5Z',
 book:'M12 5Q6 1 2 4v16q5-3 10 0 5-3 10 0V4q-4-3-10 1Zm0 0v15',update:'M21 9a9 9 0 1 0-1 10M21 2v7h-7',
};
</script>
<template>
<span v-if="name==='ipa'" class="ipa-icon" aria-hidden="true">æ</span>
<svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path :d="paths[name||'wave']||paths.wave"/>
</svg>
</template>
