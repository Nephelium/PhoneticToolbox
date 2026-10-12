import {createApp} from 'vue';
import FeedbackManager from './FeedbackManager.vue';
import {initializeFeedbackManager} from '../platform/feedback.ts';
import '../design/tokens.css';
initializeFeedbackManager().then(()=>createApp(FeedbackManager).mount('#app')).catch(error=>{document.getElementById('app')!.textContent=error.message;});
