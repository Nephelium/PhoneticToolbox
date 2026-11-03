<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {ref,watch,onMounted,onUnmounted,nextTick} from 'vue';
import ModalDialog from './ModalDialog.vue';
import {feedbackAuthor} from '../platform/feedback-profile.ts';
import {feedbackCall,prepareFeedbackImages,feedbackTime,feedbackClientLabel,onFeedbackNotification,type FeedbackConversation,type FeedbackMessage,type FeedbackImage} from '../platform/feedback.ts';
import {loadFeedbackDraft,saveFeedbackDraft,newFeedbackDraft} from '../state/feedback-drafts.ts';
const props=defineProps<{admin?:boolean;owner?:string;trashed?:boolean;managementBusy?:boolean;trashAvailable?:boolean}>();
const emit=defineEmits<{updated:[];trash:[];restore:[]}>();
const messages=ref<FeedbackMessage[]>([]),text=ref(''),images=ref<FeedbackImage[]>([]),busy=ref(false),loading=ref(false),error=messageRef(),warning=messageRef(),status=ref('open'),timeline=ref<HTMLElement>();
const imagePreview=ref('');
const mailConfigured=ref(false),emailVerified=ref(false),mailNotifications=ref(true),savingMail=ref(false),mailNote=messageRef();
let requestId:string=crypto.randomUUID(),contentHash='',timer:ReturnType<typeof setInterval>|undefined,generation=0,alive=true,draftReady=false,refreshing=false;
const draftKey=()=>`${props.admin?'admin':'user'}:${props.owner??'self'}`;
const args=()=>props.admin?{owner:props.owner}:{};
function draft(){return {text:text.value,images:images.value.map(i=>({...i})),requestId,contentHash};}
async function persist(){if(!draftReady)return;try{await saveFeedbackDraft(draftKey(),draft());}catch{warning.value=uiMessage('components.the_draft_could_not_be_saved_locally_keep_a');}}
async function discardDraft(){draftReady=false;generation++;text.value='';images.value=[];requestId=newFeedbackDraft().requestId;contentHash='';await saveFeedbackDraft(draftKey(),draft());}
defineExpose({discardDraft});
watch([text,images],()=>{void persist();},{deep:true});
async function refresh(){if(refreshing||!alive)return;refreshing=true;const own=++generation,owner=props.owner;loading.value=true;try{const all:FeedbackMessage[]=[];let after=0,more=true;while(more&&alive&&own===generation){const result=await feedbackCall<FeedbackConversation>('messages',{...(props.admin?{owner}:{}),after});all.push(...result.messages);after=result.cursor;more=result.hasMore;status.value=result.status;mailConfigured.value=result.mailConfigured;emailVerified.value=result.emailVerified;mailNotifications.value=result.mailNotifications??true;}if(!alive||own!==generation||owner!==props.owner)return;const grew=all.length>messages.value.length;messages.value=all;error.value='';if(all.length&&!(props.admin&&props.trashed)&&document.visibilityState==='visible'){await feedbackCall('read',{...args(),through:all.at(-1)!.id});emit('updated');}if(grew){await nextTick();if(timeline.value)timeline.value.scrollTop=timeline.value.scrollHeight;}}catch(exc){if(alive&&own===generation)error.value=uiErrorMessage((exc as Error));}finally{if(own===generation)loading.value=false;refreshing=false;}}
async function open(){generation++;draftReady=false;messages.value=[];error.value='';const owner=props.owner;try{const value=await loadFeedbackDraft(draftKey());if(!alive||owner!==props.owner)return;text.value=value.text;images.value=value.images;requestId=value.requestId;contentHash=value.contentHash??'';draftReady=true;}catch{draftReady=true;warning.value=uiMessage('components.local_draft_storage_is_unavailable');}await refresh();}
watch(()=>props.owner,()=>void open());
async function add(event:Event){const input=event.target as HTMLInputElement;try{const next=await prepareFeedbackImages(Array.from(input.files??[]));if(images.value.length+next.length>4)throw uiError(uiMessage('components.you_can_attach_up_to_4_images_per_message'));images.value.push(...next);error.value='';}catch(exc){error.value=uiErrorMessage((exc as Error));}input.value='';}
async function send(){if(props.trashed||busy.value||!text.value.trim()&&!images.value.length)return;busy.value=true;error.value='';try{const body={text:text.value,images:images.value.map(i=>({data:i.data}))};const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(body))))).map(x=>x.toString(16).padStart(2,'0')).join('');if(contentHash&&contentHash!==digest)requestId=crypto.randomUUID();contentHash=digest;await persist();await feedbackCall('send',{...args(),...body,requestId});text.value='';images.value=[];requestId=newFeedbackDraft().requestId;contentHash='';await persist();await refresh();}catch(exc){error.value=uiErrorMessage((exc as Error));}finally{busy.value=false;}}
async function resolve(){try{await feedbackCall('resolve',{owner:props.owner,status:status.value==='resolved'?'open':'resolved'});status.value=status.value==='resolved'?'open':'resolved';emit('updated');}catch(exc){error.value=uiErrorMessage((exc as Error));}}
async function retryMail(id:number){try{await feedbackCall('retryMail',{id});await refresh();}catch(exc){error.value=uiErrorMessage((exc as Error));}}
async function changeMail(event:Event){const input=event.target as HTMLInputElement;const enabled=input.checked;input.checked=mailNotifications.value;savingMail.value=true;mailNote.value='';try{const value=await feedbackCall<{mailNotifications:boolean}>('setMailNotifications',{mailNotifications:enabled});mailNotifications.value=value.mailNotifications;mailNote.value=enabled?uiMessage('components.email_notifications_enabled'):uiMessage('components.email_notifications_disabled_replies_are_still_saved_in_the');await refresh();}catch(exc){error.value=uiErrorMessage((exc as Error));}finally{savingMail.value=false;}}
let unsubscribe=()=>{};
function refreshVisible(){if(!document.hidden)void refresh();}
onMounted(()=>{void open();timer=setInterval(refreshVisible,15000);document.addEventListener('visibilitychange',refreshVisible);unsubscribe=onFeedbackNotification(event=>{if(props.admin&&event.owner===props.owner&&['new_feedback','conversation_changed'].includes(event.kind))refreshVisible();});});
onUnmounted(()=>{alive=false;generation++;clearInterval(timer);document.removeEventListener('visibilitychange',refreshVisible);unsubscribe();void persist();});
</script>
<template>
<section class="feedback-thread" :aria-label="tr('components.feedback_conversation')">
  <header v-if="!admin" class="thread-profile">
    <img class="author-avatar" :src="feedbackAuthor.avatar" :alt="feedbackAuthor.name+tr('components.s_avatar')"/>
    <div><strong>{{feedbackAuthor.name}}</strong><span>{{tr('components.phonetictoolbox_feedback')}}</span></div>
    <span class="profile-badge">{{tr('components.keep_in_touch')}}</span>
  </header>
  <div class="thread-toolbar">
    <span class="muted">{{admin&&trashed?tr('components.this_conversation_is_in_the_trash_restore_it_to'):status==='resolved'?tr('components.resolved_send_another_message_to_continue_the_conversation'):admin?(mailNotifications&&emailVerified&&mailConfigured?tr('components.replies_appear_in_the_app_and_trigger_email_notifications'):tr('components.replies_appear_in_the_app_email_notifications_are_disabled')):tr('components.replies_are_saved_here_you_can_return_and_continue')}}</span>
    <button :disabled="loading" @click="refresh">{{tr('components.refresh')}}</button>
    <button v-if="admin&&!trashed" :disabled="busy||managementBusy" @click="resolve">{{status==='resolved'?tr('components.reopen'):tr('components.mark_resolved')}}</button>
    <button v-if="admin&&!trashed" class="trash-conversation" :disabled="!trashAvailable||busy||loading||managementBusy" @click="emit('trash')">{{tr('components.delete_conversation')}}</button>
    <button v-if="admin&&trashed" class="primary" :disabled="!trashAvailable||managementBusy" @click="emit('restore')">{{tr('components.restore_conversation')}}</button>
  </div>
  <p v-if="admin&&!trashAvailable" class="trash-unavailable muted" role="status">{{tr('components.the_trash_is_not_enabled_on_this_server_update')}}</p>
  <div v-if="!admin" class="mail-preference">
    <label class="mail-toggle"><input type="checkbox" role="switch" :checked="mailNotifications" :disabled="savingMail||loading" @change="changeMail"/><span>{{tr('components.email_me_when_a_reply_arrives')}}</span></label>
    <small>{{savingMail?tr('components.saving'):mailNotifications?(mailConfigured&&emailVerified?tr('components.one_email_per_new_reply_you_can_turn_this'):tr('components.available_after_email_verification_and_enabling_email_service')):tr('components.disabled_replies_still_appear_in_the_app')}}</small>
  </div>
  <p v-if="mailNote" role="status" class="mail-note">{{mailNote}}</p>
  <p v-if="error" role="alert" class="feedback-error">{{error}}</p><p v-if="warning" role="status" class="muted">{{warning}}</p>
  <div ref="timeline" class="thread-timeline" aria-live="polite">
    <div v-if="!messages.length" class="thread-empty"><span class="empty-chat" aria-hidden="true">✉</span><h3>{{loading?tr('components.loading_conversation'):tr('components.tell_jing_shaozi_about_the_issue')}}</h3><p>{{tr('components.describe_the_steps_or_attach_a_screenshot')}}<br/>{{tr('components.check_images_for_personal_information_and_research_data_before')}}</p></div>
    <div v-for="message in messages" :key="message.id" :class="['message-row',{own:message.role===(admin?'admin':'user')}]">
      <img v-if="message.role==='admin'" class="message-avatar author-avatar" :src="feedbackAuthor.avatar" :alt="feedbackAuthor.name+tr('components.s_avatar')"/>
      <span v-else class="message-avatar user-avatar" aria-hidden="true">{{admin?tr('components.u'):tr('components.you')}}</span>
      <div class="message-content">
        <div class="message-meta"><strong>{{message.role==='admin'?feedbackAuthor.name:admin?tr('components.user'):tr('components.you')}}</strong><time>{{feedbackTime(message.created)}}</time><span v-if="admin&&message.role==='user'" class="message-source">{{feedbackClientLabel(message.client)}}</span></div>
        <article :class="['thread-message',{own:message.role===(admin?'admin':'user')}]">
          <p v-if="message.text" class="message-text">{{message.text}}</p>
          <div v-if="message.images.length" class="message-images"><button v-for="(image,i) in message.images" :key="image.id??i" type="button" :aria-label="tr('components.enlarge_feedback_image')" @click="imagePreview=`data:${image.mime};base64,${image.data}`"><img :src="`data:${image.mime};base64,${image.data}`" :alt="tr('components.feedback_screenshot_click_to_view')"/></button></div>
        </article>
        <div v-if="admin&&message.role==='admin'" class="message-delivery">{{message.mailError==='conversation_trashed'?tr('components.the_conversation_was_moved_to_the_trash_and_email'):({sent:tr('components.the_email_was_handed_to_the_mail_server'),pending:tr('components.the_email_is_awaiting_delivery_or_retry'),sending:tr('components.sending_email'),failed:tr('components.email_delivery_failed'),manual:tr('components.sent_to_the_app_only'),disabled:tr('components.the_user_disabled_email_notifications'),skipped:tr('components.email_cancelled_because_the_user_disabled_notifications')} as Record<string,string>)[message.mailState??'']??tr('components.sent_to_the_app_only')}}<button v-if="message.mailState==='failed'&&mailNotifications&&!trashed" @click="retryMail(message.id)">{{tr('components.retry_email')}}</button></div>
      </div>
    </div>
  </div>
  <form v-if="!admin||!trashed" class="thread-compose" @submit.prevent="send">
    <div v-if="images.length" class="draft-images"><div v-for="(image,index) in images" :key="index"><img :src="image.preview??`data:${image.mime};base64,${image.data}`" :alt="tr('components.images_to_send')"/><button type="button" :disabled="busy" :aria-label="tr('components.remove_image_value',{v0:(index+1)})" @click="images.splice(index,1)">{{tr('components.remove')}}</button></div></div>
    <textarea v-model="text" :disabled="busy" maxlength="12000" rows="3" :placeholder="admin?tr('components.write_a_reply_to_the_user'):tr('components.describe_the_issue_or_continue_the_conversation')" :aria-label="tr('components.message')" @keydown.ctrl.enter.prevent="send"/>
    <div class="compose-actions"><label class="image-picker">{{tr('components.attach_images')}}<input type="file" accept="image/png,image/jpeg,image/webp" multiple :disabled="busy" @change="add"/></label><small>{{tr('components.up_to_4_images_2_mb_each_ctrl_enter')}}</small><button class="primary" :disabled="busy||!text.trim()&&!images.length">{{busy?tr('components.sending'):admin?tr('components.send_reply'):tr('components.send')}}</button></div>
  </form>
</section>
<ModalDialog v-if="imagePreview" :title="tr('components.feedback_image')" wide @close="imagePreview=''"><img :src="imagePreview" :alt="tr('components.enlarged_feedback_image')" style="display:block;max-width:100%;max-height:70vh;margin:auto;object-fit:contain"/></ModalDialog>
</template>
<style scoped>
.message-meta{flex-wrap:wrap}.message-source{overflow-wrap:anywhere}
.feedback-thread{display:flex;flex-direction:column;flex:1;min-height:0;height:100%;gap:10px}
.thread-profile{display:flex;align-items:center;gap:12px;padding:12px 16px;background:var(--panel);border:1px solid var(--border);border-radius:16px}
.author-avatar{object-fit:cover;object-position:50% 32%;border-radius:50%;border:1px solid var(--border)}
.thread-profile>img{width:44px;height:44px}.thread-profile>div{display:flex;flex-direction:column;gap:3px}.thread-profile strong{font-size:1rem}.thread-profile>div>span{font-size:.78rem;color:var(--muted)}
.profile-badge{margin-left:auto;color:var(--accent);font-size:.75rem;background:var(--selected);padding:5px 10px;border-radius:20px;white-space:nowrap}
.thread-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:.82rem}.thread-toolbar>span{flex:1}.thread-toolbar button{font-size:.8rem;min-height:30px}
.mail-preference{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:2px 4px}.mail-toggle{display:flex;align-items:center;gap:8px;font-size:.82rem;cursor:pointer}.mail-toggle input{appearance:none;width:34px;height:20px;min-height:0;padding:0;border:1px solid var(--border);border-radius:20px;background:var(--muted);position:relative;cursor:pointer;flex:none}.mail-toggle input:before{content:'';position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:var(--panel)}.mail-toggle input:checked{background:var(--accent);border-color:var(--accent)}.mail-toggle input:checked:before{left:16px}.mail-toggle input:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.mail-toggle input:disabled{opacity:.55;cursor:wait}.mail-preference small{font-size:.75rem;color:var(--muted)}.mail-note{margin:0;font-size:.78rem;color:var(--accent)}
.thread-timeline{flex:1;min-height:150px;overflow:auto;padding:20px;background:var(--app);border:1px solid var(--border);border-radius:16px;display:flex;flex-direction:column;gap:22px;overscroll-behavior:contain}
.thread-empty{margin:auto;text-align:center;max-width:440px;color:var(--muted);line-height:1.8}.thread-empty h3{font-size:1rem;font-weight:500;color:var(--text)}.thread-empty p{font-size:.84rem}.empty-chat{display:inline-grid;place-items:center;width:54px;height:54px;font-size:1.5rem;background:var(--selected);color:var(--accent);border-radius:18px}
.message-row{display:flex;align-items:flex-start;gap:10px;max-width:88%;align-self:flex-start;min-width:0}.message-row.own{align-self:flex-end;flex-direction:row-reverse}.message-avatar{width:34px;height:34px;flex:none;margin-top:22px}.user-avatar{display:grid;place-items:center;border-radius:50%;background:var(--selected);color:var(--accent);font-size:.8rem;font-weight:600}.message-content{min-width:0}.message-meta{display:flex;align-items:center;gap:10px;font-size:.74rem;color:var(--muted);margin:0 3px 6px}.message-meta strong{font-weight:500;color:var(--text)}.own .message-meta{justify-content:flex-end}.thread-message{padding:12px 16px;background:var(--panel);border:1px solid var(--border);border-radius:4px 16px 16px 16px;box-shadow:0 2px 5px #00000004}.thread-message.own{background:var(--selected);border-radius:16px 4px 16px 16px}.message-text{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;line-height:1.65;font-size:.92rem}.message-images,.draft-images{display:flex;flex-wrap:wrap;gap:8px}.message-text+.message-images{margin-top:10px}.message-images button{padding:0;border:0;background:transparent;overflow:hidden;border-radius:9px}.message-images img{display:block;max-width:min(220px,100%);max-height:200px;object-fit:contain}.message-delivery{color:var(--muted);font-size:.7rem;margin:7px 2px 0;text-align:right}.message-delivery button{margin-left:5px;font-size:.7rem;min-height:24px;padding:2px 7px}
.thread-compose{display:flex;flex-direction:column;gap:8px;padding:12px;background:var(--panel);border:1px solid var(--border);border-radius:16px}.thread-compose textarea{font:inherit;font-size:.92rem;resize:vertical;min-height:66px;max-height:180px;padding:5px;border:0;border-radius:8px;color:var(--text);background:transparent}.thread-compose textarea:focus{outline:1px solid var(--accent);outline-offset:2px}.compose-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.compose-actions small{flex:1;font-size:.72rem;color:var(--muted)}.compose-actions .primary{min-width:70px;border-radius:10px}.image-picker{position:relative;padding:7px 10px;color:var(--text);font-size:.82rem;border:1px solid var(--border);border-radius:9px;cursor:pointer}.image-picker:focus-within{outline:2px solid var(--accent);outline-offset:2px}.image-picker input{position:absolute;inset:0;opacity:0;width:100%;cursor:pointer}.draft-images>div{display:flex;align-items:center;gap:5px}.draft-images img{width:70px;height:54px;object-fit:contain;border-radius:6px}.feedback-error{color:var(--danger);font-size:.9rem;margin:0}
@media(max-width:700px){.thread-timeline{padding:14px 10px;gap:18px}.message-row{max-width:96%;gap:7px}.message-avatar{width:28px;height:28px}.thread-message{padding:10px 12px}.message-meta{font-size:.68rem;gap:7px}.thread-profile{padding:10px 12px}.thread-profile>img{width:38px;height:38px}.thread-compose{padding:10px}.compose-actions small{display:none}.compose-actions .primary{margin-left:auto}.mail-preference{gap:5px}}
</style>
