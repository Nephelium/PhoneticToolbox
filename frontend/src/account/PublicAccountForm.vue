<script setup lang="ts">
import {messageRef,message as uiMessage,type MessageDescriptor} from '../i18n/index.ts';
import {tr} from '../i18n/index.ts';
import {computed,ref} from 'vue';
import type {components} from '../../../contracts/generated/api';
import {accountErrorMessage as accountError,type PublicAccountFlow} from './public-account.ts';
const props=defineProps<{flow:PublicAccountFlow;token?:string}>();
const emit=defineEmits<{back:[message?:string|MessageDescriptor]}>();
const email=ref(''),username=ref(''),password=ref(''),busy=ref(false),message=messageRef(),accepted=ref(false);
const title=computed(()=>({register:tr('account.create_a_researcher_account'),forgot:tr('account.reset_password'),verify:tr('account.verify_email'),reset:tr('account.set_a_new_password')}[props.flow]));
async function submit(){
 if(busy.value)return;
 busy.value=true;message.value='';
 try{
  const challengeResponse=await fetch('/api/v1/auth/challenge',{credentials:'same-origin',cache:'no-store'});
  if(!challengeResponse.ok)throw Error('registration_unavailable');
  const challenge:components['schemas']['Challenge']=await challengeResponse.json();
  const action={register:'register',forgot:'forgot-password',verify:'verify-email',reset:'reset-password'}[props.flow];
  const body=props.flow==='register'?{email:email.value,username:username.value,password:password.value}
   :props.flow==='forgot'?{email:email.value}:props.flow==='verify'?{token:props.token}:{token:props.token,password:password.value};
  const response=await fetch('/api/v1/auth/'+action,{method:'POST',credentials:'same-origin',cache:'no-store',
   headers:{'Content-Type':'application/json','X-CSRF-Token':challenge.csrf_token},body:JSON.stringify(body)});
  if(!response.ok){const error=await response.json().catch(()=>({}));throw Error(typeof error.detail==='string'?error.detail:'unavailable');}
  password.value='';
  if(props.flow==='verify'||props.flow==='reset')emit('back',props.flow==='verify'?uiMessage('account.your_email_was_verified_please_log_in'):uiMessage('account.your_password_was_updated_log_in_with_the_new'));
  else{accepted.value=true;message.value=props.flow==='forgot'?uiMessage('account.if_this_email_address_is_associated_with_an_account'):uiMessage('account.check_your_email_for_the_verification_link_if_it');}
 }catch(error){message.value=accountError(error instanceof Error?error.message:'unavailable');}
 finally{busy.value=false;}
}
</script>
<template>
 <section class="account-card login-card" :aria-label="title">
  <h1>{{title}}</h1>
  <p v-if="message" role="status" class="account-message">{{message}}</p>
  <form v-if="!accepted" @submit.prevent="submit">
   <label v-if="flow==='register'">{{tr('account.login_name')}}<input v-model="username" autocomplete="username" minlength="3" maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{2,63}" required :disabled="busy"/><small>{{tr('account.3_64_latin_letters_numbers_dots_underscores_or_hyphens')}}</small></label>
   <label v-if="flow==='register'||flow==='forgot'">{{tr('account.email')}}<input v-model="email" type="email" autocomplete="email" maxlength="254" required :disabled="busy"/></label>
   <label v-if="flow==='register'||flow==='reset'">{{flow==='reset'?tr('account.new_password'):tr('account.password')}}<input v-model="password" type="password" autocomplete="new-password" minlength="12" maxlength="128" required :disabled="busy"/><small>{{tr('account.12_128_characters')}}</small></label>
   <p v-if="flow==='verify'" class="muted">{{tr('account.confirm_to_complete_registration_with_this_email_address')}}</p>
   <button class="primary" type="submit" :disabled="busy">{{busy?tr('account.processing'):flow==='verify'?tr('account.confirm_verification'):flow==='reset'?tr('account.save_new_password'):tr('account.send_email')}}</button>
  </form>
  <button type="button" :disabled="busy" @click="emit('back')">{{tr('account.return_to_login')}}</button>
 </section>
</template>
