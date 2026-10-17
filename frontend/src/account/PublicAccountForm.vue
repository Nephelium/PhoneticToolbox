<script setup lang="ts">
import {computed,ref} from 'vue';
import type {components} from '../../../contracts/generated/api';
import {accountError,type PublicAccountFlow} from './public-account.ts';
const props=defineProps<{flow:PublicAccountFlow;token?:string}>();
const emit=defineEmits<{back:[message?:string]}>();
const email=ref(''),username=ref(''),password=ref(''),busy=ref(false),message=ref(''),accepted=ref(false);
const title=computed(()=>({register:'创建研究者账号',forgot:'找回密码',verify:'验证邮箱',reset:'设置新密码'}[props.flow]));
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
  if(props.flow==='verify'||props.flow==='reset')emit('back',props.flow==='verify'?'邮箱验证成功，请登录。':'密码已更新，请使用新密码登录。');
  else{accepted.value=true;message.value=props.flow==='forgot'?'若该邮箱已关联账号，将收到重设密码的邮件。':'请检查邮箱中的验证链接。若未收到，请检查垃圾邮件或稍后重新提交。';}
 }catch(error){message.value=accountError(error instanceof Error?error.message:'unavailable');}
 finally{busy.value=false;}
}
</script>
<template>
 <section class="account-card login-card" :aria-label="title">
  <h1>{{title}}</h1>
  <p v-if="message" role="status" class="account-message">{{message}}</p>
  <form v-if="!accepted" @submit.prevent="submit">
   <label v-if="flow==='register'">登录名<input v-model="username" autocomplete="username" minlength="3" maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{2,63}" required :disabled="busy"/><small>3–64个英文字母、数字、点、下划线或连字符。</small></label>
   <label v-if="flow==='register'||flow==='forgot'">邮箱<input v-model="email" type="email" autocomplete="email" maxlength="254" required :disabled="busy"/></label>
   <label v-if="flow==='register'||flow==='reset'">{{flow==='reset'?'新密码':'密码'}}<input v-model="password" type="password" autocomplete="new-password" minlength="12" maxlength="128" required :disabled="busy"/><small>12–128个字符。</small></label>
   <p v-if="flow==='verify'" class="muted">确认后即可使用该邮箱完成注册。</p>
   <button class="primary" type="submit" :disabled="busy">{{busy?'正在处理…':flow==='verify'?'确认验证':flow==='reset'?'保存新密码':'发送邮件'}}</button>
  </form>
  <button type="button" :disabled="busy" @click="emit('back')">返回登录</button>
 </section>
</template>
