export type PublicAccountFlow='register'|'forgot'|'verify'|'reset';

/** Email tokens stay in the URL fragment until consumed by the page, never in a request URL. */
export function accountLink(fragment:string):{flow:'verify'|'reset';token:string}|null {
  const match=/^#(verify|reset)=([A-Za-z0-9_-]{43})$/.exec(fragment);
  return match ? {flow:match[1] as 'verify'|'reset',token:match[2]} : null;
}

export function accountError(code:string):string {
  return ({email_rate_limited:'请求过于频繁，请15分钟后再试。',
    email_link_invalid:'链接已失效或已使用，请重新申请邮件。',
    registration_unavailable:'注册和密码找回暂不可用，请稍后再试。',
    csrf_rejected:'页面验证已过期，请重新提交。',
    invalid_request:'请检查邮箱、登录名和密码格式。'} as Record<string,string>)[code]
    ?? '暂时无法连接账号服务，请稍后重试。';
}
