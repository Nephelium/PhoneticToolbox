import {messageText,message as uiMessage,type MessageDescriptor} from '../i18n/index.ts';
import {tr} from '../i18n/index.ts';
export type PublicAccountFlow='register'|'forgot'|'verify'|'reset';

/** Email tokens stay in the URL fragment until consumed by the page, never in a request URL. */
export function accountLink(fragment:string):{flow:'verify'|'reset';token:string}|null {
  const match=/^#(verify|reset)=([A-Za-z0-9_-]{43})$/.exec(fragment);
  return match ? {flow:match[1] as 'verify'|'reset',token:match[2]} : null;
}

export function accountErrorMessage(code:string):MessageDescriptor {
  return ({email_rate_limited:uiMessage('account.too_many_requests_try_again_in_15_minutes'),
    email_link_invalid:uiMessage('account.this_link_has_expired_or_has_already_been_used'),
    registration_unavailable:uiMessage('account.registration_and_password_recovery_are_temporarily_unavailable_try_again'),
    csrf_rejected:uiMessage('account.page_verification_has_expired_submit_again'),
    invalid_request:uiMessage('account.check_the_email_address_login_name_and_password_format')} as Record<string,MessageDescriptor>)[code]
    ?? uiMessage('account.the_account_service_is_temporarily_unreachable_retry_later');
}
export function accountError(code:string):string {return messageText(accountErrorMessage(code));}
