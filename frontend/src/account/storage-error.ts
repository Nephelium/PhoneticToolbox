import {message as uiMessage,type MessageDescriptor} from '../i18n/index.ts';
import {LocalStorageError} from '../platform/local-assets.ts';

const errors:Record<string,MessageDescriptor>={
 quota_exceeded:uiMessage('account.there_is_not_enough_storage_download_files_you_need'),
 upload_size_limit:uiMessage('account.the_file_exceeds_the_upload_limit_for_a_single'),
 disk_space_low:uiMessage('account.the_server_disk_is_full_existing_files_can_still'),
 storage_service_unavailable:uiMessage('account.the_file_service_is_not_enabled_or_is_temporarily'),
 storage_recovery_required:uiMessage('account.the_file_service_is_checking_storage_try_again_shortly'),
 storage_policy_migration_required:uiMessage('account.uploads_are_temporarily_unavailable_existing_valid_files_can_still'),
 web_server_busy:uiMessage('account.the_server_is_busy_wait_5_seconds_and_retry'),
 storage_busy:uiMessage('account.the_file_service_is_handling_other_requests_retry_shortly'),
 upload_closed:uiMessage('account.this_upload_has_been_closed_or_expired_select_the'),
 asset_expired:uiMessage('account.the_file_has_expired_and_can_no_longer_be'),
 asset_limit_reached:uiMessage('account.the_maximum_number_of_file_records_has_been_reached'),
 input_lifetime_too_short:uiMessage('account.the_input_file_expires_in_less_than_5_minutes'),
 file_tasks_unavailable:uiMessage('account.file_tasks_have_not_been_enabled'),
 output_budget_exceeded:uiMessage('account.this_batch_exceeds_its_output_limit_incomplete_outputs_will'),
};

export function storageErrorMessage(error:unknown):MessageDescriptor {
 if(error instanceof LocalStorageError)return error.display;
 const code=error instanceof Error?error.message:'';
 return errors[code]??uiMessage('account.the_operation_has_not_been_confirmed_retry_shortly_this');
}
