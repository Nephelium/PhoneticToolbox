import {tr} from '../../i18n/index.ts';

// Exact, owned parser messages from m14_table.load_v2. Preserve the original
// diagnostics and all user table cells; only project known reasons for display.
const reasonKeys:Record<string,string>={
 '数据开始行之前':'before_start',
 '表头、空行或缺少字头 / IPA':'header_blank_or_missing',
 '只有调值或重音符号，缺少音段':'missing_segment',
 '疑似多个音节，请将每个音节放在独立记录中':'multiple_syllables',
 '含未知或替换符号，请核对原始 IPA':'unknown_symbols',
};
export function diagnosticReason(reason:string):string {
 const key=reasonKeys[reason];return key?tr('modules.phonology-induction.reason.'+key):reason;
}
