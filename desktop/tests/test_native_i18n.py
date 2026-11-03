import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ptb_desktop.native_i18n import NativeLocale,native_text
from ptb_desktop.native_messages_data import NATIVE_MESSAGES


class NativeLocaleTests(unittest.TestCase):
    def test_default_and_supported_captions(self):
        language=NativeLocale()
        self.assertEqual(language.locale,'zh-CN')
        self.assertEqual(language.text('papers.export'),'导出论文 PDF')
        language.select('en');self.assertEqual(language.text('papers.export'),'Export paper PDF')
        language.select('ja');self.assertEqual(language.text('papers.export'),'論文PDFを書き出す')

    def test_native_dialog_error_boundary_translates_complete_product_errors_only(self):
        from ptb_desktop.cache_location import DIRECTORY_NAME
        error=ValueError(f'请选择专用的 {DIRECTORY_NAME} 缓存文件夹。')
        language=NativeLocale();self.assertEqual(language.error(error),str(error))
        language.select('en');self.assertEqual(language.error(error),'Select the dedicated PhoneticToolbox-cache folder.')
        self.assertEqual(language.error('磁盘-研究文件-Error 23'),'磁盘-研究文件-Error 23')
        language.select('ja');self.assertIn('専用',language.error(error))
        self.assertEqual(language.error('项目不匹配-original.wav'),'项目不匹配-original.wav')

    def test_bootstrap_failure_uses_the_selected_language_and_keeps_recovery_guidance(self):
        language=NativeLocale();language.select('en')
        title,body=language.startup_failure(ValueError('磁盘空间不足，请释放空间后重试。'))
        self.assertIn('could not start',title);self.assertIn('disk space',body);self.assertIn('research data',body);self.assertNotIn('{reason}',body)
        language.select('ja');title,body=language.startup_failure(OSError('unregistered-private-path'))
        self.assertIn('起動',title);self.assertIn('アクセス権',body);self.assertNotIn('unregistered-private-path',body)
        self.assertEqual(language.error('There is not enough disk space. Free up space and try again.'),'ディスクの空き容量が不足しています。空き容量を確保してから再試行してください。')

    def test_real_mfa_corpus_errors_keep_original_exception_text_and_named_file_data(self):
        from types import SimpleNamespace
        from ptb_desktop.m11_bridge import M11Bridge
        from ptb_desktop.file_provider import FileAccessError
        from ptb_desktop.native_i18n import native_error_payload
        with tempfile.TemporaryDirectory(prefix='ptb-language-mfa-corpus-') as raw:
            folder=Path(raw);name='样本-{V}'
            (folder/(name+'.wav')).write_bytes(b'RIFF')
            for suffix in ('.lab','.txt'):(folder/(name+suffix)).write_text('a1',encoding='utf-8')
            bridge=M11Bridge(SimpleNamespace(service=object()));grant=bridge.grant('corpus',folder)
            cases=[('auto','errors.mfa.uniqueTranscript',{'name':name+'.wav'}),('.TextGrid','errors.mfa.missingTranscriptFormat',{'name':name+'.wav','format':'TextGrid'})]
            for source,key,values in cases:
                with self.subTest(source=source),self.assertRaises(FileAccessError) as caught:bridge.invoke({'op':'m11_corpus','id':grant['id'],'transcript_source':source})
                error=caught.exception;payload=json.loads(json.dumps(native_error_payload(error),ensure_ascii=False))
                self.assertEqual(payload['error'],str(error));self.assertIn(name+'.wav',str(error))
                self.assertEqual(payload['display'],{'key':'native.'+key,'values':values})

    def test_file_patterns_and_named_purposes_remain_intact(self):
        language=NativeLocale()
        for code in NATIVE_MESSAGES:
            language.select(code)
            self.assertIn('*.ptb-poses.json *.json',language.text('vocal.packFilter'))
            self.assertIn('*.TextGrid',language.text('download.annotationFilter'))
            label=language.text('mfa.directory',purpose=language.text('mfa.runtime'))
            self.assertNotIn('{purpose}',label)
            self.assertNotIn('[object',label)
        language.select('en')
        self.assertEqual(language.text('mfa.directory',purpose='literal {sample}'),'Select MFA literal {sample} folder')

    def test_preference_survives_reopen_and_invalid_values_default_to_chinese(self):
        with tempfile.TemporaryDirectory(prefix='ptb-native-language-test-') as raw:
            file=Path(raw)/'language.json';language=NativeLocale(file)
            self.assertTrue(language.select('ja')['saved'])
            self.assertEqual(NativeLocale(file).locale,'ja')
            for value in [None,[],{},'fr',7,True]:
                file.write_text(json.dumps({'locale':value}),encoding='utf-8')
                self.assertEqual(NativeLocale(file).locale,'zh-CN')
            file.write_text('{bad',encoding='utf-8');self.assertEqual(NativeLocale(file).locale,'zh-CN')
            file.write_text('x'*16385,encoding='utf-8');self.assertEqual(NativeLocale(file).locale,'zh-CN')

    def test_failed_atomic_save_keeps_the_old_complete_preference(self):
        with tempfile.TemporaryDirectory(prefix='ptb-native-language-test-') as raw:
            file=Path(raw)/'language.json';language=NativeLocale(file);language.select('en')
            before=file.read_bytes()
            with patch('ptb_desktop.native_i18n.os.replace',side_effect=OSError('in use')):
                self.assertFalse(language.select('ja')['saved'])
            self.assertEqual(language.locale,'ja');self.assertEqual(file.read_bytes(),before)
            self.assertEqual(NativeLocale(file).locale,'en')
            self.assertEqual(list(file.parent.glob('*.tmp')),[])

    def test_windows_are_independent_and_invalid_requests_cannot_change_language(self):
        first,second=NativeLocale(),NativeLocale();first.select('en')
        self.assertEqual(second.locale,'zh-CN')
        for value in ['../en','fr',[],{},None]:
            with self.assertRaises(ValueError):first.select(value)
            self.assertEqual(first.locale,'en')
        self.assertEqual(native_text(object(),'papers.export'),'导出论文 PDF')

    def test_repeated_choice_avoids_rewriting_but_an_external_change_is_not_ignored(self):
        import os
        with tempfile.TemporaryDirectory(prefix='ptb-native-language-test-') as raw:
            file=Path(raw)/'language.json';language=NativeLocale(file);replace=os.replace
            with patch('ptb_desktop.native_i18n.os.replace',wraps=replace) as writes:
                for _ in range(10):self.assertTrue(language.select('en')['saved'])
                self.assertEqual(writes.call_count,1)
                other=NativeLocale(file);other.select('ja')
                language.select('en')
                self.assertEqual(writes.call_count,3)
            self.assertEqual(NativeLocale(file).locale,'en')


if __name__=='__main__':unittest.main()
