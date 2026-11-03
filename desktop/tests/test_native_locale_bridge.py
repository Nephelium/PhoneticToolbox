import json
from types import SimpleNamespace
import unittest
from ptb_desktop.host import Bridge
from ptb_desktop.native_i18n import NativeLocale


class NativeLocaleBridgeTests(unittest.TestCase):
    def test_hello_and_explicit_locale_request_share_the_window_language(self):
        window=SimpleNamespace(native_locale=NativeLocale())
        fake=SimpleNamespace(window=window,provider=SimpleNamespace(session='session'),
                             service=SimpleNamespace(local_files_root=None,get=lambda _:{'api_version':'1.1.0'}))
        hello=lambda:json.loads(Bridge.invoke(fake,json.dumps({'op':'hello'})))['value']
        self.assertEqual(hello()['ui_locale'],'zh-CN')
        result=json.loads(Bridge.invoke(fake,json.dumps({'op':'locale','purpose':'ja'})))
        self.assertTrue(result['ok']);self.assertTrue(result['value']['saved'])
        self.assertEqual(hello()['ui_locale'],'ja')
        self.assertEqual(window.native_locale.text('papers.export'),'論文PDFを書き出す')
        for value in ['fr','../en',{},None]:
            result=json.loads(Bridge.invoke(fake,json.dumps({'op':'locale','purpose':value})))
            self.assertFalse(result['ok']);self.assertEqual(hello()['ui_locale'],'ja')


if __name__=='__main__':unittest.main()
