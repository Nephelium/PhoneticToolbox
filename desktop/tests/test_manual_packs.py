"""Language packs are untrusted archives, not executable application updates."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from ptb_desktop.manual_packs import MAX_INDEX_BYTES, ManualPackManager, ManualPackError


def archive(locale='en', *, text='Guide', extra=None, source='a'*64, asset_overrides=None):
    project={'schemaVersion':'ptb-manual/1','id':'phonetic-toolbox-v3','title':text,'language':locale,
             'version':'test.1','softwareVersion':'3.1.2','chapters':[{'id':'start','title':text,'path':'chapters/start.json'}],
             'assets':[],'references':[]}
    chapter={'schemaVersion':'ptb-manual-chapter/1','id':'start','title':text,'content':{'type':'doc','content':[]}}
    files={'project.json':json.dumps(project).encode(),'chapters/start.json':json.dumps(chapter).encode(),
           'assets/screen.png':b'\x89PNG\r\n\x1a\nexample-image'}
    if extra:files.update(extra)
    manifest={'schemaVersion':'ptb-manual-language-pack/1','locale':locale,'version':'test.1','sourceDigest':source,
              'projectId':project['id'],'manualVersion':'test.1','softwareVersion':'3.1.2','distribution':'software',
              'assetOverrides':asset_overrides if asset_overrides is not None else {'screen':'assets/screen.png'},'sharedAssets':[],
              'hashes':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    files['language-pack.json']=json.dumps(manifest).encode()
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as zipped:
        for name,data in files.items():zipped.writestr(name,data)
    data=output.getvalue()
    spec={'locale':locale,'version':'test.1','sourceDigest':source,'sha256':hashlib.sha256(data).hexdigest()}
    return data,spec


class ManualPacks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'cache'
        self.manager=ManualPackManager(self.root)
    def tearDown(self):self.temp.cleanup()
    def install(self,data,spec):
        request=self.manager.invoke('importBegin',{'spec':spec,'bytes':len(data)})
        import base64
        self.manager.invoke('importChunk',{'id':request['id'],'offset':0,'base64':base64.b64encode(data).decode()})
        return self.manager.invoke('importFinish',{'id':request['id']})

    def test_install_read_and_reopen_offline_without_touching_chinese(self):
        original=Path(self.temp.name)/'chinese.json';original.write_text('original',encoding='utf-8')
        data,spec=archive();installed=self.install(data,spec)
        self.assertEqual(installed['locale'],'en');self.assertEqual(self.manager.read_json('en','project.json')['title'],'Guide')
        self.assertIsNotNone(self.manager.read_asset('en','assets/screen.png'))
        reopened=ManualPackManager(self.root);self.assertEqual(reopened.list()[0]['sourceDigest'],'a'*64)
        self.assertEqual(reopened.read_json('en','chapters/start.json')['id'],'start')
        self.assertEqual(original.read_text('utf-8'),'original')

    def test_bad_hash_or_source_keeps_previously_installed_manual(self):
        data,spec=archive();self.install(data,spec)
        altered,new=archive(text='New');new['sha256']='b'*64
        with self.assertRaises(ManualPackError):self.install(altered,new)
        altered,new=archive(text='Wrong source');new['sourceDigest']='c'*64
        with self.assertRaises(ManualPackError):self.install(altered,new)
        self.assertEqual(self.manager.read_json('en','project.json')['title'],'Guide')

    def test_full_manual_metadata_supports_both_languages_and_reopening(self):
        image=b'\x89PNG\r\n\x1a\nexample-image'
        image_path='assets/'+hashlib.sha256(image).hexdigest()+'.png'
        overrides={f'manual-capture-{index:04d}-light-scaled-r20261030':image_path for index in range(343)}
        for code in ('en','ja'):
            data,spec=archive(locale=code,text='Guide '+code,extra={image_path:image},asset_overrides=overrides)
            self.install(data,spec)
        self.assertGreater((self.root/'index.json').stat().st_size,65536)
        for manager in (self.manager,ManualPackManager(self.root)):
            self.assertEqual({row['locale'] for row in manager.list()},{'en','ja'})
            for code in ('en','ja'):
                with self.subTest(reopened=manager is not self.manager,locale=code):
                    self.assertEqual(manager.read_json(code,'project.json')['title'],'Guide '+code)
                    self.assertEqual(manager.read_json(code,'chapters/start.json')['id'],'start')
                    self.assertEqual(manager.read_asset(code,image_path),image)

    def test_oversized_index_is_rejected_before_parsing(self):
        self.root.mkdir(parents=True)
        path=self.root/'index.json'
        with path.open('wb') as stream:
            stream.write(b'{}');stream.truncate(MAX_INDEX_BYTES+1)
        with patch('ptb_desktop.manual_packs.json.loads',side_effect=AssertionError('must not parse oversized index')):
            with self.assertRaisesRegex(ManualPackError,'invalidManifest'):self.manager.list()

    def test_oversized_commit_preserves_installed_index_and_pack(self):
        data,spec=archive(text='Old guide');self.install(data,spec)
        path=self.root/'index.json';index=path.read_bytes()
        before=set((self.root/'packs/en').iterdir())
        # Non-ASCII metadata makes the encoded JSON larger than its character count.
        overrides={'画面'*10:'assets/screen.png'}
        data,spec=archive(text='New guide',asset_overrides=overrides)
        proposed=json.loads(index);proposed['en']['metadata']['assetOverrides']=overrides
        encoded=json.dumps(proposed,ensure_ascii=False,allow_nan=False)
        limit=len(encoded)+1
        self.assertLess(len(index),limit)
        self.assertGreater(len(encoded.encode('utf-8')),limit)
        with patch('ptb_desktop.manual_packs.MAX_INDEX_BYTES',limit):
            with self.assertRaisesRegex(ManualPackError,'invalidManifest'):self.install(data,spec)
        self.assertEqual(path.read_bytes(),index)
        self.assertEqual(set((self.root/'packs/en').iterdir()),before)
        self.assertEqual(list(self.root.glob('index.*.tmp')),[])
        for manager in (self.manager,ManualPackManager(self.root)):
            self.assertEqual(manager.read_json('en','project.json')['title'],'Old guide')
            self.assertIsNotNone(manager.read_asset('en','assets/screen.png'))

    def test_traversal_and_executable_members_are_rejected(self):
        for name in ['../escaped.json','assets/../../escape.png','assets/script.js','C:/escape.json','assets/a\\b.png']:
            with self.subTest(name=name):
                data,spec=archive(extra={name:b'bad'})
                with self.assertRaises(ManualPackError):self.install(data,spec)
        self.assertFalse((Path(self.temp.name)/'escaped.json').exists());self.assertEqual(self.manager.list(),[])

    def test_wrong_chunk_order_does_not_commit_partial_install(self):
        data,spec=archive();request=self.manager.invoke('importBegin',{'spec':spec,'bytes':len(data)})
        with self.assertRaises(ManualPackError):self.manager.invoke('importChunk',{'id':request['id'],'offset':1,'base64':'YWJj'})
        self.manager.invoke('importCancel',{'id':request['id']});self.assertEqual(self.manager.list(),[])

    def test_external_urls_are_rejected_before_any_network_access(self):
        _,spec=archive()
        for url in ['file:///C:/secret','http://127.0.0.1:1/manual.zip','https://example.com/manual.zip','https://user:password@www.phonetictoolbox.com/a.zip']:
            with self.subTest(url=url),self.assertRaises(ManualPackError):self.manager.install({**spec,'url':url})

    def test_cache_reads_check_integrity_and_explicit_path_scope(self):
        data,spec=archive();self.install(data,spec)
        for path in ['../index.json','language-pack.json','assets/screen.png']:
            with self.subTest(path=path),self.assertRaises(ManualPackError):self.manager.read_json('en',path)
        for path in ['../index.json','project.json','assets/../index.json']:
            with self.subTest(path=path),self.assertRaises(ManualPackError):self.manager.read_asset('en',path)
        cached=next((self.root/'packs/en').glob('*/assets/screen.png'));cached.write_bytes(b'corrupted image')
        with self.assertRaises(ManualPackError):ManualPackManager(self.root).read_asset('en','assets/screen.png')

    def test_repeated_reads_do_not_reload_pack_files(self):
        data,spec=archive();self.install(data,spec)
        manager=ManualPackManager(self.root);opened={};original=Path.open
        def observe(path,*args,**kwargs):
            opened[str(path)]=opened.get(str(path),0)+1;return original(path,*args,**kwargs)
        with patch.object(Path,'open',observe):
            for _ in range(10):manager.read_json('en','project.json');manager.read_asset('en','assets/screen.png')
        for suffix in ['index.json','language-pack.json','project.json','assets/screen.png']:
            paths=[count for path,count in opened.items() if path.replace('\\','/').endswith('/'+suffix)]
            self.assertEqual(paths,[1],suffix)

    def test_real_http_download_is_verified_and_same_pack_is_not_downloaded_twice(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        import threading
        data,spec=archive();calls=[]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.path);self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            manager=ManualPackManager(self.root,allow_loopback=True);request={**spec,'url':f'http://127.0.0.1:{server.server_port}/manual.zip'}
            manager.install(request);manager.install(request)
            self.assertEqual(calls,['/manual.zip']);self.assertEqual(manager.read_json('en','project.json')['language'],'en')
        finally:server.shutdown();server.server_close();thread.join(5)

    def test_corrupted_installed_file_can_be_repaired_by_the_same_verified_download(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        import threading
        data,spec=archive();calls=[]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.path);self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            manager=ManualPackManager(self.root,allow_loopback=True);request={**spec,'url':f'http://127.0.0.1:{server.server_port}/download'}
            manager.install(request);original=manager.read_asset('en','assets/screen.png')
            old=next((self.root/'packs/en').glob('*/assets/screen.png'));old.write_bytes(b'broken')
            with self.assertRaises(ManualPackError):manager.read_asset('en','assets/screen.png')
            manager.install(request);self.assertEqual(manager.read_asset('en','assets/screen.png'),original)
            self.assertEqual(calls,['/download','/download'])
            manager.install(request);self.assertEqual(calls,['/download','/download'])
        finally:server.shutdown();server.server_close();thread.join(5)

    def test_two_independent_managers_commit_different_languages_without_losing_either(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import os
        barrier=threading.Barrier(2);replace=os.replace
        inputs=[]
        for code in ('en','ja'):
            data,spec=archive(locale=code);path=self.root/(code+'.zip');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            inputs.append((ManualPackManager(self.root),path,spec))
        def coordinated(source,target):
            replace(source,target)
            if Path(target).parent.parent==self.root/'packs':barrier.wait(3)
        with patch('ptb_desktop.manual_packs.os.replace',side_effect=coordinated),ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(manager._install_archive,path,spec) for manager,path,spec in inputs]
            for future in futures:future.result(timeout=8)
        reopened=ManualPackManager(self.root)
        self.assertEqual({row['locale'] for row in reopened.list()},{'en','ja'})
        for code in ('en','ja'):self.assertEqual(reopened.read_json(code,'project.json')['language'],code)

    def test_failed_index_replace_keeps_the_old_visible_pack_and_removes_only_the_uncommitted_target(self):
        import os
        data,spec=archive(text='Old guide');self.install(data,spec)
        old=self.manager.read_json('en','project.json');index=(self.root/'index.json').read_bytes()
        before=set((self.root/'packs/en').iterdir())
        data,new=archive(text='New guide');replace=os.replace
        def failed(source,target):
            if Path(target)==self.root/'index.json':raise PermissionError('index in use')
            return replace(source,target)
        with patch('ptb_desktop.manual_packs.os.replace',side_effect=failed),self.assertRaises(ManualPackError):self.install(data,new)
        self.assertEqual((self.root/'index.json').read_bytes(),index)
        self.assertEqual(self.manager.read_json('en','project.json'),old)
        self.assertEqual(set((self.root/'packs/en').iterdir()),before)
        self.assertEqual(ManualPackManager(self.root).read_json('en','project.json'),old)

    def test_close_cancels_only_its_own_unfinished_imports_and_keeps_the_installed_manual(self):
        data,spec=archive();self.install(data,spec)
        installed=(self.root/'index.json').read_bytes()
        owned=self.manager.invoke('importBegin',{'spec':spec,'bytes':len(data)})
        other=ManualPackManager(self.root)
        foreign=other.invoke('importBegin',{'spec':spec,'bytes':len(data)})
        owned_path=self.manager._imports[owned['id']]['path'];foreign_path=other._imports[foreign['id']]['path']
        self.manager.close();self.manager.close()
        self.assertFalse(owned_path.exists());self.assertTrue(foreign_path.is_file())
        self.assertEqual((self.root/'index.json').read_bytes(),installed)
        self.assertEqual(ManualPackManager(self.root).read_json('en','project.json')['title'],'Guide')
        with self.assertRaisesRegex(ManualPackError,'cancelled'):self.manager.invoke('list',{})
        other.close();self.assertFalse(foreign_path.exists())

    def test_close_before_commit_keeps_the_old_pack_and_cleans_the_uncommitted_one(self):
        data,spec=archive(text='Old guide');self.install(data,spec)
        index=(self.root/'index.json').read_bytes();before=set((self.root/'packs/en').iterdir())
        data,spec=archive(text='New guide');write=Path.write_bytes
        def close_before_commit(path,*args,**kwargs):
            result=write(path,*args,**kwargs)
            if path.parent==self.root and path.name.startswith('index.') and path.suffix=='.tmp':self.manager.close()
            return result
        with patch.object(Path,'write_bytes',close_before_commit),self.assertRaisesRegex(ManualPackError,'cancelled'):self.install(data,spec)
        self.assertEqual((self.root/'index.json').read_bytes(),index)
        self.assertEqual(set((self.root/'packs/en').iterdir()),before)
        self.assertEqual(ManualPackManager(self.root).read_json('en','project.json')['title'],'Old guide')

    def test_close_cooperatively_stops_an_active_http_transfer_and_removes_its_temporary_zip(self):
        from concurrent.futures import ThreadPoolExecutor
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        import threading
        data,spec=archive();started=threading.Event();release=threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers()
                started.set();release.wait(5);self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        manager=ManualPackManager(self.root,allow_loopback=True)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(manager.install,{**spec,'url':f'http://127.0.0.1:{server.server_port}/download'})
                self.assertTrue(started.wait(3));manager.close();release.set()
                with self.assertRaisesRegex(ManualPackError,'cancelled'):future.result(timeout=3)
            self.assertEqual(list((self.root/'.downloads').glob('*.zip')),[])
            self.assertEqual(ManualPackManager(self.root).list(),[])
        finally:release.set();server.shutdown();server.server_close();thread.join(5)


if __name__=='__main__':unittest.main()
