"""M11-R2: explicit public /a/ probe spellings, never an arbitrary lexicon word."""
import pytest
from ptb_worker.mfa.probe import probe_word


@pytest.mark.parametrize('content,expected', [
    ('a\ta\n', 'a'), ('啊\ta˥˥\n', '啊'), ('a1\ta1\n', 'a1'),
    ('\ufeffa1\ta1\r\n', 'a1'),
    ('a1\ta1\n啊\ta˥˥\na\ta\n', 'a'),
    ('a1\ta1\n啊\ta˥˥\n', '啊'),
])
def test_m11_r2_probe_supports_exact_a_spellings(tmp_path, content, expected):
    dictionary = tmp_path / 'dictionary.dict'
    dictionary.write_bytes(content.encode('utf8'))
    assert probe_word(dictionary) == expected
    assert dictionary.read_bytes() == content.encode('utf8')


@pytest.mark.parametrize('content', ['dai1\td ai1\n', 'a10\ta1\n', 'a1_extra\ta1\n'])
def test_m11_r2_probe_does_not_guess_other_words(tmp_path, content):
    dictionary = tmp_path / 'dictionary.dict'
    dictionary.write_text(content, encoding='utf8')
    with pytest.raises(ValueError, match='m11_probe_word_unavailable'):
        probe_word(dictionary)


def test_m11_r2_recheck_managed_resources_keeps_names_without_renaming_inputs(tmp_path):
    import json
    from ptb_worker.mfa.probe import resource_name
    root=tmp_path/'components';root.mkdir()
    (root/'registry.json').write_text(json.dumps(dict(models=[dict(model_sha256='a'*64,dictionary_sha256='b'*64,name='mandarin.zip',dictionary_name='mandarin_pinyin_tab.dict')])))
    assert resource_name(root,'model',root/'resources/model'/('a'*64+'.zip'),'a'*64)=='mandarin.zip'
    assert resource_name(root,'dictionary',root/'resources/dictionary'/('b'*64+'.dict'),'b'*64)=='mandarin_pinyin_tab.dict'
    assert resource_name(root,'model',tmp_path/'alias.zip','a'*64)=='alias.zip'
    assert resource_name(root,'model',root/'resources/model'/('c'*64+'.zip'),'c'*64)=='c'*64+'.zip'
