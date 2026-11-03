"""Focused source/translation invariants; synthetic text never ships as a manual."""
import copy
import unittest

from scripts.manual.core import ManualError
from scripts.manual.localization import Unit, apply_catalog, chapter_units, inline_source, translated_inline


class LocalizationTests(unittest.TestCase):
    def test_whole_sentence_can_reorder_formatted_spans_and_reference(self):
        source = [
            {'type': 'text', 'text': '点击'},
            {'type': 'text', 'text': '打开', 'marks': [{'type': 'bold'}]},
            {'type': 'text', 'text': '，参见'},
            {'type': 'crossReference', 'attrs': {'chapterId': 'm01', 'targetId': 'input'}},
        ]
        text, markers = inline_source(source)
        self.assertEqual(text, '点击<s id="0">打开</s>，参见<x id="1"/>')
        result = translated_inline('See <x id="1"/>, then click <s id="0">Open</s>.', markers)
        self.assertEqual(result[1], source[3])
        self.assertEqual(result[3]['marks'], source[1]['marks'])
        self.assertEqual(result[3]['text'], 'Open')
        self.assertEqual(source[1]['text'], '打开')

    def test_missing_duplicate_new_or_active_inline_objects_fail(self):
        _, markers = inline_source([{'type': 'text', 'text': '打开', 'marks': [{'type': 'link', 'attrs': {'href': 'https://example.org'}}]}])
        for value in ['Open', '<s id="0">Open</s><s id="0">Open</s>', '<script>alert(1)</script>', '<s id="0" href="javascript:x">Open</s>', '<!DOCTYPE x><s id="0">Open</s>']:
            with self.subTest(value=value), self.assertRaises(ManualError):
                translated_inline(value, markers)

    def test_inserted_paragraph_does_not_invalidate_unchanged_unit(self):
        chapter = {'title': '标题', 'body': {'type': 'doc', 'content': [{'type': 'paragraph', 'attrs': {'id': 'paragraph-a'}, 'content': [{'type': 'text', 'text': '基频为 100 Hz。'}]}]}}
        original = chapter_units(chapter)[1]
        changed = copy.deepcopy(chapter)
        changed['body']['content'].insert(0, {'type': 'paragraph', 'content': [{'type': 'text', 'text': '新增段落'}]})
        same = chapter_units(changed)[2]
        self.assertEqual(original.key, same.key)
        changed['body']['content'][1]['content'][0]['text'] = '基频为 120 Hz。'
        self.assertNotEqual(original.key, chapter_units(changed)[2].key)

    def test_old_translation_does_not_cover_changed_source_or_values(self):
        owner = {'text': '基频为 100 Hz。'}
        unit = Unit(owner, 'text', owner['text'])
        catalog = {unit.key: {'source': unit.source, 'translation': 'Fundamental frequency is 120 Hz.'}}
        self.assertEqual(len(apply_catalog([unit], catalog)['invalid']), 1)
        with self.assertRaises(ManualError):
            apply_catalog([unit], catalog, apply=True)
        catalog[unit.key]['translation'] = 'Fundamental frequency is 100 Hz.'
        self.assertEqual(apply_catalog([unit], catalog)['translated'], 1)
        changed = Unit(owner, 'text', '基频为 110 Hz。')
        self.assertEqual(len(apply_catalog([changed], catalog)['missing']), 1)

    def test_code_and_ipa_marks_and_code_blocks_are_preserved(self):
        markers = [{'type': 'text', 'text': 'F0', 'marks': [{'type': 'code'}]}]
        with self.assertRaises(ManualError):
            translated_inline('<s id="0">F1</s>', markers)
        self.assertEqual(translated_inline('<s id="0">F0</s>', markers), markers)
        chapter = {'title': 'Example', 'body': {'type': 'doc', 'content': [{'type': 'codeBlock', 'content': [{'type': 'text', 'text': '# 保留代码'}]}]}}
        self.assertEqual(chapter_units(chapter), [])

    def test_pixel_counts_allow_equivalent_language_units_but_not_changed_limits(self):
        unit = Unit({}, 'caption', '最多 4000 万像素，文件不超过 20 MB。')
        catalog = {unit.key: {'source': unit.source, 'translation': 'At most 40 million pixels and 20 MB.'}}
        self.assertEqual(apply_catalog([unit], catalog)['translated'], 1)
        catalog[unit.key]['translation'] = 'At most 16 million pixels and 20 MB.'
        self.assertEqual(len(apply_catalog([unit], catalog)['invalid']), 1)

    def test_large_unit_equivalence_is_exact_across_chinese_english_and_japanese(self):
        for source, translation in [('总 1.8 亿单元格', '180 million cells in total'),
                                    ('总 1.8 亿单元格', '合計1.8億セル'),
                                    ('最多 10 亿个值', 'At most 1 billion values')]:
            unit = Unit({}, 'caption', source)
            catalog = {unit.key: {'source': source, 'translation': translation}}
            self.assertEqual(apply_catalog([unit], catalog)['translated'], 1)
            catalog[unit.key]['translation'] = translation.replace('180', '181').replace('1.8', '1.9').replace('1 billion', '2 billion')
            self.assertEqual(len(apply_catalog([unit], catalog)['invalid']), 1)


if __name__ == '__main__':
    unittest.main()
