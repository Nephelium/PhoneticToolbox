"""Behavioral checks for source-preserving multilingual website generation."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("website_i18n", ROOT / "scripts/build_website_i18n.py")
i18n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i18n)


class WebsiteLanguages(unittest.TestCase):
    def test_changed_copy_requires_translation_but_new_versions_reuse_it(self):
        catalog = i18n.Catalog()
        self.assertEqual(catalog.translate("PhoneticToolbox 9.8.7 更新小记", "test"), "PhoneticToolbox 9.8.7 · Release notes")
        unknown = "这个新功能尚未提供英文翻译"
        self.assertEqual(catalog.translate(unknown, "test"), unknown)
        self.assertIn(unknown, catalog.missing)

    def test_missing_numeric_placeholder_is_rejected(self):
        catalog = i18n.Catalog()
        catalog.messages["数值 {0} 毫秒"] = "A value without its number"
        with self.assertRaisesRegex(ValueError, "placeholders differ"):
            catalog.translate("数值 12.5 毫秒", "test")

    def test_source_citations_and_ascii_science_remain_original(self):
        source = '<p class="reference-citation">作者. 中文原文题名, 2026.</p><p>GL 32 · 10 ms · 60 dB · /ã/</p><button title="关闭截图">关闭</button>'
        parser = i18n.Translator(source, i18n.Catalog())
        parser.feed(source)
        result = parser.result()
        self.assertIn('lang="zh-CN">作者. 中文原文题名, 2026.', result)
        self.assertIn('GL 32 · 10 ms · 60 dB · /ã/', result)
        self.assertIn('title="Close screenshot">Close</button>', result)

    def test_download_and_media_identities_match_the_maintained_template(self):
        source = i18n.TEMPLATE.read_text("utf-8")
        downloads = lambda text: re.findall(r'href="(/releases/[^\"]+)"', text)
        media = lambda text: re.findall(r'(?:src|poster)="(?:\.\./)?(assets/[^\"]+\.(?:wav|mp4|jpg|png))"', text)
        for language in ("en", "ja"):
            with self.subTest(language=language):
                localized = (i18n.SITE / language / "index.html").read_text("utf-8")
                self.assertEqual(downloads(source), downloads(localized))
                self.assertEqual(media(source), media(localized))

    def test_japanese_kanji_are_not_mistaken_for_untranslated_source(self):
        catalog = i18n.Catalog("ja")
        self.assertEqual(catalog.translate("用于：录音、参数估计", "test"), "使用箇所：録音、パラメータ推定")
        self.assertEqual(catalog.missing, {})
        unknown = "这是新出现的尚未翻译的功能说明"
        self.assertEqual(catalog.translate(unknown, "test"), unknown)
        self.assertIn(unknown, catalog.missing)

    def test_each_language_menu_contains_all_languages_and_current_marker(self):
        for language in ("zh-CN", "en", "ja"):
            source = '<header><nav><a class="header-github nav-tip" href="https://github.com/">GitHub</a></nav></header><head></head>'
            result = i18n.language_controls(source, language)
            self.assertEqual(result.count('class="language-option"'), 3)
            self.assertEqual(result.count('aria-current="page"'), 1)
            self.assertIn('class="header-actions"', result)

    def test_javascript_positions_handle_crlf_and_astral_characters(self):
        source = "// 😀\r\nconst title = '原声 A';\r\nconst note = `正在播放：${title}`;\r\n"
        helper = ROOT / "scripts/website_js_strings.cjs"
        result = subprocess.run(["node", str(helper), "-"], input=source.encode("utf-8"), capture_output=True, check=True)
        normalized = source.replace("\r\n", "\n")
        rows = json.loads(result.stdout)
        literal = next(row for row in rows if row["text"] == "原声 A")
        self.assertEqual(normalized[literal["start"]:literal["end"]], "'原声 A'")


if __name__ == "__main__":
    unittest.main()
