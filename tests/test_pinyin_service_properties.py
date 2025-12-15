"""
Property-based tests for PinyinService in Whisper Transcription Tool.

Tests pinyin conversion properties including ü representation and tone number format.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_transcription.pinyin_service import PinyinService


# ============================================================================
# Strategies for generating test data
# ============================================================================

# Chinese characters that contain the ü sound (nü, lü, etc.)
# These characters use ü in their pinyin: 女(nǚ), 绿(lǜ), 旅(lǚ), 律(lǜ), 虑(lǜ), 驴(lǘ)
u_umlaut_chars = st.sampled_from(['女', '绿', '旅', '律', '虑', '驴', '吕', '屡', '侣', '缕'])

# Common Chinese characters (a mix of different tones)
common_chinese_chars = st.sampled_from([
    '你', '好', '我', '是', '的', '一', '不', '在', '有', '这',
    '中', '国', '人', '大', '来', '上', '为', '和', '地', '到',
    '说', '时', '要', '就', '出', '会', '可', '也', '你', '对',
    '生', '能', '子', '那', '得', '于', '着', '下', '自', '之',
    '年', '过', '发', '后', '作', '里', '用', '道', '行', '所',
    '然', '家', '种', '事', '成', '方', '多', '经', '么', '去',
    '法', '学', '如', '都', '同', '现', '当', '没', '动', '面',
    '起', '看', '定', '天', '分', '还', '进', '好', '小', '部'
])

# Strategy for generating Chinese text with ü characters
chinese_text_with_u_umlaut = st.lists(
    st.one_of(u_umlaut_chars, common_chinese_chars),
    min_size=1,
    max_size=20
).map(lambda chars: ''.join(chars))

# Strategy for generating any Chinese text
chinese_text = st.lists(
    common_chinese_chars,
    min_size=1,
    max_size=20
).map(lambda chars: ''.join(chars))

# Strategy for mixed text (Chinese and non-Chinese)
non_chinese_text = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'S'),  # Letters, Numbers, Punctuation, Symbols
        blacklist_categories=('Lo',),  # Exclude "Other Letters" which includes CJK
        max_codepoint=0x7F  # ASCII only for simplicity
    ),
    min_size=0,
    max_size=10
)


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(text=chinese_text_with_u_umlaut)
@settings(max_examples=100)
def test_pinyin_u_umlaut_representation(text: str):
    """
    **Feature: whisper-transcription, Property 5: Pinyin ü Representation**
    **Validates: Requirements 4.2**
    
    For any Chinese text containing characters with the ü sound (如：女、绿、旅),
    the pinyin output should use "v" instead of "ü".
    """
    assume(len(text.strip()) > 0)
    
    result = PinyinService.convert_to_pinyin(text)
    
    # Property: The result should not contain ü character
    assert 'ü' not in result, \
        f"Pinyin output should use 'v' instead of 'ü'. Input: {text}, Output: {result}"


@given(text=chinese_text)
@settings(max_examples=100)
def test_pinyin_tone_number_format(text: str):
    """
    **Feature: whisper-transcription, Property 6: Pinyin Tone Number Format**
    **Validates: Requirements 4.3**
    
    For any Chinese text, every pinyin syllable in the output should end
    with a tone number from 1 to 5.
    """
    assume(len(text.strip()) > 0)
    
    result = PinyinService.convert_to_pinyin(text)
    
    # Split result into syllables
    syllables = result.split()
    
    # Property: Each syllable should end with a tone number (1-5)
    for syllable in syllables:
        if syllable:  # Skip empty strings
            # Check if the last character is a tone number
            assert syllable[-1] in '12345', \
                f"Pinyin syllable '{syllable}' should end with tone number (1-5). Full output: {result}"


# ============================================================================
# Unit Tests
# ============================================================================

def test_convert_to_pinyin_empty_string():
    """Test that empty string returns empty string."""
    result = PinyinService.convert_to_pinyin("")
    assert result == ""


def test_convert_to_pinyin_basic():
    """Test basic pinyin conversion."""
    result = PinyinService.convert_to_pinyin("你好")
    assert result == "ni3 hao3"


def test_convert_to_pinyin_u_umlaut():
    """Test that ü is converted to v."""
    result = PinyinService.convert_to_pinyin("女生")
    assert "nv3" in result
    assert "ü" not in result


def test_convert_to_pinyin_mixed_text():
    """Test mixed Chinese and non-Chinese text."""
    result = PinyinService.convert_to_pinyin("Hello世界")
    assert "Hello" in result
    assert "shi4" in result
    assert "jie4" in result


def test_convert_to_pinyin_tone_numbers():
    """Test that tone numbers are present."""
    result = PinyinService.convert_to_pinyin("中国")
    # Each syllable should end with a number
    syllables = result.split()
    for syllable in syllables:
        assert syllable[-1] in '12345'


def test_replace_u_with_v():
    """Test the _replace_u_with_v helper method."""
    assert PinyinService._replace_u_with_v("nü3") == "nv3"
    assert PinyinService._replace_u_with_v("lü4") == "lv4"
    assert PinyinService._replace_u_with_v("ni3") == "ni3"  # No change needed
