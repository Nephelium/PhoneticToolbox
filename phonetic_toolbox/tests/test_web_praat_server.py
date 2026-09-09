# -*- coding: utf-8 -*-
"""嵌入式「语音标注对齐」网页服务的冒烟测试。"""
import io
import json
import pickle
import urllib.error
import urllib.request
import wave

import numpy as np
import pytest

from phonetic_toolbox.services import web_praat_server
from phonetic_toolbox.services.io.lip import read_lip_data


def _write_wav(path, duration_s=0.5, sr=16000):
    t = np.arange(int(duration_s * sr)) / sr
    data = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.5
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "ma1"
"""


@pytest.fixture()
def corpus(tmp_path):
    wav = tmp_path / "sample.wav"
    _write_wav(wav)
    (tmp_path / "sample.TextGrid").write_text(TEXTGRID, encoding="utf-8")
    (tmp_path / "sample.lab").write_text("ma1 ma2", encoding="utf-8")
    rec = {
        "metadata": {"lip_manual_offset": 0.0, "audio_first_frame_time": 100.0},
        "absolute_timestamps": [100.0, 100.033, 100.066, 100.1],
        "open": [0.1, 0.5, 0.4, 0.1],
        "outer_width": [1.0, 1.1, 1.05, 1.0],
    }
    with open(tmp_path / "audio_recording.pkl", "wb") as f:
        pickle.dump(rec, f)
    return tmp_path


@pytest.fixture()
def server_url(corpus, monkeypatch):
    monkeypatch.setattr(web_praat_server, "_load_last_root", lambda: str(corpus))
    monkeypatch.setattr(web_praat_server, "_save_last_root", lambda root: None)
    url = web_praat_server.ensure_server(folder_picker=lambda initial: "")
    yield url
    web_praat_server.shutdown_server()


@pytest.fixture()
def item_id(server_url):
    return _get_json(server_url + "api/list")["items"][0]["id"]


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read()


def _get_json(url):
    return json.loads(_get(url).decode("utf-8"))


def _post_json(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_static_index_served(server_url):
    body = _get(server_url).decode("utf-8")
    assert "语音标注对齐" in body


def test_static_app_js_served(server_url):
    body = _get(server_url + "app.js").decode("utf-8")
    assert "wordTierName" in body
    assert "lipDirty: false" in body
    assert "function markLipDirty()" in body
    assert "const lipOk = await saveLipAlignment" in body


def test_default_dict_served(server_url):
    body = _get(server_url + "default.dict").decode("utf-8")
    assert "a1" in body


def test_list_items(server_url):
    data = _get_json(server_url + "api/list")
    assert len(data["items"]) == 1
    assert data["items"][0]["wav"] == "sample.wav"


def test_item_detail(server_url, item_id):
    data = _get_json(server_url + "api/item?id=" + item_id)
    assert data["textgridName"] == "sample.TextGrid"
    assert "IntervalTier" in data["textgrid"]
    assert data["labWords"] == ["ma1", "ma2"]
    assert data["audioUrl"].startswith("/api/audio")


def test_audio_bytes(server_url, corpus, item_id):
    body = _get(server_url + "api/audio?id=" + item_id)
    assert body == (corpus / "sample.wav").read_bytes()


def test_lip_data(server_url, item_id):
    data = _get_json(server_url + "api/lip?id=" + item_id)
    assert data["available"] is True
    assert data["offset"] == 0.0
    assert len(data["times"]) == 4
    assert data["times"][0] == pytest.approx(0.0)
    assert "lipWidth" in data


def test_save_textgrid_and_lip(server_url, corpus, item_id):
    out = _post_json(server_url + "api/save", {"id": item_id, "textgrid": TEXTGRID, "suffix": "_webedit"})
    assert out["ok"] is True
    assert (corpus / "sample_webedit.TextGrid").exists()

    out = _post_json(server_url + "api/lip/save", {"id": item_id, "offset": 0.123})
    assert out["ok"] is True
    with open(corpus / "audio_recording.pkl", "rb") as f:
        rec = pickle.load(f)
    assert rec["metadata"]["lip_manual_offset"] == pytest.approx(0.123)


def test_lip_offset_write_failure_preserves_original(corpus, monkeypatch):
    rec_path = corpus / "audio_recording.pkl"
    original = rec_path.read_bytes()

    def fail_dump(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(web_praat_server.pickle, "dump", fail_dump)
    ok, message = web_praat_server.save_lip_offset(corpus / "sample.wav", 0.456)

    assert ok is False
    assert "simulated write failure" in message
    assert rec_path.read_bytes() == original
    assert list(corpus.glob(".audio_recording.pkl.*.tmp")) == []


def test_scan_new_root(server_url, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    _write_wav(other / "x.wav")
    (other / "x.TextGrid").write_text(TEXTGRID, encoding="utf-8")
    data = _get_json(server_url + "api/scan?path=" + urllib.parse.quote(str(other)))
    assert len(data["items"]) == 1
    data = _get_json(server_url + "api/list")
    assert data["items"][0]["wav"] == "x.wav"


def test_ensure_server_idempotent(server_url):
    assert web_praat_server.ensure_server() == server_url


@pytest.mark.parametrize("same_root", [False, True])
def test_stale_tab_cannot_save_after_scan(server_url, corpus, item_id, tmp_path, same_root):
    other = corpus if same_root else tmp_path / "other_corpus"
    other.mkdir(exist_ok=True)
    if not same_root:
        _write_wav(other / "sample.wav")
        (other / "sample.TextGrid").write_text(TEXTGRID, encoding="utf-8")
        (other / "audio_recording.pkl").write_bytes((corpus / "audio_recording.pkl").read_bytes())
    originals = {p: p.read_bytes() for directory in {corpus, other} for p in directory.glob("*") if p.is_file()}
    current = _get_json(server_url + "api/scan?path=" + urllib.parse.quote(str(other)))
    assert current["items"][0]["id"] != item_id
    for endpoint, payload in (
        ("api/save", {"id": item_id, "textgrid": "stale edit", "suffix": "_webedit"}),
        ("api/lip/save", {"id": item_id, "offset": 0.75}),
    ):
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(server_url + endpoint, payload)
        assert error.value.code == 404
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert not list(corpus.glob("*_webedit.TextGrid"))
    assert not list(other.glob("*_webedit.TextGrid"))
    result = _post_json(server_url + "api/save", {
        "id": current["items"][0]["id"], "textgrid": TEXTGRID, "suffix": "_webedit",
    })
    assert result["ok"]


@pytest.mark.parametrize("mode", ["legacy_absolute", "metadata_priority", "timestamps_fallback", "legacy_relative", "anchored_relative"])
def test_web_offset_round_trip_matches_parameter_export(corpus, mode):
    rec_path = corpus / "audio_recording.pkl"
    data = {"open": [1.0, 2.0, 3.0], "metadata": {}, "relative_times": [0.2, 0.3, 0.4]}
    if mode in {"legacy_absolute", "metadata_priority", "timestamps_fallback"}:
        data["absolute_timestamps"] = [100.2, 100.3, 100.4]
        companion_start = 99.0 if mode == "metadata_priority" else 100.0
        (corpus / "audio_recording_timestamps.pkl").write_bytes(pickle.dumps({"start_time": companion_start}))
        if mode != "timestamps_fallback":
            data["metadata"]["audio_first_frame_time"] = 100.0
    if mode == "anchored_relative":
        data["metadata"]["time_alignment_mode"] = "anchored_audio_start"
    rec_path.write_bytes(pickle.dumps(data))
    wav = corpus / "sample.wav"
    before = web_praat_server.read_lip_data_json(wav)
    expected_start = 0.0 if mode == "legacy_relative" else 0.2
    assert before["times"][0] == pytest.approx(expected_start)
    assert web_praat_server.save_lip_offset(wav, 0.05)[0]
    browser = web_praat_server.read_lip_data_json(wav)
    times = np.asarray(browser["times"]) + browser["offset"]
    exported = read_lip_data(str(rec_path), times)["LipOpen"]
    np.testing.assert_allclose(exported, browser["lipOpen"])
    assert times[0] == pytest.approx(expected_start + 0.05)
