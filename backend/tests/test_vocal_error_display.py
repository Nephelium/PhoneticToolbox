"""Owned rejection envelopes, without encoding media or changing scientific limits."""
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ptb_api.vocal import create_vocal_router
from ptb_worker.vocal_sessions import VocalSessions,VocalError
from ptb_worker.vocal_video import VideoSession
from ptb_worker.vocal_web import validate_web_operation,WEB_VIDEO_SECONDS
from phonetic_core.vocal_tract.trajectory import MAX_SEQUENCE_SECONDS


def test_dynamic_video_rejections_preserve_exact_limits_and_raw_errors(tmp_path):
    with pytest.raises(ValueError) as caught:VideoSession(tmp_path/'unused.mp4',{'width':640,'height':480,'fps':30,'duration':MAX_SEQUENCE_SECONDS+1})
    assert str(caught.value)==f'视频时长应为 0.05–{MAX_SEQUENCE_SECONDS:g} 秒'
    assert caught.value.display=={'key':'native.errors.vocalWeb.videoDurationRange','values':{'maximum':f'{MAX_SEQUENCE_SECONDS:g}'}}
    runtime=SimpleNamespace(live=SimpleNamespace(active=False))
    with pytest.raises(ValueError) as caught:validate_web_operation(runtime,'video/begin',{'duration':WEB_VIDEO_SECONDS+1})
    assert str(caught.value)==f'单次视频导出最多 {WEB_VIDEO_SECONDS:g} 秒，请缩短序列。'
    assert caught.value.display=={'key':'native.errors.vocalWeb.videoLimit','values':{'duration':f'{WEB_VIDEO_SECONDS:g}'}}
    assert list(tmp_path.iterdir())==[]


def test_real_worker_and_http_keep_detail_and_optional_display_without_rendering_video():
    resources=Path(__file__).resolve().parents[2]/'resources/vocal_tract/native'
    service=VocalSessions(resources)
    # Authentication is an explicit test double. This verifies only the owned
    # worker/error/HTTP envelope, not account or database acceptance.
    ctx=SimpleNamespace(mutation=lambda:{'id':'owned-user'},project=lambda session,project:None,store=SimpleNamespace(get_project=lambda owner,project:True),settings=SimpleNamespace(cookie='owned-token'))
    app=FastAPI();app.include_router(create_vocal_router(ctx,service));project=str(uuid4())
    try:
        with TestClient(app) as client:
            response=client.post('/api/v1/vocal/sessions',params={'project_id':project});assert response.status_code==200
            session=response.json()['id']
            response=client.post('/api/v1/vocal/sessions/'+session,params={'project_id':project},json={'op':'video/begin','body':{'width':640,'height':480,'fps':30,'duration':WEB_VIDEO_SECONDS+1}})
            assert response.status_code==422
            assert response.json()=={'detail':f'单次视频导出最多 {WEB_VIDEO_SECONDS:g} 秒，请缩短序列。','display':{'key':'native.errors.vocalWeb.videoLimit','values':{'duration':f'{WEB_VIDEO_SECONDS:g}'}}}
            response=client.post('/api/v1/vocal/sessions/'+session,params={'project_id':project},json={'op':'video/begin','body':{'width':640,'height':480,'fps':30,'duration':0}})
            assert response.status_code==422
            assert response.json()=={'detail':f'视频时长应为 0.05–{MAX_SEQUENCE_SECONDS:g} 秒','display':{'key':'native.errors.vocalWeb.videoDurationRange','values':{'maximum':f'{MAX_SEQUENCE_SECONDS:g}'}}}
    finally:service.close()


def test_unknown_worker_diagnostic_stays_literal():
    raw='ExternalCodec: 被试甲.mp4 E_RAW_17';error=VocalError(raw)
    assert error.code==str(error)==raw and error.status==422 and error.display is None
