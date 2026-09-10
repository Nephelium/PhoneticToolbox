import pytest
from phonetic_core.models.associations import Interval, Tier
from phonetic_core.segmentation import plan_segments, slice_parameter_table


def test_legacy_sample_boundary_channels_and_silence_policy():
    tiers=(Tier('音节',(Interval(0.,.001,'sil'),Interval(.00101,.00409,'ɑ̃'),Interval(.00409,.006,'<EPS>'))),)
    plan=plan_segments(tiers,'音节',1000,6)
    assert len(plan)==1
    assert (plan[0].index,plan[0].first,plan[0].last,plan[0].label)==(1,1,4,'ɑ̃')


@pytest.mark.parametrize('start,end',[(-.01,.01),(.01,.1),(.001,.00101),(float('nan'),.01)])
def test_bad_or_empty_segment_never_silently_slices(start,end):
    with pytest.raises(ValueError):plan_segments((Tier('x',(Interval(start,end,'a'),)),),'x',1000,20)


def test_missing_duplicate_empty_tier_and_count_budget():
    t=Tier('x',(Interval(0,.01,'a'),Interval(.01,.02,'b')))
    with pytest.raises(ValueError):plan_segments((t,),'absent',1000,20)
    with pytest.raises(ValueError):plan_segments((t,t),'x',1000,20)
    with pytest.raises(ValueError):plan_segments((t,),'x',1000,20,max_segments=1)
    assert plan_segments((Tier('x',(Interval(0,.01,' '),)),),'x',1000,20)==()


def test_parameter_times_use_actual_audio_slice_and_keep_original_axis():
    segment=plan_segments((Tier('x',(Interval(.0051,.0119,'a'),)),),'x',1000,20)[0]
    table={'columns':['Time_s','F0','text_词'],'kinds':['number','number','text'],
           'rows':[[0,120,'a'],[.005,None,'ɑ̃'],[.01,'+Infinity','=1+1'],[.015,130,'b']]}
    actual=slice_parameter_table(table,segment,1000)
    assert actual=={'columns':['Time_s','Source_Time_s','F0','text_词'],
                    'kinds':['number','number','number','text'],
                    'rows':[[0.,.005,None,'ɑ̃'],[.005,.01,'+Infinity','=1+1']]}
    assert table['rows'][1][0]==.005
    empty=plan_segments((Tier('x',(Interval(.011,.014,'a'),)),),'x',1000,20)[0]
    assert slice_parameter_table(table,empty,1000) is None
