"""Metadata-only admission for bounded M01. No decoder or numeric imports."""
MAX_SECONDS=1800.0

def validate_source(frames,rate,channels):
    if type(frames) is not int or type(rate) is not int or frames<=0 or not 8000<=rate<=192000 or not 1<=channels<=8:
        raise ValueError('m01_invalid_source')
    if frames>rate*1800:raise ValueError('m01_duration_limit')
