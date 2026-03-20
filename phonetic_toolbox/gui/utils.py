import sounddevice as sd
import matplotlib.pyplot as plt

def apply_plot_theme(ax, is_dark=True):
    """
    Apply theme to a matplotlib axes object.
    
    Args:
        ax: The matplotlib axes to style.
        is_dark (bool): Whether to apply dark theme.
    """
    if is_dark:
        ax.set_facecolor('black')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
    else:
        ax.set_facecolor('white')
        ax.tick_params(colors='black', which='both')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

def play_audio_sd(snd_obj):
    """
    Play a Parselmouth Sound object using sounddevice.
    """
    if snd_obj is None: return
    try:
        fs = int(snd_obj.sampling_frequency)
        data = snd_obj.values.T 
        sd.play(data, fs)
    except Exception as e:
        print(f"播放错误: {e}")
        # Not raising here to avoid crashing UI if audio device fails
