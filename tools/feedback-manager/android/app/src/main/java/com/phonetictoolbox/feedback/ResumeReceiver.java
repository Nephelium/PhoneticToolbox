package com.phonetictoolbox.feedback;
import android.content.*;
public final class ResumeReceiver extends BroadcastReceiver {
    @Override public void onReceive(Context c,Intent i){if((Intent.ACTION_BOOT_COMPLETED.equals(i.getAction())||Intent.ACTION_MY_PACKAGE_REPLACED.equals(i.getAction()))&&ReceivingService.enabled(c))ReceivingService.start(c);}
}
