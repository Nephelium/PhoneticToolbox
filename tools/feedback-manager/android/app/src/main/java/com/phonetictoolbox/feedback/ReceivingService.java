package com.phonetictoolbox.feedback;

import android.app.*;
import android.content.*;
import android.os.*;
import android.content.pm.ServiceInfo;
import java.io.InputStreamReader;
import javax.net.ssl.HttpsURLConnection;
import org.json.*;

public final class ReceivingService extends Service {
    static final String EVENT="com.phonetictoolbox.feedback.EVENT";
    static final String STOP="com.phonetictoolbox.feedback.STOP";
    private volatile boolean running;
    private volatile HttpsURLConnection connection;
    private Thread worker;
    static volatile boolean connected;
    private static final String CHANNEL="new-feedback",ONGOING="receiving";
    static boolean enabled(Context c){try{return FeedbackApi.store(c).read().optBoolean("receiving",false)&&!FeedbackApi.store(c).read().optString("token").isEmpty();}catch(Exception e){return false;}}
    static void start(Context c){try{if(FeedbackApi.store(c).read().optString("token").isEmpty())return;if(enabled(c))c.startForegroundService(new Intent(c,ReceivingService.class));else c.startService(new Intent(c,ReceivingService.class));}catch(Exception ignored){}}
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        if(intent!=null&&STOP.equals(intent.getAction())){try{FeedbackApi.store(this).update(new JSONObject().put("receiving",false));}catch(Exception ignored){}stopSelf();return START_NOT_STICKY;}
        NotificationManager manager=getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL,"新的用户反馈",NotificationManager.IMPORTANCE_HIGH));
        manager.createNotificationChannel(new NotificationChannel(ONGOING,"后台接收状态",NotificationManager.IMPORTANCE_LOW));
        if(enabled(this)){
            PendingIntent stop=PendingIntent.getService(this,1,new Intent(this,ReceivingService.class).setAction(STOP),PendingIntent.FLAG_IMMUTABLE|PendingIntent.FLAG_UPDATE_CURRENT);
            Notification n=new Notification.Builder(this,ONGOING).setSmallIcon(com.phonetictoolbox.feedback.R.drawable.ic_inbox).setContentTitle("反馈收件箱").setContentText("正在接收新留言").setContentIntent(open()).setOngoing(true).addAction(new Notification.Action.Builder(null,"停止接收",stop).build()).build();
            if(Build.VERSION.SDK_INT>=34)startForeground(1,n,ServiceInfo.FOREGROUND_SERVICE_TYPE_REMOTE_MESSAGING);else startForeground(1,n);
        }else stopForeground(STOP_FOREGROUND_REMOVE);
        if(worker==null){running=true;worker=new Thread(this::listen,"feedback-events");worker.start();}
        return enabled(this)?START_STICKY:START_NOT_STICKY;
    }
    private PendingIntent open(){return PendingIntent.getActivity(this,0,new Intent(this,MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),PendingIntent.FLAG_IMMUTABLE|PendingIntent.FLAG_UPDATE_CURRENT);}
    private void emit(JSONObject event){sendBroadcast(new Intent(EVENT).setPackage(getPackageName()).putExtra("payload",event.toString()));}
    private void online(boolean value){connected=value;try{emit(new JSONObject().put("kind","connection").put("online",value));}catch(JSONException ignored){}}
    private boolean unread(String owner)throws Exception{
        JSONArray rows=FeedbackApi.call(this,"/admin/conversations",null,true).getJSONArray("conversations");
        for(int i=0;i<rows.length();i++){JSONObject row=rows.getJSONObject(i);if((owner.isEmpty()||owner.equals(row.getString("id")))&&row.getInt("unread")>0)return true;}return false;
    }
    private void listen(){
        while(running){
            try{
                JSONObject state=FeedbackApi.store(this).read();String token=state.optString("token");if(token.isEmpty()){stopSelf();break;}long cursor=state.optLong("cursor",0);
                connection=FeedbackApi.connection("/admin/events?after="+cursor,token);connection.setRequestProperty("Accept","text/event-stream");
                int status=connection.getResponseCode();
                if(status==401){FeedbackApi.store(this).update(new JSONObject().put("token","").put("receiving",false));emit(new JSONObject().put("kind","identity_lost"));stopSelf();break;}
                if(status!=200)throw new Exception();online(true);
                try(InputStreamReader reader=new InputStreamReader(connection.getInputStream(),java.nio.charset.StandardCharsets.UTF_8)){
                    while(running){StringBuilder line=new StringBuilder();int ch;while((ch=reader.read())!=-1&&ch!='\n'){if(line.length()>=8192)throw new Exception();line.append((char)ch);}if(ch==-1)break;
                        if(!line.toString().startsWith("data: "))continue;
                        JSONObject event=new JSONObject(line.substring(6));long id=event.getLong("id");if(id<=cursor)continue;
                        String kind=event.optString("kind");
                        if(enabled(this)&&(kind.equals("new_feedback")||kind.equals("daily_feedback"))&&unread(event.optString("owner"))){
                            int notificationId=100+Math.abs(event.optString("owner","daily").hashCode()%100000);
                            Notification n=new Notification.Builder(this,CHANNEL).setSmallIcon(R.drawable.ic_inbox).setContentTitle("新的用户反馈").setContentText(kind.equals("daily_feedback")?"每日检查发现待回复的留言":"收到新留言，点开查看与回复").setContentIntent(open()).setAutoCancel(true).setCategory(Notification.CATEGORY_MESSAGE).build();
                            getSystemService(NotificationManager.class).notify(notificationId,n);
                        }
                        emit(event);FeedbackApi.store(this).update(new JSONObject().put("cursor",id));cursor=id;
                    }
                }
            }catch(Exception ignored){if(running)online(false);}
            finally{HttpsURLConnection active=connection;connection=null;if(active!=null)active.disconnect();}
            if(running)try{Thread.sleep(3000);}catch(InterruptedException e){break;}
        }
    }
    @Override public void onDestroy(){running=false;connected=false;if(connection!=null)connection.disconnect();if(worker!=null)worker.interrupt();super.onDestroy();}
    @Override public IBinder onBind(Intent intent){return null;}
}
