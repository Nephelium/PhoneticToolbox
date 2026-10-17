package com.phonetictoolbox.feedback;

import android.Manifest;
import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.*;
import android.provider.Settings;
import android.webkit.*;
import android.view.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import org.json.*;

public final class MainActivity extends Activity {
    private static final String ORIGIN="https://appassets.androidplatform.net/";
    private WebView web;
    private ValueCallback<Uri[]> files;
    private final ThreadPoolExecutor pool=new ThreadPoolExecutor(1,1,0,TimeUnit.SECONDS,new ArrayBlockingQueue<>(4));
    private boolean resumed,registered;
    private final BroadcastReceiver events=new BroadcastReceiver(){@Override public void onReceive(Context c,Intent i){notification(i.getStringExtra("payload"));}};
    @Override public void onCreate(Bundle saved){
        super.onCreate(saved);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
        web=new WebView(this);android.widget.FrameLayout frame=new android.widget.FrameLayout(this);frame.addView(web,new android.widget.FrameLayout.LayoutParams(-1,-1));setContentView(frame);
        frame.setOnApplyWindowInsetsListener((v,insets)->{if(Build.VERSION.SDK_INT>=30){android.graphics.Insets bars=insets.getInsets(WindowInsets.Type.systemBars()|WindowInsets.Type.ime());v.setPadding(bars.left,bars.top,bars.right,bars.bottom);}else v.setPadding(insets.getSystemWindowInsetLeft(),insets.getSystemWindowInsetTop(),insets.getSystemWindowInsetRight(),insets.getSystemWindowInsetBottom());return insets;});
        web.setBackgroundColor(0xfff3f6fb);web.getSettings().setJavaScriptEnabled(true);web.getSettings().setDomStorageEnabled(true);web.getSettings().setAllowFileAccess(false);web.getSettings().setAllowContentAccess(true);web.getSettings().setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);web.getSettings().setSupportMultipleWindows(false);
        web.addJavascriptInterface(new Bridge(),"ptbFeedbackNative");
        web.setWebViewClient(new WebViewClient(){
            @Override public void onPageFinished(WebView v,String url){try{notification(new JSONObject().put("kind","connection").put("online",ReceivingService.connected).toString());}catch(JSONException ignored){}}
            @Override public boolean shouldOverrideUrlLoading(WebView v,WebResourceRequest r){return true;}
            @Override public WebResourceResponse shouldInterceptRequest(WebView v,WebResourceRequest r){
                Uri u=r.getUrl();String path=u.getPath();
                if(!"https".equals(u.getScheme())||!"appassets.androidplatform.net".equals(u.getHost())||u.getPort()!=-1||!"GET".equals(r.getMethod())||path==null||path.contains("..")||path.contains("\\"))return denied();
                String name=path.substring(1);if(!name.matches("[A-Za-z0-9_./-]+"))return denied();
                String mime=name.endsWith(".js")?"text/javascript":name.endsWith(".css")?"text/css":name.endsWith(".html")?"text/html":name.endsWith(".svg")?"image/svg+xml":name.endsWith(".png")?"image/png":(name.endsWith(".jpg")||name.endsWith(".jpeg"))?"image/jpeg":name.endsWith(".webp")?"image/webp":"application/octet-stream";
                try{Map<String,String> headers=new HashMap<>();headers.put("Content-Security-Policy","default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'");headers.put("X-Content-Type-Options","nosniff");return new WebResourceResponse(mime,"UTF-8",200,"OK",headers,getAssets().open(name));}catch(IOException e){return denied();}
            }
        });
        web.setWebChromeClient(new WebChromeClient(){
            @Override public boolean onShowFileChooser(WebView v,ValueCallback<Uri[]> callback,FileChooserParams params){if(files!=null)files.onReceiveValue(null);files=callback;Intent i=new Intent(Intent.ACTION_GET_CONTENT).setType("image/*").addCategory(Intent.CATEGORY_OPENABLE).putExtra(Intent.EXTRA_ALLOW_MULTIPLE,true);try{startActivityForResult(Intent.createChooser(i,"选择反馈图片"),2);}catch(ActivityNotFoundException e){files.onReceiveValue(null);files=null;}return true;}
            @Override public boolean onJsConfirm(WebView v,String url,String message,JsResult result){new AlertDialog.Builder(MainActivity.this).setMessage(message).setPositiveButton("确认",(d,w)->result.confirm()).setNegativeButton("取消",(d,w)->result.cancel()).setOnCancelListener(d->result.cancel()).show();return true;}
        });
        web.loadUrl(ORIGIN+"feedback-manager.html");
    }
    private WebResourceResponse denied(){return new WebResourceResponse("text/plain","UTF-8",403,"Forbidden",Collections.emptyMap(),new ByteArrayInputStream(new byte[0]));}
    private void notification(String payload){if(payload==null||payload.length()>10000)return;runOnUiThread(()->web.evaluateJavascript("window.ptbFeedbackAndroid?.notification("+JSONObject.quote(payload)+")",null));}
    private void reply(String id,JSONObject value){runOnUiThread(()->web.evaluateJavascript("window.ptbFeedbackAndroid?.ready("+JSONObject.quote(id)+","+JSONObject.quote(value.toString())+")",null));}
    private JSONObject receiving()throws Exception{
        boolean permission=getSystemService(NotificationManager.class).areNotificationsEnabled();if(Build.VERSION.SDK_INT>=33)permission&=checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)==PackageManager.PERMISSION_GRANTED;
        return new JSONObject().put("enabled",ReceivingService.enabled(this)).put("permission",permission).put("batteryExempt",getSystemService(PowerManager.class).isIgnoringBatteryOptimizations(getPackageName()));
    }
    private void openSettings(String kind){
        Intent intent;
        if(kind.equals("notifications"))intent=new Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(Settings.EXTRA_APP_PACKAGE,getPackageName());
        else if(kind.equals("battery"))intent=new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS);
        else if(kind.equals("autostart"))intent=new Intent().setComponent(new ComponentName("com.miui.securitycenter","com.miui.permcenter.autostart.AutoStartManagementActivity"));
        else intent=new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,Uri.parse("package:"+getPackageName()));
        try{startActivity(intent);}catch(Exception e){startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,Uri.parse("package:"+getPackageName())));}
    }
    private final class Bridge {
        @JavascriptInterface public void request(String id,String payload){
            if(id==null||!id.matches("[A-Za-z0-9_-]{1,96}"))return;
            if(payload==null||payload.length()>14000000){error(id,"图片过大，请减少图片后重试。");return;}
            try{pool.execute(()->execute(id,payload));}catch(RejectedExecutionException e){error(id,"操作繁忙，请稍后再试。");}
        }
    }
    private void error(String id,String message){try{reply(id,new JSONObject().put("ok",false).put("error",message));}catch(JSONException ignored){}}
    private void execute(String id,String payload){
        try{
            JSONObject request=new JSONObject(payload);if(request.length()!=2)throw new Exception();String op=request.getString("operation");JSONObject args=request.getJSONObject("args");
            Map<String,String> allowed=new HashMap<>();
            for(String key:new String[]{"status","health","conversations","listDevices","createPairing","receivingStatus"})allowed.put(key,"");
            allowed.put("pair","code,name");allowed.put("revokeDevice","deviceId");allowed.put("messages","owner,after");allowed.put("send","owner,text,images,requestId");allowed.put("read","owner,through");allowed.put("resolve","owner,status");allowed.put("retryMail","id");allowed.put("setReceiving","enabled");allowed.put("openSettings","kind");
            allowed.put("trashConversation","owner,through");allowed.put("restoreConversation","owner");allowed.put("webStatus","");allowed.put("webRequests","state,before");allowed.put("webDecision","request_id,granted_bytes,duration,note");allowed.put("webQuota","quota_bytes,owner_id");
            if(!allowed.containsKey(op))throw new Exception();Set<String> expected=new HashSet<>();if(!allowed.get(op).isEmpty())Collections.addAll(expected,allowed.get(op).split(","));Set<String> actual=new HashSet<>();args.keys().forEachRemaining(actual::add);if(!actual.equals(expected))throw new Exception();
            JSONObject result;
            if(op.equals("receivingStatus"))result=receiving();
            else if(op.equals("setReceiving")){
                boolean enabled=args.getBoolean("enabled");if(enabled&&!receiving().getBoolean("permission")){runOnUiThread(()->{if(Build.VERSION.SDK_INT>=33)requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},3);else openSettings("notifications");});result=receiving();}
                else{FeedbackApi.store(this).update(new JSONObject().put("receiving",enabled));runOnUiThread(()->{if(enabled)ReceivingService.start(this);else{stopService(new Intent(this,ReceivingService.class));if(resumed)ReceivingService.start(this);}});result=receiving();}
            }else if(op.equals("openSettings")){String kind=args.getString("kind");if(!Arrays.asList("notifications","battery","autostart","app").contains(kind))throw new Exception();runOnUiThread(()->openSettings(kind));result=new JSONObject().put("ok",true);}
            else {result=FeedbackApi.request(this,op,args);if(op.equals("pair"))runOnUiThread(()->{if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},3);else try{FeedbackApi.store(this).update(new JSONObject().put("receiving",true));ReceivingService.start(this);}catch(Exception ignored){}});}
            reply(id,new JSONObject().put("ok",true).put("value",result));
        }catch(Exception e){String m=e.getMessage();String message="无法完成请求，请检查网络后重试。未发送内容仍保留。";if(m!=null&&(m.startsWith("配对码")||m.startsWith("请输入")||m.startsWith("这台设备")||m.startsWith("发送过于")||m.startsWith("服务器暂时")||m.startsWith("网页版管理")||m.startsWith("这条申请")||m.startsWith("批准额度")||m.startsWith("服务器可用")||m.startsWith("服务器繁忙")))message=m;error(id,message);try{if(FeedbackApi.store(this).read().optString("token").isEmpty())notification(new JSONObject().put("kind","identity_lost").toString());}catch(Exception ignored){}}
    }
    @Override protected void onResume(){super.onResume();resumed=true;if(!registered){
        // The signature permission also protects releases before the exported flag.
        int receiverFlags=Build.VERSION.SDK_INT>=33?Context.RECEIVER_NOT_EXPORTED:0;
        registerReceiver(events,new IntentFilter(ReceivingService.EVENT),"com.phonetictoolbox.feedback.INTERNAL",null,receiverFlags);
        registered=true;}ReceivingService.start(this);}
    @Override protected void onStop(){resumed=false;if(registered){unregisterReceiver(events);registered=false;}if(!ReceivingService.enabled(this))stopService(new Intent(this,ReceivingService.class));super.onStop();}
    @Override public void onRequestPermissionsResult(int r,String[] p,int[] results){super.onRequestPermissionsResult(r,p,results);if(r==3&&results.length>0&&results[0]==PackageManager.PERMISSION_GRANTED){try{FeedbackApi.store(this).update(new JSONObject().put("receiving",true));ReceivingService.start(this);notification(new JSONObject().put("kind","receiving_changed").toString());}catch(Exception ignored){}}}
    @Override protected void onActivityResult(int r,int code,Intent data){super.onActivityResult(r,code,data);if(r==2&&files!=null){Uri[] selected=null;if(code==RESULT_OK&&data!=null){android.content.ClipData clips=data.getClipData();if(clips!=null){selected=new Uri[clips.getItemCount()];for(int i=0;i<selected.length;i++)selected[i]=clips.getItemAt(i).getUri();}else if(data.getData()!=null)selected=new Uri[]{data.getData()};}files.onReceiveValue(selected);files=null;}}
    @Override public void onBackPressed(){web.evaluateJavascript("(()=>{const b=document.querySelector('.back-button');if(b&&getComputedStyle(b).display!=='none'){b.click();return true;}return false;})()",v->{if(!"true".equals(v))moveTaskToBack(true);});}
    @Override protected void onDestroy(){if(files!=null)files.onReceiveValue(null);pool.shutdownNow();web.removeJavascriptInterface("ptbFeedbackNative");web.destroy();super.onDestroy();}
}
