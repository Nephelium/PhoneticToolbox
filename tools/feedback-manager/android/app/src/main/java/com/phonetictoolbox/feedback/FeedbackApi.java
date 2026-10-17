package com.phonetictoolbox.feedback;

import android.content.Context;
import java.net.URL;
import javax.net.ssl.HttpsURLConnection;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

final class FeedbackApi {
    static final String BASE="https://www.phonetictoolbox.com/api/v1/feedback";
    private static SecureStore store;
    static synchronized SecureStore store(Context c){if(store==null)store=new SecureStore(c.getApplicationContext());return store;}
    static HttpsURLConnection connection(String path,String token)throws Exception{
        if(!path.matches("/[A-Za-z0-9_/?=&.-]+")||path.contains(".."))throw new Exception("请求无效");
        HttpsURLConnection c=(HttpsURLConnection)new URL(BASE+path).openConnection();c.setInstanceFollowRedirects(false);c.setConnectTimeout(15000);c.setReadTimeout(25000);c.setRequestProperty("Accept","application/json");c.setRequestProperty("User-Agent","PhoneticToolbox-Feedback-Android/1");if(!token.isEmpty())c.setRequestProperty("Authorization","Bearer "+token);return c;
    }
    static JSONObject call(Context context,String path,JSONObject body,boolean auth)throws Exception {
        HttpsURLConnection c=connection(path,auth?store(context).read().optString("token"):"");
        try {
            if(body!=null){byte[] bytes=body.toString().getBytes(StandardCharsets.UTF_8);if(bytes.length>14000000)throw new Exception("图片过大");c.setRequestMethod("POST");c.setDoOutput(true);c.setRequestProperty("Content-Type","application/json");c.setFixedLengthStreamingMode(bytes.length);try(java.io.OutputStream out=c.getOutputStream()){out.write(bytes);}}
            int status=c.getResponseCode();if(status!=200){
                String detail="";InputStream error=c.getErrorStream();if(error!=null){try(InputStream in=error){detail=new JSONObject(read(in,4096)).optString("detail");}catch(Exception ignored){}}
                if(status==401&&auth){store(context).update(new JSONObject().put("token","").put("receiving",false));throw new Exception("这台设备的授权已失效，请重新配对。");}
                if((status==404||status==405)&&path.startsWith("/admin/conversations/")&&(path.endsWith("/trash")||path.endsWith("/restore"))&&(detail.isEmpty()||detail.equals("Not Found")||detail.equals("Method Not Allowed")))throw new Exception("回收站尚未在此服务器启用，请先更新反馈服务。");
                if(path.startsWith("/admin/web/")&&(status==404||status==502||status==503||status==504))throw new Exception("网页版管理暂未连接，反馈功能仍可使用。");
                if(detail.equals("pairing_invalid"))throw new Exception("配对码无效、已使用或已过期，请在电脑重新生成。");
                if(detail.equals("pairing_rate_limited"))throw new Exception("配对码生成过于频繁，请稍后再试。");
                if(detail.equals("message_rate_limited"))throw new Exception("发送过于频繁，请稍后再试。");
                if(detail.equals("web_operations_unavailable"))throw new Exception("网页版管理暂未连接，反馈功能仍可使用。");
                if(detail.equals("conversation_changed"))throw new Exception("会话有新消息，请刷新后重新确认。");
                if(detail.equals("conversation_trashed"))throw new Exception("会话已移入回收站，请恢复后再回复。");
                if(detail.equals("conversation_mail_busy"))throw new Exception("此会话正在发送邮件，请稍后再移入回收站。");
                if(detail.equals("conversation_unavailable"))throw new Exception("会话已不存在，请刷新会话列表。");
                if(detail.equals("capacity_request_closed"))throw new Exception("这条申请已处理，请刷新查看结果。");
                if(detail.equals("capacity_not_increase"))throw new Exception("批准额度应高于用户当前额度。");
                if(detail.equals("disk_space_low"))throw new Exception("服务器可用空间不足，请先清理或降低批准额度。");
                if(detail.equals("web_server_busy"))throw new Exception("服务器繁忙，请稍后重试。");
                throw new Exception("服务器暂时无法完成请求，请稍后重试。");
            }
            try(InputStream in=c.getInputStream()){return new JSONObject(read(in,18000000));}
        }finally{c.disconnect();}
    }
    static String read(InputStream in,int max)throws Exception{ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1){if(out.size()+n>max)throw new Exception("响应过大");out.write(b,0,n);}return out.toString("UTF-8");}
    static JSONObject request(Context c,String op,JSONObject args)throws Exception{
        SecureStore s=store(c);JSONObject state=s.read();
        switch(op){
            case "status":return new JSONObject().put("connected",!state.optString("token").isEmpty()).put("platform","android").put("baseUrl",BASE).put("admin",true).put("verified",false).put("owner","").put("email","");
            case "pair":
                String code=args.getString("code"),name=args.getString("name");if(!code.matches("[0-9]{8}")||name.length()>40)throw new Exception("请输入 8 位配对码。");
                JSONObject pair=call(c,"/admin/devices/pairing/complete",args,false);s.update(new JSONObject().put("token",pair.getString("token")).put("deviceId",pair.getString("deviceId")).put("cursor",0).put("receiving",false));return request(c,"status",new JSONObject());
            case "health":return call(c,"/health",null,false);
            case "conversations":return call(c,"/admin/conversations?includeTrash=true",null,true);
            case "webStatus":return call(c,"/admin/web/status",null,true);
            case "webRequests":
                String stateFilter=args.optString("state","pending"),before=args.optString("before","");
                if(!stateFilter.matches("pending|approved|declined"))throw new Exception("申请筛选无效。");
                if(!before.isEmpty())before=java.util.UUID.fromString(before).toString();
                return call(c,"/admin/web/requests?state="+stateFilter+(before.isEmpty()?"":"&before="+before),null,true);
            case "webDecision":return call(c,"/admin/web/decision",args,true);
            case "webQuota":return call(c,"/admin/web/quota",args,true);
            case "listDevices":return call(c,"/admin/devices",null,true);
            case "createPairing":return call(c,"/admin/devices/pairing",new JSONObject(),true);
            case "revokeDevice":String id=args.getString("deviceId");if(!id.matches("[A-Za-z0-9_-]{24}"))throw new Exception("设备无效");return call(c,"/admin/devices/"+id+"/revoke",new JSONObject(),true);
            case "retryMail":long mid=args.getLong("id");if(mid<1)throw new Exception("消息无效");return call(c,"/admin/mail/"+mid+"/retry",new JSONObject(),true);
        }
        String owner=args.getString("owner");if(!owner.matches("[A-Za-z0-9_-]{8,80}"))throw new Exception("会话无效");String prefix="/admin/conversations/"+owner;
        switch(op){
            case "messages":long after=args.getLong("after");if(after<0)throw new Exception("会话位置无效");return call(c,prefix+"?after="+after,null,true);
            case "read":long through=args.getLong("through");if(through<0)throw new Exception("会话位置无效");return call(c,prefix+"/read",new JSONObject().put("through",through),true);
            case "resolve":String status=args.getString("status");if(!status.equals("open")&&!status.equals("resolved"))throw new Exception("状态无效");return call(c,prefix+"/status",new JSONObject().put("status",status),true);
            case "trashConversation":long last=args.getLong("through");if(last<1)throw new Exception("会话位置无效");return call(c,prefix+"/trash",new JSONObject().put("through",last),true);
            case "restoreConversation":return call(c,prefix+"/restore",new JSONObject(),true);
            case "send":return call(c,prefix+"/messages",new JSONObject().put("text",args.getString("text")).put("images",args.getJSONArray("images")).put("requestId",args.getString("requestId")),true);
            default:throw new Exception("操作无效");
        }
    }
}
