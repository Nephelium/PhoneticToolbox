package com.phonetictoolbox.feedback;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;
import java.security.KeyStore;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;
import org.json.JSONObject;

final class SecureStore {
    private static final String ALIAS="ptb.feedback.manager.v1";
    private final SharedPreferences prefs;
    SecureStore(Context c){prefs=c.getSharedPreferences("feedback-private",Context.MODE_PRIVATE);}
    private SecretKey key() throws Exception {
        KeyStore ks=KeyStore.getInstance("AndroidKeyStore");ks.load(null);
        if(!ks.containsAlias(ALIAS)) {
            KeyGenerator gen=KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES,"AndroidKeyStore");
            gen.init(new KeyGenParameterSpec.Builder(ALIAS,KeyProperties.PURPOSE_ENCRYPT|KeyProperties.PURPOSE_DECRYPT).setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build());gen.generateKey();
        }
        return (SecretKey)ks.getKey(ALIAS,null);
    }
    synchronized JSONObject read() throws Exception {
        String data=prefs.getString("cipher","");if(data.isEmpty())return new JSONObject();
        Cipher cipher=Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE,key(),new GCMParameterSpec(128,Base64.decode(prefs.getString("iv",""),Base64.NO_WRAP)));
        return new JSONObject(new String(cipher.doFinal(Base64.decode(data,Base64.NO_WRAP)),java.nio.charset.StandardCharsets.UTF_8));
    }
    synchronized void update(JSONObject changes) throws Exception {
        JSONObject state=read();java.util.Iterator<String> keys=changes.keys();while(keys.hasNext()){String k=keys.next();state.put(k,changes.get(k));}
        Cipher cipher=Cipher.getInstance("AES/GCM/NoPadding");cipher.init(Cipher.ENCRYPT_MODE,key());
        if(!prefs.edit().putString("iv",Base64.encodeToString(cipher.getIV(),Base64.NO_WRAP)).putString("cipher",Base64.encodeToString(cipher.doFinal(state.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)),Base64.NO_WRAP)).commit())throw new Exception("无法保存登录信息");
    }
}
