from __future__ import annotations
import math,numpy as np
from scipy.signal import resample_poly
TARGET_FS=250;EPS=1e-6
BANDS=((1,4),(4,8),(8,13),(13,30),(30,55),(65,100))
def resample_channel_once(x,fs,target_fs=TARGET_FS):
 x=np.asarray(x,np.float64);g=math.gcd(int(round(fs)),int(target_fs));return resample_poly(x,int(target_fs)//g,int(round(fs))//g).astype(np.float32) if fs!=target_fs else x.astype(np.float32)
def rms(x):return float(np.sqrt(np.mean(np.square(x))+EPS))
def line_length(x):return float(np.mean(np.abs(np.diff(x))))
def spectral(x,fs=TARGET_FS):
 p=np.abs(np.fft.rfft(x))**2+EPS;f=np.fft.rfftfreq(len(x),1/fs);bp=np.array([p[(f>=a)&(f<b)].mean() if np.any((f>=a)&(f<b)) else EPS for a,b in BANDS]);q=p/p.sum();return bp,float(-(q*np.log(q)).sum()/np.log(len(q)))
def reference(pre,fs=TARGET_FS):
 med=float(np.median(pre));mad=float(np.median(np.abs(pre-med)));bp,ent=spectral(pre,fs);return {'median':med,'scale':max(1.4826*mad,EPS),'rms':rms(pre),'ll':line_length(pre),'bp':bp,'entropy':ent,'zcr':float(np.mean(np.diff(np.signbit(pre))))}
def normalize_window(x,ref,fs=TARGET_FS):
 finite=np.isfinite(x);ff=float(finite.mean())
 if ff<.95:return None,None,'finite_fraction'
 x=np.nan_to_num(x,nan=ref['median'],posinf=ref['median'],neginf=ref['median']);z=(x-ref['median'])/ref['scale'];clip=float(np.mean(np.abs(z)>8));z=np.clip(z,-8,8)
 if np.std(z)<1e-5:return None,None,'near_constant'
 bp,ent=spectral(x,fs);side=np.r_[np.log((rms(x)+EPS)/(ref['rms']+EPS)),np.log((line_length(x)+EPS)/(ref['ll']+EPS)),np.log((bp+EPS)/(ref['bp']+EPS)),ent-ref['entropy'],float(np.mean(np.diff(np.signbit(x))))-ref['zcr'],clip,ff].astype(np.float32)
 return z.astype(np.float32),side,''
