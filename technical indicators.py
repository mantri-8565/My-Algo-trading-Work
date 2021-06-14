import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from stocktrends import Renko


ohlcv=pd.DataFrame()
start_dt=dt.date.today()-dt.timedelta(360)
end_dt=dt.date.today()
stock="MSFT"
ohlcv= yf.download(stock,start_dt,end_dt)

# MACD IMPLEMENTATION
def MACD(DF,a,b,c):
    df=DF.copy()
    df['high line']=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df['low line']=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df['MACD']=df['low line']-df['high line']
    df['Signal']=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    df=df.iloc[:,[-1,-2,-6]]    
    return df



# ATR IMPLEMENTATION
def ATR(DF, r):
    df=DF.copy()
    df['hl']=abs(df['High']-df['Low'])
    df['hc'] =   abs(df['High']-df['Adj Close'].shift(1))
    df['lc'] =   abs(df['Low']-df['Adj Close'].shift(1))
    df['tr']=df[['hl','hc','lc']].max(axis=1,skipna=False)                  
    df['atr']=df['tr'].rolling(window=r).mean()
    df.drop(['hl','hc','lc'],axis=1)
    return df



#BollingerBands IMPLEMENTATION
def BollingerBands(DF):
    df=DF.copy()
    df['20MA']=df['Adj Close'].rolling(20).mean()
    df['1BB']=df['20MA']+2*(df['Adj Close'].rolling(20).std())
    df['2BB']=df['20MA']-2*(df['Adj Close'].rolling(20).std())
    df.dropna(inplace=True)
    return df
# bb=BollingerBands(ohlcv)
# bb[['Adj Close','20MA','1BB','2BB']].plot()

#RSI Implementation
def RSI(DF,p):
    
    df=DF.copy()
    df["gain"]=np.where(df["Adj Close"]>df["Adj Close"].shift(1),df["Adj Close"]-df["Adj Close"].shift(1),0)
    df["loss"]=np.where(df["Adj Close"]<df["Adj Close"].shift(1),df["Adj Close"].shift(1)-df["Adj Close"],0)
    Avgain=[]
    Avloss=[]
    gain=df["gain"].tolist()
    loss=df["loss"].tolist()
    for i in range(p):
        Avgain.append(np.NaN)
        Avloss.append(np.NaN)
    Avgain.append(sum(gain[:15])/p)
    Avloss.append(sum(loss[:15])/p)
    for i in range (p+1,(len(df))):
        Avgain.append(((Avgain[-1]*(p-1))+gain[i])/p)
        Avloss.append(((Avloss[-1]*(p-1))+loss[i])/p)
    df['Avgain']=np.array(Avgain)
    df['Avloss']=np.array(Avloss)
    df['RS']=df['Avgain']/df['Avloss']
    df['RSI']=100-(100/(1+df['RS']))
    return df 

# rsi=RSI(ohlcv, 14)   
 




# ADX IMPLEMENTATION
def ADX(DF):
    df=DF.copy()
    df['zero']=np.array(0)
    df['hl']=abs(df['High']-df['Low'])
    df['hc'] =   abs(df['High']-df['Adj Close'].shift(1))
    df['lc'] =   abs(df['Low']-df['Adj Close'].shift(1))
    df['tr']=df[['hl','hc','lc']].max(axis=1,skipna=False)    
    df['hh']=df["High"]-df['High'].shift(1)
    df['ll']=df["Low"]-df['Low'].shift(1)
    df['dm+']=np.where(df['hh']>df["ll"],df[['hh','zero']].max(axis=1,skipna=False),0)
    df['dm-']=np.where(df['hh']<df["ll"],df[['ll','zero']].max(axis=1,skipna=False),0)
    tr=[]
    dmp=[]
    dmm=[]
    dmp14=[]
    dmm14=[]
    tr14=[]
    tr=df['tr'].tolist()
    dmp=df['dm+'].tolist()
    dmm=df['dm-'].tolist()
    for i in range(len(df)):
        if i<14:
            tr14.append(np.NaN)
            dmp14.append(np.NaN)
            dmm14.append(np.NaN)
        
        elif i==14:
            tr14.append(sum(tr[1:15]))
            dmp14.append(sum(dmp[1:15]))
            dmm14.append(sum(dmm[1:15]))
        else:
            tr14.append(tr14[-1]-(tr14[-1]/14)+tr[i])
            dmp14.append(dmp14[-1]-(dmp14[-1]/14)+dmp[i])
            dmm14.append(dmm14[-1]-(dmm14[-1]/14)+dmm[i])
    df['tr14']=np.array(tr14)
    df['dm+14']=np.array(dmp14)
    df['dm-14']=np.array(dmm14)
    df['di+']=100*(df['dm+14']/df['tr14'])
    df['di-']=100*(df['dm-14']/df['tr14'])
    df['disum']=df['di+']+ df['di-']
    df['didiff']=abs(df['di+']- df['di-'])
    df['dx']=100*((df['didiff'])/df['disum'])
    dx=[]
    dx=df['dx'].tolist()
    adx=[]
    for i in range(len(df)):
        if i < 27:
            adx.append(np.NaN)
        elif i==27:
            adx.append(df['dx'].rolling(14).mean()[27])
        else:
            adx.append(((adx[-1]*13)+dx[i])/14)
    df['ADX']=np.array(adx)
    df.drop(['Volume', 'zero', 'hl',
       'hc', 'lc', 'tr', 'hh', 'll', 'dm+', 'dm-', 'tr14', 'dm+14', 'dm-14',
       'di+', 'di-', 'disum', 'didiff', 'dx'],axis=1,inplace=True)
    return df



# adx=    ADX(ohlcv)


 #Opening Balanace volume
 
def OBV (DF):
     df=DF.copy()
     df['Truedaily']=df["Adj Close"].pct_change()
     df['daily']=np.where(df['Truedaily']>=0,1,-1)
     df['daily'][0]=0
     df['adj vol']=df['daily']*df['Volume']
     df['OBV']=df['adj vol'].cumsum()
     return df

# obv=OBV(ohlcv)


# Slope Implementation in Python
def Slope(ser,n):
    slope=[]
    for i in range(n-1):
        slope.append(0)
    for i in range(n,len(ser)+1):
        y=ser[i-n:i]
        x=np.array(range(n))
        x_scaled=(x-x.min())/(x.max()-x.min())
        y_scaled=(y-y.min())/(y.max()-y.min())
        x_scaled=sm.add_constant(x_scaled)
        model=sm.OLS(y_scaled,x_scaled)
        results=model.fit()
        # results.summary()
        slope.append(results.params[-1])
    slope_angle=np.rad2deg(np.arctan(np.array(slope)))
    return np.array(slope_angle)
# ohlcv['angle']= Slope(ohlcv["Adj Close"], 5)   
# ohlcv[['Adj Close', 'angle']].plot(subplots=True,layout=(2,1))

# Renko IMPLEMENTATION
def RENKO(DF,n):
    renko_df=DF.copy()
    renko_df=renko_df.iloc[:,[0,1,2,4,5]]
    renko_df.reset_index(inplace=True)
    renko_df.columns=['date','Open','High','Low','Close','Volume']
    renko=Renko(renko_df)
    renko.brick_size=round(ATR(ohlcv, n).iloc[-1,-1],0)
    renko2=renko.period_close_bricks()
    return renko2
    
# Candle Stick Patterns T A library Github
# import talib
# ohlcv["patterns"]=CDL2CROWS(df["open"], df['high'], df['low'], df['close'])
# output wii be +100 and -100 depending upon the pattern up for +100 and down for - 100

# Key Performance Indicators or KPIS


# CAGR IMPLEMENTATION
def CAGR(DF): 
    df=DF.copy()
    df["daily_return"]=df['Adj Close'].pct_change()
    df['cum_return']=(1+df['daily_return']).cumprod()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    n=len(df)/252
    cagr=((df['cum_return'].iloc[-1])**(1/n))-1
    return cagr

# VOLATILITY IMPLEMENTATION
def volatility(DF):
    df=DF.copy()
    df["daily_return"]=df['Adj Close'].pct_change()
    vol=df['daily_return'].std()*np.sqrt(252)
    return vol


# Sharpe ratio

def Sharpe(DF,rf):
    df=DF.copy()
    sharpe=(CAGR(df)-rf)/volatility(df)
    return sharpe

# s=Sharpe(ohlcv,0.022)
# df=ohlcv.copy()
# df["daily_return"]=df['Adj Close'].pct_change()
# df['cum_return']=(1+df['daily_return']).cumprod()
# max_inv=df['cum_return'].max()
# min_inv=df['cum_return'].min()
# dd=(max_inv-min_inv)/max_inv

# MAX DRAWDOWN IMPLEMENTATION
def MAXDD(DF):   
    df=DF.copy()
    df["daily_return"]=df['Adj Close'].pct_change()
    df['cum_return']=(1+df['daily_return']).cumprod()
    df['cum_rollmax']=df["cum_return"].cummax()
    df['drawdown']=df['cum_rollmax']-df['cum_return']
    df['drawdown_pct']=df['drawdown']/df['cum_rollmax']
    maxdd=df['drawdown_pct'].max()
    return maxdd

# CALMAR RATIO
def CALMAR(DF):
    df=DF.copy()
    cal=CAGR(df)/MAXDD(df)
    return cal































   

    
    