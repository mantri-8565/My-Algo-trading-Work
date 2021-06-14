# Getting important libraries

import copy
import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import statsmodels.api as sm
from stocktrends import Renko

# defining functions for Technical Analysis
def ATR(DF, r):
    df=DF.copy()
    df['hl']=abs(df['High']-df['Low'])
    df['hc'] =   abs(df['High']-df['Close'].shift(1))
    df['lc'] =   abs(df['Low']-df['Close'].shift(1))
    df['tr']=df[['hl','hc','lc']].max(axis=1,skipna=False)                  
    df['atr']=df['tr'].rolling(window=r).mean()
    df.drop(['hl','hc','lc'],axis=1,inplace=True)
    return df['atr']


# CAGR IMPLEMENTATION
def CAGR(DF): 
    df=DF.copy()
    # df["daily_return"]=df['avg_mean'].pct_change()
    df['cum_return']=(1+df['return']).cumprod()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    n=len(df)/(252*78)
    cagr=(df['cum_return'].tolist()[-1])**(1/n)-1
    return cagr

# VOLATILITY IMPLEMENTATION
def volatility(DF):
    df=DF.copy()
    # df["daily_return"]=df['Close'].pct_change()
    vol=df['return'].std()*np.sqrt(252*78)
    return vol


# Sharpe ratio

def Sharpe(DF,rf):
    df=DF.copy()
    sharpe=(CAGR(df)-rf)/volatility(df)
    return sharpe

# MAX DRAWDOWN IMPLEMENTATION
def MAXDD(DF):   
    df=DF.copy()
    # df["daily_return"]=df['Close'].pct_change()
    df['cum_return']=(1+df['return']).cumprod()
    df['cum_rollmax']=df["cum_return"].cummax()
    df['drawdown']=df['cum_rollmax']-df['cum_return']
    df['drawdown_pct']=df['drawdown']/df['cum_rollmax']
    maxdd=df['drawdown_pct'].max()
    return maxdd


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

# Renko IMPLEMENTATION
def RENKO(DF,n):
    renko_df=DF.copy()
    renko_df.reset_index(inplace=True)
    renko_df=renko_df.iloc[:,[0,1,2,3,4,5]]
    renko_df.columns=['date','open','high','low','close','volume']
    renko=Renko(renko_df)
    renko.brick_size=max(0.5,round(ATR(DF, n).iloc[-1],0))
    renko2=renko.period_close_bricks()
    # renko2.columns=['Date','Open','High','Low','Close','Volume','uptrend']
    renko2["bar_num"] = np.where(renko2["uptrend"]==True,1,np.where(renko2["uptrend"]==False,-1,0))
    for i in range(1,len(renko2['bar_num'])):
        if(renko2['bar_num'][i]>0 and renko2['bar_num'][i-1]>0 ):
            renko2['bar_num'][i]+=renko2['bar_num'][i-1]
        elif(renko2['bar_num'][i]<0 and renko2['bar_num'][i-1]<0 ):
            renko2['bar_num'][i]+=renko2['bar_num'][i-1]
    renko2.drop_duplicates(subset="date",keep="last",inplace=True)       
    renko2.columns = ["Date","Open","High","Low","Close","uptrend","bar_num"]
    return renko2
 #Opening Balanace volume
 
def OBV (DF):
     df=DF.copy()
     df['Truedaily']=df["Close"].pct_change()
     df['daily']=np.where(df['Truedaily']>=0,1,-1)
     df['daily'][0]=0
     df['adj vol']=df['daily']*df['Volume']
     df['OBV']=df['adj vol'].cumsum()
     return df["OBV"]


# Gathering neccesary date and removing noise and problems
keypath="D:\\Sourav\\api key\\apikey.txt"
ts=TimeSeries(key=(open(keypath,"r")).read(),output_format="pandas")
tickers = ["MSFT","AAPL","FB","AMZN","INTC", "CSCO","VZ","IBM","QCOM","LYFT"]
ohlcv={}
start_time=time.time()
api_call=0
for ticker in tickers:
    api_call+=1
    data=ts.get_intraday(symbol=ticker,interval="5min",outputsize="full")[0]
    data.columns = ["Open","High","Low","Close","Volume"]
    ohlcv[ticker]=data
    if api_call==5:
        api_call=0
        time.sleep(60 - ((time.time() - start_time) % 60.0))
    ohlcv[ticker]=ohlcv[ticker].between_time('09:35','16:00')
    ohlcv[ticker]=ohlcv[ticker].iloc[::-1]    


# getting technical indicators that will be needed for analysis 
tickers=ohlcv.keys()
ohlcv_renko={}
ohlcv_dict=copy.deepcopy(ohlcv)
for ticker in tickers:
    print("Calculating RENKO and OBV  ",ticker)
    ohlcv_dict[ticker]['Date']=ohlcv_dict[ticker].index
    renko_data=RENKO(ohlcv_dict[ticker],120)
    ohlcv_renko[ticker]=ohlcv_dict[ticker].merge(renko_data.loc[:,["Date","bar_num"]],how="outer",on="Date")
    ohlcv_renko[ticker]['bar_num'].fillna(method='ffill',inplace=True)
    ohlcv_dict[ticker]["obv"]=OBV(ohlcv_dict[ticker])
    ohlcv_dict[ticker]["slope"]=Slope(ohlcv_dict[ticker]['obv'],5)
   
ticker_ret ={}
ticker_signal={}
for ticker in tickers:
    ticker_ret[ticker]=[]
    ticker_signal[ticker]=""

#  Generating Signal and calculating returns
for ticker in tickers:
    for i in range(len(ohlcv_renko[ticker])):
        if ticker_signal[ticker] == "":
            ticker_ret[ticker].append(0)
            if ohlcv_renko[ticker]["bar_num"][i]>=2 and ohlcv_renko[ticker]["slope"][i]>30:
                ticker_signal[ticker] = "Buy"
            elif ohlcv_renko[ticker]["bar_num"][i]<=-2 and ohlcv_renko[ticker]["slope"][i]<-30:
                ticker_signal[ticker] = "Sell"
        elif ticker_signal[ticker] == "Buy":
            ticker_ret[ticker].append((ohlcv_renko[ticker]["Close"][i]/ohlcv_renko[ticker]["Close"][i-1])-1)
            if ohlcv_renko[ticker]["bar_num"][i]<=-2 and ohlcv_renko[ticker]["slope"][i]<-30:
                ticker_signal[ticker] = "Sell"
            elif ohlcv_renko[ticker]["bar_num"][i]<2:
                ticker_signal[ticker] = ""        
        elif ticker_signal[ticker] == "Sell":
            ticker_ret[ticker].append((ohlcv_renko[ticker]["Close"][i-1]/ohlcv_renko[ticker]["Close"][i])-1)
            if ohlcv_renko[ticker]["bar_num"][i]>=2 and ohlcv_renko[ticker]["slope"][i]>30:
                ticker_signal[ticker] = "Buy"
            elif ohlcv_renko[ticker]["bar_num"][i]>-2:
                ticker_signal[ticker] = ""

    ohlcv_renko[ticker]['return']=np.array(ticker_ret[ticker])


#  Calculating overall KPIs
strategy_df=pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker]=ohlcv_renko[ticker]['return']
strategy_df['return']=strategy_df.mean(axis=1)
CAGR(strategy_df)
Sharpe(strategy_df,0.025)
MAXDD(strategy_df)

cagr={}
sharpe_ratio={}
maxdd={}
for ticker in tickers:
    cagr[ticker]=CAGR(ohlcv_renko[ticker])
    sharpe_ratio[ticker]=Sharpe(ohlcv_renko[ticker],0.025)
    maxdd[ticker]=MAXDD(ohlcv_renko[ticker])
    
KPI_df=pd.DataFrame([cagr,sharpe_ratio,maxdd],index=["CAGR","SHARPE","MAXDD"])
KPI_df.T
#  visualization of data
(1+strategy_df['return']).cumprod().plot()



























