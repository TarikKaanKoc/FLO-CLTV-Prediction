#################################################################################################
####                                                                                        ####
####      FLO CLTV PREDİCTİON --> {CLTV VALUE PREDİCTİON WİTH BG/NBD AND GAMMA GAMMA}       ####
####                                                                                        ####
################################################################################################

# -------------------------------------------------------------------------------------------------------------------

# Bussines Problem

""""
FLO would like to set a roadmap for sales and marketing activities.
In order for the company to make a medium-long-term plan,
it is necessary to estimate the potential value that existing customers will provide to the company in the future.
"""

# Features:

# Total Features : 12
# Total Row : 19.945
# CSV File Size : 2.7 MB

""""
- master_id : Unique Customer Number
- order_channel : Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile)
- last_order_channel : The channel where the most recent purchase was made
- first_order_date : Date of the customer's first purchase
- last_order_channel : Customer's previous shopping history
- last_order_date_offline : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online : Total number of purchases made by the customer on the online platform
- order_num_total_ever_offline : Total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
- customer_value_total_ever_online :  Total fees paid for the customer's online purchases
- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
"""

# -------------------------------------------------------------------------------------------------------------------

# Projcet


import pandas as pd
import numpy as np
import datetime as dt
from lifetimes import BetaGeoFitter
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_period_transactions
from lifetimes import GammaGammaFitter

pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv('flo_data_20k.csv')
df = df_.copy()

# Missing Value Analysis
def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

# Check DataFrame - Function
def check_df(df, head=5):
    print("--------------------- Shape ---------------------")
    print(df.shape)
    print("--------------------- Types ---------------------")
    print(df.dtypes)
    print("--------------------- Head ---------------------")
    print(df.head(head))
    print("--------------------- Missing Values Analysis ---------------------")
    print(missing_values_analysis(df))
    print("--------------------- Quantiles ---------------------")
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Outlier Analysis
def outlier_thresholds(df, feautre):
    q_1 = df[feautre].quantile(0.01)
    q_3 = df[feautre].quantile(0.99)
    IQR_Range = q_3 - q_1
    up_limit = q_3 + 1.5 * IQR_Range
    low_limit = q_1 - 1.5 * IQR_Range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

replace_with_thresholds(df,"customer_value_total_ever_online")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"order_num_total_ever_online")

check_df(df)

# Omnichannel refers to the total purchase made over both online and offline platforms.
df['Omnichannel'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

df.head()

df.info()

""""
         Column                                     Dtype  
------>  ----------------          -------------->  ------  
# ---->  first_order_date          -------------->  object
# ---->  last_order_date           -------------->  object
# ---->  last_order_date_online    -------------->  object
# ---->  last_order_date_offline   -------------->  object
"""

# Converting the above mentioned column types from object to datetime format
convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[convert] = df[convert].apply(pd.to_datetime)

df.info()

""""
         Column                                     Dtype  
------>  ----------------          -------------->  --------------  
# ---->  first_order_date          -------------->  datetime64[ns]
# ---->  last_order_date           -------------->  datetime64[ns]
# ---->  last_order_date_online    -------------->  datetime64[ns]
# ---->  last_order_date_offline   -------------->  datetime64[ns]
"""


#                                            * Creating the CLTV Data Structure *
# -------------------------------------------------------------------------------------------------------------------

df["last_order_date"].max() #Timestamp('2021-05-30 00:00:00')

af_date = dt.datetime(2021,7,1)
type(af_date)

df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']


new_df = pd.DataFrame({"CUSTOMER_ID": df["master_id"],
             "RECENCY_CLTV_WEEKLY": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_WEEKLY": ((af_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
             "FREQUENCY": df["order_num_total"],
             "MONETARY_CLTV_AVG": df["Omnichannel"] / df["order_num_total"]})

new_df.head()


""""
                            CUSTOMER_ID  RECENCY_CLTV_WEEKLY  T_WEEKLY  FREQUENCY  MONETARY_CLTV_AVG
0  ************************************             17.00000  34.85714    5.00000          187.87400
1  ************************************            209.85714 229.14286   21.00000           95.88333
2  ************************************             52.28571  83.14286    5.00000          117.06400
3  ************************************              1.57143  25.14286    2.00000           60.98500
4  ************************************             83.14286  99.71429    2.00000          104.99000
"""

new_df.info()

"""" 
     Column                                Dtype  
---  ----------------     ------------->   ------   
 0   CUSTOMER_ID          ------------->   object 
 1   RECENCY_CLTV_WEEKLY  ------------->   float64
 2   T_WEEKLY             ------------->   float64
 3   FREQUENCY            ------------->   float64
 4   MONETARY_CLTV_AVG    ------------->   float64
"""



#                                            * Establishment of BG/NBD, Gamma-Gamma Models and calculation of CLTV *
# -------------------------------------------------------------------------------------------------------------------

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(new_df['FREQUENCY'],
        new_df['RECENCY_CLTV_WEEKLY'],
        new_df['T_WEEKLY'])

# OUT : <lifetimes.BetaGeoFitter: fitted with 19945 subjects, a: 0.00, alpha: 82.71, b: 0.00, r: 3.79>

# Forecasting expected purchases from customers in 3 months
new_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               new_df['FREQUENCY'],
                                               new_df['RECENCY_CLTV_WEEKLY'],
                                               new_df['T_WEEKLY'])

new_df.head()

""""
                            CUSTOMER_ID  RECENCY_CLTV_WEEKLY  T_WEEKLY  FREQUENCY  MONETARY_CLTV_AVG  exp_sales_3_month
0  ************************************             17.00000  34.85714    5.00000          187.87400            0.89673
1  ************************************            209.85714 229.14286   21.00000           95.88333            0.95374
2  ************************************             52.28571  83.14286    5.00000          117.06400            0.63566
3  ************************************              1.57143  25.14286    2.00000           60.98500            0.64371
4  ************************************             83.14286  99.71429    2.00000          104.99000            0.38057
"""

# Forecasting expected purchases from customers in 6 months
new_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               new_df['FREQUENCY'],
                                               new_df['RECENCY_CLTV_WEEKLY'],
                                               new_df['T_WEEKLY'])

new_df.head()

""""
                            CUSTOMER_ID  RECENCY_CLTV_WEEKLY  T_WEEKLY  FREQUENCY  MONETARY_CLTV_AVG  exp_sales_3_month  exp_sales_6_month
0  ************************************             17.00000  34.85714    5.00000          187.87400            0.89673            1.79346
1  ************************************            209.85714 229.14286   21.00000           95.88333            0.95374            1.90748
2  ************************************             52.28571  83.14286    5.00000          117.06400            0.63566            1.27132
3  ************************************              1.57143  25.14286    2.00000           60.98500            0.64371            1.28741
4  ************************************             83.14286  99.71429    2.00000          104.99000            0.38057            0.76114

"""

# prediction validation
plot_period_transactions(bgf)
plt.show()

# Establishing the GAMMA-GAMMA Model
ggf = GammaGammaFitter(penalizer_coef=0.01)

# Expected Number Of Transaction - Expected Average Profit
ggf.fit(new_df['FREQUENCY'], new_df['MONETARY_CLTV_AVG'])

# EXP_AVERAGE_VALUE = Expected Number Of Transaction - Expected Average Profit
new_df['EXP_AVERAGE_VALUE'] = ggf.conditional_expected_average_profit(
    new_df['FREQUENCY'], new_df['MONETARY_CLTV_AVG'])

new_df.head()

""""
                            CUSTOMER_ID  RECENCY_CLTV_WEEKLY  T_WEEKLY  FREQUENCY  MONETARY_CLTV_AVG  exp_sales_3_month  exp_sales_6_month  EXP_AVERAGE_VALUE
0  ************************************             17.00000  34.85714    5.00000          187.87400            0.89673            1.79346          193.63268
1  ************************************            209.85714 229.14286   21.00000           95.88333            0.95374            1.90748           96.66505
2  ************************************             52.28571  83.14286    5.00000          117.06400            0.63566            1.27132          120.96762
3  ************************************              1.57143  25.14286    2.00000           60.98500            0.64371            1.28741           67.32015
4  ************************************             83.14286  99.71429    2.00000          104.99000            0.38057            0.76114          114.32511

"""

# Calculation of CLTV with BG-NBD and GG model  - (6 MONTHS)


cltv = ggf.customer_lifetime_value(bgf,
                                   new_df['FREQUENCY'],
                                   new_df['RECENCY_CLTV_WEEKLY'],
                                   new_df['T_WEEKLY'],
                                   new_df['MONETARY_CLTV_AVG'],
                                   time=6, # 6 MONTH
                                   freq="W",  # T's frequency information. (We mentioned that it was weekly.)
                                   discount_rate=0.01) # consider discounts that can be made over time (discount rate)


cltv

""""
0       364.36586
1       193.46227
2       161.35790
3        90.93503
4        91.30097
           ...   
           ...
"""
new_df['CLTV'] = cltv

new_df.sort_values(by='CLTV', ascending = False).head(20)


#                                            * Creating Segments Based on CLTV Values *
# -------------------------------------------------------------------------------------------------------------------

new_df["SEGMENT"] = pd.qcut(new_df["CLTV"], 4, labels=["D", "C", "B", "A"])

new_df.head()

# RESULT ! --->  Action Time!

new_df.groupby("SEGMENT").agg({"count","mean","sum"})

""""
        RECENCY_CLTV_WEEKLY                        T_WEEKLY                        FREQUENCY                     MONETARY_CLTV_AVG                         exp_sales_3_month                    exp_sales_6_month                    EXP_AVERAGE_VALUE                          CLTV                        
                      count      mean          sum    count      mean          sum     count    mean         sum             count      mean           sum             count    mean        sum             count    mean        sum             count      mean           sum count      mean           sum
SEGMENT                                                                                                                                                                                                                                                                                                     
D                      4987 136.76539 682049.00000     4987 164.25483 819138.85714      4987 3.74133 18658.00000              4987  92.27589  460179.83968              4987 0.39684 1979.03858              4987 0.79368 3958.07715              4987  97.80296  487743.34577  4987  77.55272  386755.39892
C                      4986  93.15225 464457.14286     4986 117.49977 585853.85714      4986 4.37405 21809.00000              4986 126.02551  628363.17842              4986 0.50104 2498.20542              4986 1.00209 4996.41083              4986 132.51034  660696.57538  4986 132.35107  659902.45681
B                      4986  82.09461 409323.71429     4986 104.76829 522374.71429      4986 5.07962 25327.00000              4986 160.52095  800357.47151              4986 0.57167 2850.32481              4986 1.14333 5700.64962              4986 167.89807  837139.78006  4986 190.07141  947696.04491
A                      4986  69.03341 344200.57143     4986  88.49817 441251.85714      4986 6.71781 33495.00000              4986 229.58569 1144714.23688              4986 0.72730 3626.32803              4986 1.45460 7252.65607              4986 238.75506 1190432.71344  4986 341.65475 1703490.60775
"""
# -------------------------------------------------------------------------------------------------------------------