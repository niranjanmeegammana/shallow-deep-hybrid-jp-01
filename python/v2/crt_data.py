import psutil
import time
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

#-----------------
# Define the features
features20 = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt']

features40 = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

#-----------------
current_folder = "d:\\miniconda\\UNSW-NB15\\testing"
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "models" + slash
f_train=datapath+'unsw-nb15_training' + split +'.csv'
f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'
f_val=datapath + 'unsw-nb15_validation' + split+ '.csv'

f_Xtest20=datapath + 'unsw-nb15_Xtest20' + split+ '.csv'
f_Xtest40=datapath + 'unsw-nb15_Xtest40' + split+ '.csv'
f_ytest=datapath + 'unsw-nb15_ytest' + split+ '.csv'

print(f_test)
print(f_train)
print(f_val)
print(f_Xtest20)
print(f_Xtest40)
print(f_ytest)

df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)
df_val=pd.read_csv(f_val)

# Concatenate the DataFrames vertically
concat_df = pd.concat([df_train, df_test, df_val], axis=0)

# Reset the index of the concatenated DataFrame
concat_df.reset_index(drop=True, inplace=True)
print(concat_df.shape)

# Separate features and labels for testing set
X_test20 = concat_df[features20].values 
X_test40 = concat_df[features40].values  
y_test = concat_df['label'].values

# Standardize the features
scaler = StandardScaler()
X_test20 = scaler.fit_transform(X_test20)
X_test40 = scaler.fit_transform(X_test40)

np.savetxt(f_Xtest20, X_test20, delimiter=',')
np.savetxt(f_Xtest40, X_test40, delimiter=',')
np.savetxt(f_ytest, y_test, delimiter=',')
print('done')