
current_folder = "D:\\miniconda\\UNSW-NB15\\testing"
print(current_folder)
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "14002" + slash
f_train=datapath+'unsw-nb15_training' + split +'.csv'
f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'
f_performance_shallow20=datapath + 'shallow20.txt'

shallow_model_path_20=modelpath+ 'shallow_model_ANN5_hp.keras'
deep_model_path_20=modelpath+ 'deep_model_ANN5_hp.keras'

print(f_test)
print(f_train)
print(shallow_model_path_20)
