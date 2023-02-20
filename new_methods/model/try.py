"""
import os
import csv
store_path = r'C:\Users\meemu\Downloads\CODE_AND_RESULTS\new_methods\utils\Res'
path = '../utils/model_trained/params_single2_'
select_model = 'DA_PAM'
methods = ['all', 'pam', 'cam']
#K = [4, 8, 16]
K = [4, 8, 16,32]
#COS = [0.01, 0.02, 0.05, 0.1, 0.2]
COS = [0.01, 0.005]

model_path = path + select_model + '_' + str(cos) + '_' + str(k) + '.pkl'
file = os.path.join(store_path, 'result.csv')

with open(file, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    spamwriter.writerow(['method', 'K', 'COS'])
    for method in methods:
        for k in K:
            for cos in COS:
                spamwriter.writerow([str(method), str(k), str(cos)])
"""