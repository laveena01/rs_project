import os
import csv

store_path = r'C:\Users\meemu\Downloads\CODE_AND_RESULTS\new_methods\utils\Res'
path = '../utils/model_trained/params_single2_'
select_model = 'DA_PAM'
methods = ['all', 'pam', 'cam']
K = [4, 8, 16,32]
COS = [0.01, 0.005]

for method in methods:
    for k in K:
        for cos in COS:
            model_path = path + select_model + '_' + str(cos) + '_' + str(k) + '.pkl'

file = os.path.join(store_path, 'result.csv')

with open(file, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    spamwriter.writerow(['method', 'K', 'COS'])
    for method in methods:
        for k in K:
            for cos in COS:
                spamwriter.writerow([str(method), str(k), str(cos)])
