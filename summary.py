datasets = ["Inspec", "SemEval2017", "SemEval2010", "DUC2001", "krapivin", "nus"]

def get_setting_dict():
    setting_dict = {}
    setting_dict["max_len"] = None
    setting_dict["temp_en"] = None
    setting_dict["temp_de"] = None
    setting_dict["model"] = None
    setting_dict["enable_filter"] = None
    setting_dict["enable_pos"] = None
    setting_dict["position_factor"] = None
    setting_dict["length_factor"] = None
    return setting_dict

F1_scores =[]
setting_dict = get_setting_dict()

for dataset in datasets:
    log_name = dataset + ".log"
    F1_score = []
    with open(log_name, "r") as file:
        for line in file:
            if line[0:3] == "F1=":
                F1_score.append(float(line[3:-1]))

            if setting_dict["length_factor"] is None:
                for key in setting_dict.keys():
                    
                    l = len(key)
                    if line[0:l] == key:
                        setting_dict[key] = line[l + 2:-1]

    F1_scores.append(F1_score)

F1 = [0] * 3

for i in range(3):
    for j in range(len(datasets)):
        F1[i] += F1_scores[j][i]
    F1[i] /= 6

    
with open("./result.txt", "w") as file:
    for i, j in setting_dict.items():
        file.write(i + ": {}\n".format(j))
    
    for i in range(len(datasets)):
        file.write(datasets[i] + "\n")
        for j in range(3):
            file.write(str(F1_scores[i][j]) + "\n")
    file.write("\n")
    for i in range(3):
        file.write(str(F1[i]) + "\n")

