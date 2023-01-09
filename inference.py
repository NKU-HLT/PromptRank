import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer
from nltk import PorterStemmer
from sys import exit

pd.options.mode.chained_assignment = None

MAX_LEN = None
enable_pos = None
temp_en = None
temp_de = None
length_factor = None
position_factor = None
tokenizer = None

def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, temp_en, temp_de, tokenizer
    global enable_pos, length_factor, position_factor

    MAX_LEN = setting_dict["max_len"]
    temp_en = setting_dict["temp_en"]
    temp_de = setting_dict["temp_de"]
    enable_pos = setting_dict["enable_pos"]
    position_factor = setting_dict["position_factor"]
    length_factor = setting_dict["length_factor"]

    tokenizer = T5Tokenizer.from_pretrained("t5-" + setting_dict["model"], model_max_length=MAX_LEN)


def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e!=0 else 0.0
    R = float(num_c) / float(num_s) if num_s!=0 else 0.0
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1

def print_PRF(P, R, F1, N, log):

    log.logger.info("\nN=" + str(N))
    log.logger.info("P=" + str(P))
    log.logger.info("R=" + str(R))
    log.logger.info("F1=" + str(F1) + "\n")
    return 0

def keyphrases_selection(setting_dict, doc_list, labels_stemed, labels,  model, dataloader, device, log):

    init(setting_dict)

    model.eval()

    cos_similarity_list = {}
    candidate_list = []
    cos_score_list = []
    doc_id_list = []
    pos_list = []
    
    num_c_5 = num_c_10 = num_c_15 = 0
    num_e_5 = num_e_10 = num_e_15 = 0
    num_s = 0
    
    template_len = tokenizer(temp_de, return_tensors="pt")["input_ids"].shape[1] - 3 # single space
    # print(template_len)
    # etting_dict["temp_en"] +
    

    for id, [en_input_ids,  en_input_mask, de_input_ids, dic] in enumerate(tqdm(dataloader,desc="Evaluating:")):

        en_input_ids = en_input_ids.to(device)
        en_input_mask = en_input_mask.to(device)
        de_input_ids = de_input_ids.to(device)

        score = np.zeros(en_input_ids.shape[0])
        
        # print(dic["candidate"])
        # print(dic["de_input_len"])
        # exit(0)

        with torch.no_grad():
            output = model(input_ids=en_input_ids, attention_mask=en_input_mask, decoder_input_ids=de_input_ids)[0]
            #print(en_output.shape)
            # x = empty_ids.repeat(en_input_ids.shape[0], 1, 1).to(device)
            # empty_output = model(input_ids=x, decoder_input_ids=de_input_ids)[2]
            
            
            for i in range(template_len, de_input_ids.shape[1] - 3):
                logits = output[:, i, :]
                logits = logits.softmax(dim=1)
                logits = logits.cpu().numpy()

                for j in range(de_input_ids.shape[0]):
                    if i < dic["de_input_len"][j]:
                        score[j] = score[j] + np.log(logits[j, int(de_input_ids[j][i + 1])])
                    elif i == dic["de_input_len"][j]:
                        score[j] = score[j] / np.power(dic["de_input_len"][j] - template_len, length_factor)
                                        # score = score + 0.005 (score - empty_score)
            
               
            doc_id_list.extend(dic["idx"])
            candidate_list.extend(dic["candidate"])
            cos_score_list.extend(score)
            pos_list.extend(dic["pos"])

    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    cos_similarity_list["score"] = cos_score_list
    cos_similarity_list["pos"] = pos_list
    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)

    for i in range(len(doc_list)):
        
        doc_len = len(doc_list[i].split())
        
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        if enable_pos == True:
            #doc_results.loc[:,"pos"] = torch.Tensor(doc_results["pos"].values.astype(float)) / doc_len + position_factor / (doc_len ** 3)
            doc_results["pos"] = doc_results["pos"] / doc_len + position_factor / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]
        #* doc_results["score"].values.astype(float)
        ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        top_k = ranked_keyphrases.reset_index(drop = True)
        top_k_can = top_k.loc[:, ['candidate']].values.tolist()
        #print(top_k)
        #exit()
        candidates_set = set()
        candidates_dedup = []
        for temp in top_k_can:
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)

        # log.logger.debug("Sorted_Candidate: {} \n".format(top_k_can))
        # log.logger.debug("Candidates_Dedup: {} \n".format(candidates_dedup))

        j = 0
        Matched = candidates_dedup[:15]
        porter = PorterStemmer()
        for id, temp in enumerate(candidates_dedup[0:15]):
            tokens = temp.split()
            tt = ' '.join(porter.stem(t) for t in tokens)
            if (tt in labels_stemed[i] or temp in labels[i]):
                Matched[id] = [temp]
                if (j < 5):
                    num_c_5 += 1
                    num_c_10 += 1
                    num_c_15 += 1

                elif (j < 10 and j >= 5):
                    num_c_10 += 1
                    num_c_15 += 1

                elif (j < 15 and j >= 10):
                    num_c_15 += 1
            j += 1

        log.logger.info("TOP-K {}: {} \n".format(i, Matched))
        log.logger.info("Reference {}: {} \n".format(i,labels[i]))

        if (len(top_k[0:5]) == 5):
            num_e_5 += 5
        else:
            num_e_5 += len(top_k[0:5])

        if (len(top_k[0:10]) == 10):
            num_e_10 += 10
        else:
            num_e_10 += len(top_k[0:10])

        if (len(top_k[0:15]) == 15):
            num_e_15 += 15
        else:
            num_e_15 += len(top_k[0:15])

        num_s += len(labels[i])

    p, r, f = get_PRF(num_c_5, num_e_5, num_s)
    print_PRF(p, r, f, 5, log)
    p, r, f = get_PRF(num_c_10, num_e_10, num_s)
    print_PRF(p, r, f, 10, log)
    p, r, f = get_PRF(num_c_15, num_e_15, num_s)
    print_PRF(p, r, f, 15, log)
    
