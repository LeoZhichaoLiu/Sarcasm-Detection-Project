import os
import numpy as np
import json
import re
from tqdm import tqdm
def load_word():
    file= open('data\\renttherunway_final_data.json\\renttherunway_final_data.json',encoding='utf-8')
    words_list = []
    for data in file.readlines():
        data=json.loads(data)
        if data["rating"]==None:
            continue
        label=int(data["rating"])
        # 去除标点符号
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
        text1=str(data["review_text"])
        text2=str(data["review_summary"])
        text1 = text1.replace('\n', '')
        text1 = text1.replace('<br /><br />', ' ')
        text1 = re.sub(r, '', text1)
        text1 = text1.split(' ')
        text1 = [text1[i].lower() for i in range(len(text1)) if text1[i] != '']
        text2 = text2.replace('\n', '')
        text2 = text2.replace('<br /><br />', ' ')
        text2 = re.sub(r, '', text2)
        text2 = text2.split(' ')
        text2 = [text2[i].lower() for i in range(len(text2)) if text2[i] != '']
        words_list.append([text1+text2, label])
    return words_list

def get_word_vector(vocabulary_path,word_list_path,dim):
    vocabulary_vectors = np.load(vocabulary_path, allow_pickle=True)
    word_list = np.load(word_list_path, allow_pickle=True)
    word_list = word_list.tolist()
    data = load_word()
    word_vector=[]
    labels=[]
    for i in tqdm(range(len(data))):
        # print(i)
        sentence = data[i][0]
        labels.append(data[i][1])
        temp = []
        index = 0
        for j in range(len(sentence)):
            try:
                index = word_list.index(sentence[j])
            except ValueError:  # 没找到
                index = 399999
            finally:
                temp.append(list(vocabulary_vectors[index]))  # temp表示一个单词在词典中的序号
        if len(temp) < 50:
            for k in range(len(temp), 50):  # 不足补0
                temp.append([0]*50)
        else:
            temp = temp[0:50]  # 只保留250个
        word_vector.append(temp)

    # print(sentence_code)
    train_data=word_vector[:int(len(word_vector)*0.8)]
    train_label=labels[:int(len(word_vector)*0.8)]
    test_data=word_vector[int(len(word_vector)*0.8):]
    test_label=labels[int(len(word_vector)*0.8):]

    train_data = np.array(train_data)
    train_label=np.array(train_label)
    test_data=np.array(test_data)
    test_label=np.array(test_label)
    np.save('data/train_data'+str(dim), train_data)
    np.save('data/train_label'+str(dim), train_label)
    np.save('data/test_data' + str(dim), test_data)
    np.save('data/test_label' + str(dim), test_label)



if __name__=="__main__":
    get_word_vector('data/vocabulary_vectors_50d.npy','data/word_list_50d.npy',50)
