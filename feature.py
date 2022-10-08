import random
import re
from tqdm import tqdm
import math
import json
import sys
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, fbeta_score
from collections import defaultdict
import copy
random.seed(2333)
def c_b(item):
	k = copy.deepcopy(item)
	k["text"] = eval(k["text"])
	return k
train_list = [json.load(open("train"+str(i)+".json", "r")) for i in range(5)]
accuracy = 0
f1 = 0
fb = 0
precision = 0
recall = 0

"""
subject用词表的形式
date和from用概率的形式
"""
for test_index in range(5):
	# 处理数据集 
	ttrain_list = []
	for kkk in range(5):
		if kkk == test_index:
			continue
		ttrain_list += train_list[kkk]
	test_list = []
	test = train_list[test_index]
	train = list(map(c_b,ttrain_list))
	test = list(map(c_b, test))
	from_pattern = b'From:.+?@(.+?)>\\n'
	hour_pattern = b'Date:.+(\d{2}):\d{2}:\d{2}(.+|.?)\\n'
	for e in test:
		if re.search(from_pattern, e["text"]):
			test_list.append({"label":e["label"], "date": re.search(from_pattern, e["text"])[1]})
		else:
			test_list.append({"label":e["label"], "date": "none"})
	spam_hour_list = []
	ham_hour_list = []
	for e in train[:]:
		if e["label"] == 0:
			li = re.search(from_pattern, e["text"])
			if li:
				spam_hour_list.append(li[1])
		else:
			li = re.search(from_pattern, e["text"])
			if li:
				ham_hour_list.append(li[1])
	spam_hour_emerge_num = defaultdict(lambda: 0)
	ham_hour_emerge_num = defaultdict(lambda: 0)
	spam_hour_prob = defaultdict(lambda: 1/len(spam_hour_list))	# 垃圾邮件中词的概率
	ham_hour_prob = defaultdict(lambda : 1/len(ham_hour_list))
	spam_prob = len(spam_hour_list) / len(train)
	ham_prob = len(ham_hour_list) / len(train)
	for i in spam_hour_list:
		spam_hour_emerge_num[i] += 1
	for i in ham_hour_list:
		ham_hour_emerge_num[i] += 1
	for i in ham_hour_emerge_num.keys():
		ham_hour_prob[i] = ham_hour_emerge_num[i]/len(ham_hour_list)
	for i in spam_hour_emerge_num.keys():
		spam_hour_prob[i] = spam_hour_emerge_num[i]/len(spam_hour_list)
	label_list = []
	pred_list = []
	for t in test_list:
		label_list.append(t["label"])
		spam = math.log(spam_prob) + math.log(spam_hour_prob[t["date"]])
		ham = math.log(ham_prob) + math.log(ham_hour_prob[t["date"]])
		if spam > ham: # spam
			pred_list.append(0)
		else:
			pred_list.append(1)
	f1 += f1_score(label_list, pred_list,average="macro")
	fb += fbeta_score(label_list, pred_list, beta=0.5)
	accuracy += accuracy_score(label_list, pred_list)
	precision += precision_score(label_list, pred_list, average=None)
	recall += recall_score(label_list, pred_list, average=None)

print("accuracy:", accuracy / 5)
print("f1_score: ", f1/5)
print("f0.5_score:", fb / 5)
print("precision_score:", precision/5)
print("recall_score:", recall/5)