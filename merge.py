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
		l = e["text"].split(b'\n')
		ll = []
		for i in l:
			ll += i.split(b' ')
		a = []
		for i in ll:
			if i != b'' and (re.fullmatch(b'[a-zA-Z0-9,.?\']{1,20}',i)):
				a.append(i)
		if re.search(from_pattern, e["text"]):
			test_list.append({"label":e["label"], "date": re.search(from_pattern, e["text"])[1], "text_list": a})
		else:
			test_list.append({"label":e["label"], "date": "none", "text_list": a})
	spam_hour_list = []
	ham_hour_list = []
	spam_vocab_list = []
	ham_vocab_list = []
	for e in train[:]:
		l = e["text"].split(b'\n')
		ll = []
		for i in l:
			ll += i.split(b' ')
		a = []
		for i in ll:
			if i != b'' and (re.fullmatch(b'[a-zA-Z0-9,.?\']{1,20}',i)):
				a.append(i)
		if e["label"] == 0:
			li = re.search(from_pattern, e["text"])
			if li:
				spam_hour_list.append(li[1])
			spam_vocab_list.append(a)
			# else:
			# 	spam_hour_list.append("None")
		else:
			ham_vocab_list.append(a)
			li = re.search(from_pattern, e["text"])
			if li:
				ham_hour_list.append(li[1])
			# else:
			# 	ham_hour_list.append("None")
	spam_length = 0
	ham_length = 0
	for i in spam_vocab_list:
		spam_length += len(i)
	for i in ham_vocab_list:
		ham_length += len(i)
	spam_hour_emerge_num = defaultdict(lambda: 0)
	ham_hour_emerge_num = defaultdict(lambda: 0)
	spam_hour_prob = defaultdict(lambda: 1/len(spam_hour_list))	# 垃圾邮件中词的概率
	ham_hour_prob = defaultdict(lambda : 1/len(ham_hour_list))
	spam_word_emerge_num = defaultdict(lambda: 0)
	ham_word_emerge_num = defaultdict(lambda: 0)
	spam_word_prob = defaultdict(lambda: 1/spam_length)	# 垃圾邮件中词的概率
	ham_word_prob = defaultdict(lambda : 1/ham_length)
	spam_prob = len(spam_vocab_list) / len(train)
	ham_prob = len(ham_vocab_list) / len(train)
	for i in spam_vocab_list:
		for j in i:
			spam_word_emerge_num[j] += 1
	for i in ham_vocab_list:
		for j in i:
			ham_word_emerge_num[j] += 1
	for i in ham_word_emerge_num.keys():
		ham_word_prob[i] = ham_word_emerge_num[i]/ham_length
	for i in spam_word_emerge_num.keys():
		spam_word_prob[i] = spam_word_emerge_num[i]/spam_length
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
	for t in test_list[:]:
		label_list.append(t["label"])
		# print(math.log(spam_hour_prob[t["date"]]))
		# print(sum([math.log(spam_word_prob[i]) for i in t["text_list"]]))
		spam = math.log(spam_prob) + sum([math.log(spam_word_prob[i]) for i in t["text_list"]])+2*math.log(spam_hour_prob[t["date"]])
		ham = math.log(ham_prob) + sum([math.log(ham_word_prob[i]) for i in t["text_list"]])+2*math.log(ham_hour_prob[t["date"]])
		# spam = math.log(spam_prob) + 0*math.log(spam_hour_prob[t["date"]]) + sum([math.log(spam_word_prob[i]) for i in t["text_list"]])
		# ham = math.log(ham_prob) + 0*math.log(ham_hour_prob[t["date"]]) + sum([math.log(spam_word_prob[i]) for i in t["text_list"]])
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