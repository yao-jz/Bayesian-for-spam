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
# train_list = [json.load(open("train"+str(i)+".json", "r")) for i in range(5)]
email = json.load(open("./email_list.json","r"))
ratio = [i/100 for i in range(5,100,5)]
f1_list = []
fb_list = []
accuracy_list = []
positive_precision = []
negative_recall = []
accuracy = 0
f1 = 0
fb = 0
precision = 0
recall = 0
for r in tqdm(ratio):
	test_list = []
	train_number = int(len(email) * r)
	ttrain_list = email[:train_number]
	test = email
	train = list(map(c_b,ttrain_list))
	test = list(map(c_b, test))
	for e in test:
		l = e["text"].split(b'\n')
		ll = []
		for i in l:
			ll += i.split(b' ')
		a = []
		for i in ll:
			if i != b'' and (re.fullmatch(b'[a-zA-Z0-9,.?\']{1,20}',i)):
				a.append(i)
		test_list.append({"label":e["label"], "text_list": a})
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
			spam_vocab_list.append(a)
		else:
			ham_vocab_list.append(a)
	spam_length = 0
	ham_length = 0
	for i in spam_vocab_list:
		spam_length += len(i)
	for i in ham_vocab_list:
		ham_length += len(i)
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
	label_list = []
	pred_list = []
	for t in test_list:
		label_list.append(t["label"])
		spam = math.log(spam_prob) + sum([math.log(spam_word_prob[i]) for i in t["text_list"]])
		ham = math.log(ham_prob) + sum([math.log(ham_word_prob[i]) for i in t["text_list"]])
		if spam > ham: # spam
			pred_list.append(0)
		else:
			pred_list.append(1)
	f1_list.append(f1_score(label_list, pred_list,average="macro"))
	fb_list.append(fbeta_score(label_list, pred_list, beta=0.5))
	accuracy_list.append(accuracy_score(label_list, pred_list))
	positive_precision.append(float(precision_score(label_list, pred_list, average=None)[1]))
	negative_recall.append(float(recall_score(label_list, pred_list, average=None)[0]))
print([f1_list, fb_list, accuracy_list, positive_precision, negative_recall])
#[[0.9111775523561807, 0.9610923194579677, 0.9702332451811291, 0.9765149675287548, 0.9790772270075571, 0.9795336697611352, 0.9814225651906704, 0.9836160491309014, 0.9847563596771222, 0.9850907674460533, 0.9855555409830885, 0.9871638430768293, 0.9876416068101546, 0.988376123233462, 0.9887870683716729, 0.9890789841322793, 0.9896945390757127, 0.9898714510133404, 0.9903386753376111], [0.9387766368554709, 0.9611555384269518, 0.9636108074340154, 0.9687780564069222, 0.9718656231633, 0.9732599596961711, 0.9753057748877536, 0.9768625636279499, 0.9787375312683365, 0.9782675661522348, 0.9792765288185771, 0.981991165477404, 0.9818593754816436, 0.9827564952075691, 0.9832958362868678, 0.9838580370633038, 0.9847783843804941, 0.9849209041290202, 0.9858492021851176], [0.9243297551689493, 0.9653905134577759, 0.9732959653111946, 0.9788747289937074, 0.9811749775263074, 0.9815980117391994, 0.9832901485907672, 0.9852466818253927, 0.9862778277193168, 0.98656866374068, 0.986991697953572, 0.9884458780603881, 0.9888689122732801, 0.9895299032309238, 0.9899000581672043, 0.9901644545502618, 0.9907196869546825, 0.990878324784517, 0.9913013589974089], [0.9850357211816954, 0.970018637063447, 0.9655712050078247, 0.9685831463282519, 0.9714749536178108, 0.9734108527131783, 0.9751547987616099, 0.975808936825886, 0.9779372058936975, 0.9768532759151031, 0.9781302941629447, 0.9812485531290994, 0.9806049411221427, 0.9814558325638658, 0.981994459833795, 0.9826763165999384, 0.9836709543248864, 0.9837529837529837, 0.9848916981422955], [0.9937780989081567, 0.9851477199743096, 0.9823378291586384, 0.9837026332691072, 0.9851878612716763, 0.9862315350032113, 0.9871146435452793, 0.9873956326268465, 0.988519588953115, 0.987917469492614, 0.9885998715478485, 0.9902456647398844, 0.9898843930635838, 0.9903259473346179, 0.990606936416185, 0.9909682080924855, 0.9914900449582531, 0.9915301862556197, 0.9921323057161208]]
json.dump([f1_list, fb_list, accuracy_list, positive_precision, negative_recall], open("is1.json","w"))
plt.plot(ratio, f1_list, label="f1_score")
plt.plot(ratio, fb_list, label="f0.5_score")
plt.plot(ratio, accuracy_list, label="accuracy")
plt.plot(ratio, positive_precision, label="positive precision")
plt.plot(ratio, negative_recall, label="negative recall")
plt.xlabel("train ratio")
plt.ylabel("metrics")
plt.legend()
plt.savefig("pic1.png")
