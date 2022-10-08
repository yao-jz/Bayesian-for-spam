import sys
import json
import re
from pathlib import Path
import random

"""
spam = 0
ham = 1
"""
def get_label(s):
	if s == "spam":
		return 0
	else:
		return 1
random.seed(2333)
label_file = open("../trec06p/label/index", "r").read().split("\n")
label_file.pop(-1)
data_path = Path("../trec06p/data")
label_pattern = r'(ham|spam) \.\./data/(\d{3})/(\d{3})'
email_name_pattern = r'.+(\d{3})/(\d{3})'
email = {}
email_list = []
for i in label_file[:]:
	l = re.findall(label_pattern, i)[0]
	email["".join(l[1:])] = {"label":get_label(l[0])}
for i in data_path.iterdir():
	for j in i.iterdir():
		index = "".join(re.findall(email_name_pattern, str(j))[0])
		email[index]["text"] = str(open(j, "rb").read())
for k in list(email.keys())[:]:
	email_list.append(email[k])
random.shuffle(email_list)
json.dump(email_list, open("email_list.json","w"))