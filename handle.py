import json
train_ratio = 0.2
email_list = json.load(open("email_list.json", "r"))
train_num = int(train_ratio * len(email_list))
train1 = email_list[:train_num]
train2 = email_list[train_num:2*train_num]
train3 = email_list[2*train_num:3*train_num]
train4 = email_list[3*train_num:4*train_num]
train5 = email_list[4*train_num:]
json.dump(train1, open("train0.json", "w"))
json.dump(train2, open("train1.json", "w"))
json.dump(train3, open("train2.json", "w"))
json.dump(train4, open("train3.json", "w"))
json.dump(train5, open("train4.json", "w"))