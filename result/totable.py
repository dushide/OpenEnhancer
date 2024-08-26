import re
import csv

# 1. 创建文件对象







# while (line):
#
#     for u in us:
#
#         for knownsMinimumMag in knownsMinimumMags:
#             # line = f.readline()
#             d = 12
#             print(u, knownsMinimumMag)
#             csv_writer.writerow(['u=' + str(u), 'knownsMinimumMag=' + str(knownsMinimumMag)])
#             while(d!=0):
#                 d-=1
#
#
#                 dataset = line.split(":")[6].split(",")[0]
#                 line = f.readline()
#                 ccrs = re.findall("\d+.\d+",line.split(":")[-1])
#
#                 while(len(ccrs)!=5):
#                     ccrs.insert(0,'N/A')
#                 print(dataset,  ccrs)
#
#                 csv_writer.writerow([dataset] + ccrs)
#                 line = f.readline()
#                 # line = f.readline()
#             line = f.readline()
#             if line.strip() == "":
#                 break

def para2(paras,para2s,):
    line = f.readline()

    while(line):
        csv_writer.writerow(['u/known'] + para2s)
        line = f.readline()
        for alpha in paras:
            ccrs = []

            print(line)
            dataset = line.split("dataset:")[1].split(",")[0]

            for beta in para2s:
                ccr = re.findall("\d+.\d+", line.split(":")[-1])[-2]
                ccrs.append(ccr)
                line = f.readline()
                line = f.readline()
                line = f.readline()
                print(line)
            csv_writer.writerow([alpha] + ccrs)
            # print(ccrs)
        csv_writer.writerow([dataset])

def para1(paras,p):
    line = f.readline()
    csv_writer.writerow([p] + paras)
    while(line):

        ccrs = []
        for para in paras:
            line = f.readline()
            print(line)
            dataset = line.split("dataset:")[1].split(",")[0]
            print(dataset)
            ccr = re.findall("\d+.\d+", line.split(":")[-1])[-2]
            line = f.readline()
            print(line)

            ccrs.append(ccr)
            line = f.readline()
        csv_writer.writerow([dataset] + ccrs)
            # print(ccrs)
        line = f.readline()
        # line = f.readline()
        # line = f.readline()
        print(line)

SCORE=['ACC','f1_ma','f1_mi','Precision','Recall','Time(s)']
blocks =  [1,2,3,4,5,6,7,8,9,10]



f2 = open('tmp.csv', 'w', encoding='utf-8', newline="")
# 2. 基于文件对象构建 csv写入对象
csv_writer = csv.writer(f2)
# f = open('weightcenter/v3/result/openness0.1_alphabeta_batch=50.txt', "r")

f = open('res_attention_openness0.1_layers.txt', "r")
# f = open('res_attention_openness0.1_lambda.txt', "r")
# f = open('weightcenter/v3/result/openness0.1_u_batch=50.txt', "r")

fpr_values=[0.001,0.005,0.01,0.05,0.1]
us = [0.001,0.005,0.01,0.05, 0.1]
knownsMinimumMags = [0.05, 0.1, 0.5, 1, 5, 10]
alphas = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
lambda1s = lambda2s = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

i=6
datasets=[ 'Animals','AWA','NUSWIDEOBJ','VGGFace2-50','ESP-Game','NUSWIDE20K ']
# para2()
# us = [0.001, 0.005, 0.01, 0.05, 0.1]
# knownsMinimumMags = [0.05,0.1,0.5,1,5,10]
para1(blocks,'blocks')
# para1(alphas,'alphas')
# para2(lambda1s,lambda2s)
# para1(us,'u')