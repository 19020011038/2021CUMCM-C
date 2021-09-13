import pandas as pd
import numpy as np
import xlwt
import matplotlib.pyplot as plt
from problem1 import final_list, id_list, tag_list5
import statsmodels.api as sm
import arrow

df = pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name='企业的订货量（m³）')
df2 = pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name='供应商的供货量（m³）')

fifty_tag5_list = []

# 将tag5写入列表，并乘系数
for i in range(402):
    index_s = final_list[i].get("shop")[1:4]
    pos = 0
    while index_s[pos] == '0':
        pos += 1
    index = int(index_s[pos:4])

    if final_list[i].get("type") == 'A':
        tag5 = (tag_list5[index - 1] * 5) / 3
    elif final_list[i].get("type") == 'B':
        tag5 = (tag_list5[index - 1] * 50) / 33
    else:
        tag5 = (tag_list5[index - 1] * 25) / 18
    fifty_tag5_list.append({"shop": final_list[i].get("shop"), "type": final_list[i].get("type"),
                            "tag5": tag5, "index": index - 1})

# 按照tag5供货能力排序
for i in range(402):
    for j in range(401):
        if fifty_tag5_list[j].get("tag5") < fifty_tag5_list[j + 1].get("tag5"):
            temp = fifty_tag5_list[j]
            fifty_tag5_list[j] = fifty_tag5_list[j + 1]
            fifty_tag5_list[j + 1] = temp


#  求解问题1
flag_over_282 = False
sum = 0
num = 0
while not flag_over_282:
    if fifty_tag5_list[num].get("type") == 'A':
        sum += (fifty_tag5_list[num].get("tag5") * 5) / 3
    elif fifty_tag5_list[num].get("type") == 'B':
        sum += (fifty_tag5_list[num].get("tag5") * 50) / 33
    else:
        sum += (fifty_tag5_list[num].get("tag5") * 25) / 18
    num += 1
    if sum >= 28200:
        flag_over_282 = True
print(num)
for i in range(num):
    print(fifty_tag5_list[i].get("shop"))


# arima
x_list = [i + 1 for i in range(240)]
x2_list = [i + 1 + 240 for i in range(24)]

matrix_402x24 = []
p_list = []
q_list = []

for i in range(329, 330):
    print("..............................." + str(i) + "....................................................")
    s = df2.loc[(df['供应商ID'] == id_list[i])]
    s_list = np.array(s).tolist()[0][2:]
    t = []
    suum = 0
    for j in range(240):
        if s_list[j] != 0:
            t.append(s_list[j])
        suum += s_list[j]
    if suum / 240 > 1:
        dta = pd.Series(t)

        diff1 = dta.diff(1).dropna()
        (p, q) = (sm.tsa.arma_order_select_ic(dta, max_ar=6, max_ma=4, ic='bic')['bic_min_order'])
        try:
            model = sm.tsa.ARIMA(dta, order=(p, 1, q)).fit()

        except:
            model = sm.tsa.ARIMA(dta, order=(0, 1, 0)).fit()
            p = 0
            q = 0

        p_list.append(p)
        q_list.append(q)

        predict = model.predict(1,240)
        print(np.array(predict).tolist())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax = diff1.plot(ax=ax)
        predict.plot(ax=ax)

        plt.savefig('402img/' + np.array(s).tolist()[0][0])
        matrix_402x24.append(np.array(predict).tolist())
        # # Model evaluation
        # residuals = pd.DataFrame(model.resid)
        # fig, ax = plt.subplots(1, 2)
        # residuals.plot(title="Residuals", ax=ax[0])
        # residuals.plot(kind='kde', title='Density', ax=ax[1])
        # plt.plot()
        # plt.show()
        # print(residuals)
        #
        # s = pd.DataFrame(model.resid, columns=['value'])
        # u = s['value'].mean()  # calculate the average
        # print(u)
        # std = s['value'].std()  # calculate the standard deviation
        # print(std)
    else:
        matrix_402x24.append([0 for m in range(24)])

print(matrix_402x24)
print(p_list)
print(q_list)
