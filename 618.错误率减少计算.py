'''
2022年 05月 08日 星期日 13:59:24 CST
Ding Ning
'''
while(1):
    base = float(input("base:"))
    ours = float(input("ours:"))
    # import ipdb;ipdb.set_trace()
    relative_error = ((100-base)-(100-ours)) / (100-base)
    relative_error = round((100*relative_error),1)
    print(relative_error)

