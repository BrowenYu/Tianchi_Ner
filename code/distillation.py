path = './round1_test/'

global category
category = {}


def fun(x):
    if x[3] == '\t':
        return x[4:]
    else:
        return x[3:]


def get_category(annX):
    for a in annX:
        a_ = a.split(' ')[0]
        if a_ not in category:
            category[a_] = 0
        category[a_] += 1


def check_error(annX, n):
    for a in annX:
        if a[0] == '\t':
            print(f'error:{a} in the {n}')


for num in range(0, 1000):

    print(num)
    print('*'*100)

    with open(f'./round1_test/submission_new_0/{num}.ann', 'r') as f:
        ann0 = f.readlines()
        print(ann0)
        ann0 = [fun(i) for i in ann0]
        # print(ann0)
        # check_error(ann0, 0)
        get_category(ann0)

    with open(f'./round1_test/submission_new_1/{num}.ann', 'r') as f:
        ann1 = f.readlines()
        ann1 = [fun(i) for i in ann1]
        # print(ann1)
        # check_error(ann1, 1)
        get_category(ann1)

    with open(f'./round1_test/submission_new_2/{num}.ann', 'r') as f:
        ann2 = f.readlines()
        ann2 = [fun(i) for i in ann2]
        # print(ann2)
        # check_error(ann2, 2)
        get_category(ann2)

    with open(f'./round1_test/submission_new_3/{num}.ann', 'r') as f:
        ann3 = f.readlines()
        ann3 = [fun(i) for i in ann3]
        # print(ann3)
        # check_error(ann3, 3)
        get_category(ann3)

    with open(f'./round1_test/submission_new_4/{num}.ann', 'r') as f:
        ann4 = f.readlines()
        ann4 = [fun(i) for i in ann4]
        # print(ann4)
        # check_error(ann4, 4)
        get_category(ann4)

    with open(f'./round1_test/submission_new_5/{num}.ann', 'r') as f:
        ann5 = f.readlines()
        ann5 = [fun(i) for i in ann5]
        # print(ann5)
        # check_error(ann5, 5)
        get_category(ann5)

    with open(f'./round1_test/submission_new_6/{num}.ann', 'r') as f:
        ann6 = f.readlines()
        ann6 = [fun(i) for i in ann6]
        # print(ann6)
        # check_error(ann6, 6)
        get_category(ann6)

    with open(f'./round1_test/submission_7/{num}.ann', 'r') as f:
        ann7 = f.readlines()
        ann7 = [fun(i) for i in ann7]
        # print(ann7)
        # check_error(ann7, 7)
        get_category(ann7)

    intersec = list(set(ann0) & set(ann1) & set(ann2) & set(ann3) & set(ann4) & set(ann5) & set(ann6) & set(ann7))

    with open(f'./round1_test/train/{num}.ann', 'r') as f:
        trainAnn = f.readlines()
        trainAnn = [fun(i) for i in trainAnn]
        trainAnn = list(filter(lambda x: x != '', trainAnn))

    for a in trainAnn:
        if (a not in ann0) and (a not in ann1) and (a not in ann2) and (a not in ann3) and (a not in ann4) and (
                a not in ann5) and (a not in ann6) and (a not in ann7):
            trainAnn.remove(a)
        for inters in list(set(intersec).difference(set(trainAnn))):
            trainAnn.append(inters)

    with open(f'./round1_test/train/{num}.ann', 'w') as f:
        f.truncate()

    with open(f'./round1_test/train/{num}.ann', 'a', encoding="utf-8") as f:

        n = 1
        for text in trainAnn:
            print(f'T{n}\t' + text)
            f.writelines(f'T{n}\t' + text)
            n += 1
print(category)
