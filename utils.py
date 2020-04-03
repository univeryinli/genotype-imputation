from scipy.stats import pearsonr

def correct_rate(pre,groud_s):
    pre_s=torch.round(pre).type(torch.int8)
    statis = pre_s == groud_s
    return statis.sum().item() / statis.size()[0]


def r2_score(pre,groud_s):
    tot = groud_s.float().var()
    res = ((groud_s.float() - pre.float()).abs() ** 2).mean()
    return 1 - (res / tot)


def pearson(a,b):
    return pearsonr(a,b)[0]

