import numpy as np
from sklearn.decomposition import FastICA
import processing

def fastICA(video):
    blist = []
    for frame in video:
        blist.append(processing.Imeanb(frame))
    glist = []
    for frame in video:
        glist.append(processing.Imeang(frame))
    rlist = []
    for frame in video:
        rlist.append(processing.Imeanr(frame))

    #中心化
    s = np.c_[blist, glist, rlist]
    s /= s.std(axis=0)
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(s)
    S = S_.T

    # 利用皮尔森常数找到最优分量,与绿色通道最相近的独立成分
    mubiao = S[0, :]
    X = np.vstack([glist, S[0, :]])
    d1 = np.corrcoef(X)[0][1]
    Y = np.vstack([glist, S[1, :]])
    d2 = np.corrcoef(Y)[0][1]
    Z = np.vstack([glist, S[2, :]])
    d3 = np.corrcoef(Z)[0][1]
    if d1 < d2:
        mubiao = S[1, :]
        if d2 < d3:
            mubiao = S[2, :]

    return mubiao


def fastICA1(video):
    blist = []
    for frame in video:
        blist.append(processing.Imeanb(frame))
    glist = []
    for frame in video:
        glist.append(processing.Imeang(frame))
    rlist = []
    for frame in video:
        rlist.append(processing.Imeanr(frame))
    templist = []
    for frame in video:
        templist.append(processing.Imeang(frame) - processing.Imeanr(frame) )
    x=x = np.linspace(1, len(glist), len(glist))
    #中心化
    s = np.c_[blist, glist, rlist]
    s /= s.std(axis=0)
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(s)
    S = S_.T
    # 利用皮尔森常数找到最优分量
    mubiao = S[0, :]
    X = np.vstack([templist, S[0, :]])
    d1 = np.corrcoef(X)[0][1]
    Y = np.vstack([templist, S[1, :]])
    d2 = np.corrcoef(Y)[0][1]
    Z = np.vstack([templist, S[2, :]])
    d3 = np.corrcoef(Z)[0][1]
    if d1 < d2:
        mubiao = S[1, :]
        if d2 < d3:
            mubiao = S[2, :]
    # 与绿色通道最相近的独立成分

    return mubiao
