import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 参数设置
labLen = 419
unLen = 9424
MAXEDGES = 3000000
ClassNum = 2
THRESHOLD = 0.5
spamclass = 1
DD = 1
LL = 0
MAXITTOTAL = 10

#读取用户特征
user_features = {}  # user_no(int) -> [10个float]
with open("Data/UserFeature.txt") as f:
    for idx, line in enumerate(f, 1):  # idx 从1开始
        feats = list(map(float, line.strip().split()))
        user_features[idx] = feats

#特征归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
all_feats = np.array([user_features[i] for i in range(1, unLen+1)])
scaler.fit(all_feats)
for i in range(1, unLen+1):
    user_features[i] = scaler.transform([user_features[i]])[0]

class UPM:
    def __init__(self):
        self.uID = 0
        self.shill = -10
        self.tempLab = -10
        self.neighbors = []
        self.neighborsNum = 0
        self.pTheta = [0.0] * ClassNum
        self.pFinalLabelWeight = [0.0] * ClassNum
        self.z_jk = []  # shape: [neighborsNum][ClassNum]

class LRUP:
    def __init__(self):
        self.tmpLab = 0
        self.Data_Index = 0
        self.w_jk_i = 0.0

LabUPM = [UPM() for _ in range(labLen)]
UnUPM = [UPM() for _ in range(unLen+2)]  # 1-based
datasetForLR = [LRUP() for _ in range(MAXEDGES+2)]
UnLabels = [-10] * (unLen+2)
Train_Index = [0] * labLen
Train_Label = [0] * labLen
Predit_Pro = [0.0] * (unLen+2)
alpha_k = [0.0] * ClassNum

def initialization():
    with open("Data/Training_Testing/5percent/train_4.csv") as fin1, \
         open("Data/Training_Testing/5percent/test_4.csv") as fin4, \
         open("jaccard0.2.txt") as fin3:
        for i in range(1, unLen+1):
            UnUPM[i].shill = -10
            UnLabels[i] = -10

        i = 0
        for line in fin1:
            userID, labelID = map(int, line.strip().split())
            UnUPM[userID].shill = labelID
            UnUPM[userID].tempLab = labelID
            UnUPM[userID].uID = userID
            UnLabels[userID] = labelID
            Train_Index[i] = userID
            Train_Label[i] = labelID
            i += 1

        for line in fin4:
            userID, labelID = map(int, line.strip().split())
            UnUPM[userID].shill = -1
            UnUPM[userID].tempLab = -1
            UnUPM[userID].uID = userID
            UnLabels[userID] = labelID

        tempUserID = 1
        cNeighbors = 0
        for line in fin3:
            userID, neighbor = map(int, line.strip().split())
            if tempUserID == userID:
                UnUPM[tempUserID].neighbors.append(neighbor)
                cNeighbors += 1
            else:
                UnUPM[tempUserID].neighborsNum = cNeighbors
                tempUserID = userID
                cNeighbors = 1
                UnUPM[tempUserID].neighbors = [neighbor]
        UnUPM[userID].neighborsNum = cNeighbors

        # 统计信息
        sum_edges = N0 = N1 = N00 = N11 = 0
        for i in range(1, unLen+1):
            UnUPM[i].neighborsNum = len(UnUPM[i].neighbors)
            sum_edges += UnUPM[i].neighborsNum
            if UnLabels[i] == 1:
                N1 += UnUPM[i].neighborsNum
            elif UnLabels[i] == 0:
                N0 += UnUPM[i].neighborsNum
            for j in range(UnUPM[i].neighborsNum):
                neighbor_id = UnUPM[i].neighbors[j]
                if UnLabels[i] == 1 and UnLabels[neighbor_id] == 1:
                    N11 += 1
                if UnLabels[i] == 0 and UnLabels[neighbor_id] == 0:
                    N00 += 1
        print(f"Number of Edges in Network: {sum_edges} Total Edges of Normal:{N0} Total Edges of Spammer:{N1}")
        print(f"Number of Spam-Spam Edges in Network: {N11} Number of Normal-Normal Edges in Network:{N00}")
        print(f"Purity1: {N11/N1 if N1 else 0}  Purity0:{N00/N0 if N0 else 0}")

def python_lr_invoke(Index, pLabel, iWeight, length):
    X = np.array([user_features[i] for i in Index])
    y = np.array(pLabel)
    sample_weight = np.array(iWeight)
    lr = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, class_weight='balanced', random_state=0)
    lr.fit(X, y, sample_weight=sample_weight)
    X_pred = np.array([user_features[i] for i in range(1, unLen+1)])
    probs = lr.predict_proba(X_pred)[:, 1]
    for i in range(1, unLen+1):
        Predit_Pro[i] = probs[i-1]
    return 1

def LossFunction(lambda_, d):
    Loss = 0
    for i in range(1, unLen+1):
        P1 = max(UnUPM[i].pTheta[1], 1e-6)
        P0 = max(UnUPM[i].pTheta[0], 1e-6)
        if UnUPM[i].shill == -1 and UnUPM[i].tempLab == 1:
            Loss -= lambda_ * math.log(P1)
        elif UnUPM[i].shill == 1:
            Loss -= math.log(P1)
        elif UnUPM[i].shill == -1 and UnUPM[i].tempLab == 0:
            Loss -= lambda_ * math.log(P0)
        elif UnUPM[i].shill == 0:
            Loss -= math.log(P0)
        if UnUPM[i].shill != -10 and UnUPM[i].neighborsNum > 0:
            ww = 1
            if UnUPM[i].shill == -1:
                ww = lambda_
            PP = 0
            if UnUPM[i].shill == 1 or UnUPM[i].tempLab == 1:
                for j in range(UnUPM[i].neighborsNum):
                    neighbor = UnUPM[i].neighbors[j]
                    if UnUPM[neighbor].shill == 1:
                        PP += math.log(alpha_k[1])
                    elif UnUPM[neighbor].shill == 0:
                        PP += math.log(alpha_k[0])
                    elif UnUPM[neighbor].shill == -1:
                        NP1 = max(UnUPM[neighbor].pTheta[1], 1e-6)
                        NP0 = max(UnUPM[neighbor].pTheta[0], 1e-6)
                        if UnUPM[neighbor].tempLab == 1:
                            PP += math.log(alpha_k[1] * NP1)
                        else:
                            PP += math.log(alpha_k[0] * NP0)
            if UnUPM[i].shill == 0 or UnUPM[i].tempLab == 0:
                for j in range(UnUPM[i].neighborsNum):
                    neighbor = UnUPM[i].neighbors[j]
                    if UnUPM[neighbor].shill == 1:
                        PP += math.log(1 - alpha_k[1])
                    elif UnUPM[neighbor].shill == 0:
                        PP += math.log(1 - alpha_k[0])
                    elif UnUPM[neighbor].shill == -1:
                        NP1 = max(UnUPM[neighbor].pTheta[1], 1e-6)
                        NP0 = max(UnUPM[neighbor].pTheta[0], 1e-6)
                        if UnUPM[neighbor].tempLab == 1:
                            PP += math.log((1 - alpha_k[1]) * NP1)
                        else:
                            PP += math.log((1 - alpha_k[0]) * NP0)
            Loss -= (d * ww / UnUPM[i].neighborsNum) * PP
    print("Go out loss function....")
    return Loss

def InitClassifier(lambda_):
    iWeight = [1.0] * labLen
    python_lr_invoke(Train_Index, Train_Label, iWeight, labLen)
    clsFriendsNum = [0.0] * ClassNum
    for i in range(1, unLen+1):
        if UnUPM[i].shill == 1:
            UnUPM[i].pTheta[1] = 1
            UnUPM[i].pTheta[0] = 0
            UnUPM[i].pFinalLabelWeight[0] = 0
            UnUPM[i].pFinalLabelWeight[1] = 1
        elif UnUPM[i].shill == 0:
            UnUPM[i].pTheta[1] = 0
            UnUPM[i].pTheta[0] = 1
            UnUPM[i].pFinalLabelWeight[0] = 1
            UnUPM[i].pFinalLabelWeight[1] = 0
        elif UnUPM[i].shill == -1:
            UnUPM[i].pTheta[1] = Predit_Pro[i]
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1]
            UnUPM[i].pFinalLabelWeight[0] = UnUPM[i].pTheta[0]
            UnUPM[i].pFinalLabelWeight[1] = UnUPM[i].pTheta[1]
        if UnUPM[i].pTheta[1] >= THRESHOLD:
            UnUPM[i].tempLab = 1
        else:
            UnUPM[i].tempLab = 0
    for i in range(1, unLen+1):
        if UnUPM[i].shill != -10 and UnUPM[i].neighborsNum > 0:
            for j in range(UnUPM[i].neighborsNum):
                neighbor = UnUPM[i].neighbors[j]
                if UnUPM[i].shill == 1:
                    if UnUPM[neighbor].shill == 1 or (UnUPM[neighbor].shill == -1 and UnUPM[neighbor].tempLab == 1):
                        clsFriendsNum[1] += 1
                    if UnUPM[neighbor].shill == 0 or (UnUPM[neighbor].shill == -1 and UnUPM[neighbor].tempLab == 0):
                        clsFriendsNum[0] += 1
    tp = fn = fp = tn = 0
    for i in range(1, unLen+1):
        if UnUPM[i].shill == -1:
            if UnUPM[i].tempLab == 1 and UnLabels[i] == 1:
                tp += 1
            if UnUPM[i].tempLab == 0 and UnLabels[i] == 1:
                fn += 1
            if UnUPM[i].tempLab == 1 and UnLabels[i] == 0:
                fp += 1
            if UnUPM[i].tempLab == 0 and UnLabels[i] == 0:
                tn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f = 2 * recall * precision / (recall + precision) if (recall + precision) else 0
    print(f"tp = {tp}")
    print(f"fn = {fn}")
    print(f"fp = {fp}")
    print(f"tn = {tn}")
    print(f"RECALL = {recall}")
    print(f"PRECISION = {precision}")
    print(f"F-MEASURE = {f}")
    sum_cls = clsFriendsNum[0] + clsFriendsNum[1]
    for k in range(ClassNum):
        alpha_k[k] = clsFriendsNum[k] / sum_cls if sum_cls else 0
        print(f"alpha[{k}] = {alpha_k[k]}")
    Loss1 = LossFunction(lambda_, DD)
    print(f"IntiLR Loss: {Loss1}")

def ComputeAlphaK(lambda_, d):
    allFriendsNum = [0.0] * ClassNum
    clsFriendsNum = [0.0] * ClassNum
    for k in range(ClassNum):
        clsFriendsNum[k] = 0
        allFriendsNum[k] = 0
    for i in range(1, unLen+1):
        if UnUPM[i].shill != -10 and UnUPM[i].neighborsNum > 0:
            for j in range(UnUPM[i].neighborsNum):
                if UnUPM[i].shill == 1 or UnUPM[i].shill == 0:
                    if UnUPM[i].shill == 1:
                        clsFriendsNum[1] += UnUPM[i].z_jk[j][1] / UnUPM[i].neighborsNum
                        clsFriendsNum[0] += UnUPM[i].z_jk[j][0] / UnUPM[i].neighborsNum
                        allFriendsNum[0] += UnUPM[i].z_jk[j][0] / UnUPM[i].neighborsNum
                        allFriendsNum[1] += UnUPM[i].z_jk[j][1] / UnUPM[i].neighborsNum
                    else:
                        allFriendsNum[0] += UnUPM[i].z_jk[j][0] / UnUPM[i].neighborsNum
                        allFriendsNum[1] += UnUPM[i].z_jk[j][1] / UnUPM[i].neighborsNum
                else:
                    if UnUPM[UnUPM[i].neighbors[j]].tempLab == 1:
                        clsFriendsNum[1] += lambda_ * UnUPM[i].z_jk[j][1] / UnUPM[i].neighborsNum
                        clsFriendsNum[0] += lambda_ * UnUPM[i].z_jk[j][0] / UnUPM[i].neighborsNum
                        allFriendsNum[0] += lambda_ * UnUPM[i].z_jk[j][0] / UnUPM[i].neighborsNum
                        allFriendsNum[1] += lambda_ * UnUPM[i].z_jk[j][1] / UnUPM[i].neighborsNum
                    else:
                        allFriendsNum[0] += lambda_ * UnUPM[i].z_jk[j][0] / UnUPM[i].neighborsNum
                        allFriendsNum[1] += lambda_ * UnUPM[i].z_jk[j][1] / UnUPM[i].neighborsNum
    for k in range(ClassNum):
        alpha_k[k] = clsFriendsNum[k] / allFriendsNum[k] if allFriendsNum[k] else 0
        print(f"alpha[{k}] = {alpha_k[k]}")

def EStep_Crisp(lambda_, d):
    print("Go into EStep....")
    for i in range(1, unLen+1):
        if UnUPM[i].shill != -10:
            UnUPM[i].z_jk = [[0.0 for _ in range(ClassNum)] for _ in range(UnUPM[i].neighborsNum)]
            for j in range(UnUPM[i].neighborsNum):
                neighbor = UnUPM[i].neighbors[j]
                if UnUPM[neighbor].shill == 0:
                    UnUPM[i].z_jk[j][0] = 1
                    UnUPM[i].z_jk[j][1] = 0
                elif UnUPM[neighbor].shill == 1:
                    UnUPM[i].z_jk[j][1] = 1
                    UnUPM[i].z_jk[j][0] = 0
                elif UnUPM[neighbor].shill == -1:
                    if UnUPM[i].shill == 0 or UnUPM[i].tempLab == 0:
                        sum_ = (1 - alpha_k[0]) * UnUPM[neighbor].pTheta[0] + (1 - alpha_k[1]) * UnUPM[neighbor].pTheta[1]
                        for k in range(ClassNum):
                            UnUPM[i].z_jk[j][k] = ((1 - alpha_k[k]) * UnUPM[neighbor].pTheta[k]) / sum_ if sum_ else 0
                    elif UnUPM[i].shill == 1 or UnUPM[i].tempLab == 1:
                        sum_ = alpha_k[0] * UnUPM[neighbor].pTheta[0] + alpha_k[1] * UnUPM[neighbor].pTheta[1]
                        for k in range(ClassNum):
                            UnUPM[i].z_jk[j][k] = (alpha_k[k] * UnUPM[neighbor].pTheta[k]) / sum_ if sum_ else 0
    ComputeAlphaK(lambda_, d)

def IterLogReg(lambda_, d):
    print("Start Generating New Dataset....")
    cc = 1
    for i in range(1, unLen+1):
        if UnUPM[i].shill != -10:
            if UnUPM[i].shill == -1 and lambda_ > 0:
                datasetForLR[cc].w_jk_i = lambda_
            elif UnUPM[i].shill == 0 or UnUPM[i].shill == 1:
                datasetForLR[cc].w_jk_i = 1
            datasetForLR[cc].tmpLab = UnUPM[i].tempLab
            datasetForLR[cc].Data_Index = UnUPM[i].uID
            cc += 1
    print(f"After Inserting Nodes ITSELVES, Data Size: {cc}")
    DD_ = d
    for i in range(1, unLen+1):
        if UnUPM[i].shill != -10 and UnUPM[i].neighborsNum > 0 and (UnUPM[i].pTheta[1] >= 0.9 or UnUPM[i].pTheta[1] <= 0.01):
            for cl in range(UnUPM[i].neighborsNum):
                neighbor = UnUPM[i].neighbors[cl]
                if UnUPM[neighbor].shill == -1:
                    if UnUPM[i].shill == 0 or UnUPM[i].shill == 1:
                        for k in range(ClassNum):
                            datasetForLR[cc].w_jk_i = DD_ * UnUPM[i].z_jk[cl][k] / UnUPM[i].neighborsNum
                            datasetForLR[cc].tmpLab = k
                            datasetForLR[cc].Data_Index = UnUPM[neighbor].uID
                            cc += 1
                    else:
                        for k in range(ClassNum):
                            datasetForLR[cc].w_jk_i = lambda_ * DD_ * UnUPM[i].z_jk[cl][k] / UnUPM[i].neighborsNum
                            datasetForLR[cc].tmpLab = k
                            datasetForLR[cc].Data_Index = UnUPM[neighbor].uID
                            cc += 1
                if UnUPM[neighbor].shill == 0:
                    if UnUPM[i].shill == 0 or UnUPM[i].shill == 1:
                        datasetForLR[cc].w_jk_i = DD_ / UnUPM[i].neighborsNum
                        datasetForLR[cc].tmpLab = 0
                        datasetForLR[cc].Data_Index = UnUPM[neighbor].uID
                        cc += 1
                    else:
                        datasetForLR[cc].w_jk_i = lambda_ * DD_ / UnUPM[i].neighborsNum
                        datasetForLR[cc].tmpLab = 0
                        datasetForLR[cc].Data_Index = UnUPM[neighbor].uID
                        cc += 1
                if UnUPM[neighbor].shill == 1:
                    if UnUPM[i].shill == 0 or UnUPM[i].shill == 1:
                        datasetForLR[cc].w_jk_i = DD_ / UnUPM[i].neighborsNum
                        datasetForLR[cc].tmpLab = 1
                        datasetForLR[cc].Data_Index = UnUPM[neighbor].uID
                        cc += 1
                    else:
                        datasetForLR[cc].w_jk_i = lambda_ * DD_ / UnUPM[i].neighborsNum
                        datasetForLR[cc].tmpLab = 1
                        datasetForLR[cc].Data_Index = UnUPM[neighbor].uID
                        cc += 1
    print(f"Size of New DATASETS: {cc-1}")
    tTrainIndex = [datasetForLR[i+1].Data_Index for i in range(cc-1)]
    tTrainLabel = [datasetForLR[i+1].tmpLab for i in range(cc-1)]
    tIWeight = [datasetForLR[i+1].w_jk_i for i in range(cc-1)]
    python_lr_invoke(tTrainIndex, tTrainLabel, tIWeight, cc-1)

def MStep(lambda_, d):
    IterLogReg(lambda_, d)
    for i in range(1, unLen+1):
        if UnUPM[i].shill == 1:
            UnUPM[i].pTheta[1] = 1
            UnUPM[i].pTheta[0] = 0
        elif UnUPM[i].shill == 0:
            UnUPM[i].pTheta[1] = 0
            UnUPM[i].pTheta[0] = 1
        elif UnUPM[i].shill == -1:
            UnUPM[i].pTheta[1] = Predit_Pro[i]
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1]
            if UnUPM[i].neighborsNum > 10 and UnUPM[i].pTheta[1] < 0.4:
                UnUPM[i].pTheta[1] += 0.2
                UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1]
    for i in range(1, unLen+1):
        if UnUPM[i].shill == -1 and UnUPM[i].neighborsNum > 0:
            mul = [UnUPM[i].pTheta[k] if UnUPM[i].pTheta[k] != 0 else 0.0001 for k in range(ClassNum)]
            neighW = [1.0] * ClassNum
            for j in range(UnUPM[i].neighborsNum):
                neighbor = UnUPM[i].neighbors[j]
                if UnUPM[neighbor].shill == -1:
                    tempSum0 = (1 - alpha_k[0]) * UnUPM[neighbor].pTheta[0] + (1 - alpha_k[1]) * UnUPM[neighbor].pTheta[1]
                    tempSum0 = pow(tempSum0, d / UnUPM[i].neighborsNum)
                    neighW[0] *= tempSum0
                    tempSum1 = alpha_k[0] * UnUPM[neighbor].pTheta[0] + alpha_k[1] * UnUPM[neighbor].pTheta[1]
                    tempSum1 = pow(tempSum1, d / UnUPM[i].neighborsNum)
                    neighW[1] *= tempSum1
                if UnUPM[neighbor].shill == 0:
                    neighW[0] *= pow(1 - alpha_k[0], d / UnUPM[i].neighborsNum)
                    neighW[1] *= pow(alpha_k[0], d / UnUPM[i].neighborsNum)
                if UnUPM[neighbor].shill == 1:
                    neighW[1] *= pow(alpha_k[1], d / UnUPM[i].neighborsNum)
                    neighW[0] *= pow(1 - alpha_k[1], d / UnUPM[i].neighborsNum)
            for k in range(ClassNum):
                mul[k] *= neighW[k]
            sum_ = mul[0] + mul[1]
            mul[0] = mul[0] / sum_ if sum_ else 0
            mul[1] = mul[1] / sum_ if sum_ else 0
            for k in range(ClassNum):
                UnUPM[i].pTheta[k] = mul[k]
                UnUPM[i].pFinalLabelWeight[k] = mul[k]
            if mul[1] >= THRESHOLD:
                UnUPM[i].tempLab = 1
            else:
                UnUPM[i].tempLab = 0
    Loss = LossFunction(lambda_, d)
    tp = fn = fp = tn = 0
    for i in range(1, unLen+1):
        if UnUPM[i].shill == -1:
            if UnUPM[i].tempLab == 1 and UnLabels[i] == 1:
                tp += 1
            if UnUPM[i].tempLab == 0 and UnLabels[i] == 1:
                fn += 1
            if UnUPM[i].tempLab == 1 and UnLabels[i] == 0:
                fp += 1
            if UnUPM[i].tempLab == 0 and UnLabels[i] == 0:
                tn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f = 2 * recall * precision / (recall + precision) if (recall + precision) else 0
    print(f"tp = {tp}, Spammer to Spammer")
    print(f"fn = {fn}, Spammer to Normal")
    print(f"fp = {fp}, Normal to Spammer")
    print(f"tn = {tn}, Normal to Normal")
    print(f"RECALL = {recall}")
    print(f"PRECISION = {precision}")
    print(f"F-MEASURE = {f}")
    return Loss

def Output_to_File(k):
    file = f"our_output_files/AmazonLRF{k}"
    with open(file, "w") as fout2:
        for i in range(1, unLen+1):
            if UnUPM[i].shill == -1:
                fout2.write(f"{i},{UnUPM[i].pTheta[1]}\n")

def control(lambda_, d):
    initialization()
    InitClassifier(lambda_)
    IterNum = 0
    while IterNum < MAXITTOTAL:
        EStep_Crisp(lambda_, d)
        Loss = MStep(lambda_, d)
        IterNum += 1
        print(f"Iteration Number: {IterNum}, Loss: {Loss}")
        Output_to_File(IterNum)

if __name__ == "__main__":
    control(LL, DD)