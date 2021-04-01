# coding='utf-8'
from math import log 
import copy
import matplotlib.pyplot as plt
import TreePlot

#  ====================数据初始化====================

labels_word = ['色泽','根蒂','敲声','纹理','脐部','触感','密度']
labels_lisan = [0,1,2,3,4,5] # '色泽','根蒂','敲声','纹理','脐部','触感' 离散变量放这里，对应数据集中的位置
M = labels_lisan[-1] 

labels_lianxu = [6] # 连续变量放这里

label_value = [     # 各属性的取值，连续变量只用高和低表示
    ['青绿','乌黑','浅白'],
    ['蜷缩','稍蜷','硬挺'],
    ['浊响','沉闷','清脆'],
    ['清晰','稍糊','模糊'],
    ['凹陷','稍凹','平坦'],
    ['硬滑','软粘'],
    ['高','低'] 
]
dataSet = [
    ['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,'好瓜'],
    ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑',0.774,'好瓜'],
    ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑',0.634,'好瓜'],
    ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑',0.608,'好瓜'],
    ['浅白','蜷缩','浊响','清晰','凹陷','硬滑',0.556,'好瓜'],
    ['青绿','稍蜷','浊响','清晰','稍凹','软粘',0.403,'好瓜'],
    ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘',0.481,'好瓜'],
    ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑',0.437,'好瓜'],
    ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑',0.666,'坏瓜'],
    ['青绿','硬挺','清脆','清晰','平坦','软粘',0.243,'坏瓜'],
    ['浅白','硬挺','清脆','模糊','平坦','硬滑',0.245,'坏瓜'],
    ['浅白','蜷缩','浊响','模糊','平坦','软粘',0.343,'坏瓜'],
    ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑',0.639,'坏瓜'],
    ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑',0.657,'坏瓜'],
    ['乌黑','稍蜷','浊响','清晰','稍凹','软粘',0.360,'坏瓜'],
    ['浅白','蜷缩','浊响','模糊','平坦','硬滑',0.593,'坏瓜']
   # ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑',0.719,'坏瓜']  
]

#  ====================计算决策树====================

def information_entropy(data): # 计算信息熵
    num = len(data)  # 样本数
    label_yes = 0 # 好瓜数
    for current_data in data: # 遍历整个数据集，每次取一行
        if current_data[-1] == '好瓜': label_yes+=1
       
    Ent = 0.0  # 初始化信息熵   
    prob = float(label_yes/num)

    if prob == 0 or prob == 1:
        Ent = 0
    else:
        Ent = -1*prob * log(prob,2) -1*(1-prob)*log((1-prob),2) # 计算信息熵

    return Ent

def information_gain(data,n): # 计算信息增益
    num = len(label_value[n]) # 某个标签有几个值
    son_Ent = 0.0
    for i in range(num):
        son_data = []       
        for j in data:
            if j[n] == label_value[n][i]:
                son_data.append(j)

        if len(son_data) == 0:
            son_Ent += 0
        else:
            son_Ent += information_entropy(son_data)*len(son_data)/len(data)

    Gain =  information_entropy(data)-son_Ent
    return Gain

def best_gain(data,label): # 计算最优特征值
    maxgain = 0.0
    feature = -1    # 最优划分特征
    for i in label:
        nowgain = information_gain(data,i)
        if information_gain(data,i)>maxgain:
            maxgain = nowgain
            feature = i
    
    return maxgain,feature      


def leaf(data): # 返回叶节点
    li = [x[-1] for x in data]
    return max(li,key=li.count)

def continuing_value(data,label_lianxu): # 连续性变量处理
    data_test = copy.deepcopy(data)
    maxgain = 0.0
    maxfeature = 0
    bestsplite = 0.0
    for k in label_lianxu:
        gain = []
        density_list = []
        density_list1 = [x[k] for x in data]
        density_list1.sort(reverse=False)
        for i in range(len(data)-1): 
            density = (density_list1[i]+density_list1[i+1])/2
            for j in range(len(data)):
                if data[j][k]>=density: 
                    data_test[j][k]='高'
                else:
                    data_test[j][k]='低'

            gain.append(information_gain(data_test,k))
            density_list.append(density)

        if max(gain) > maxgain:
            maxgain = max(gain)
            bestsplite = density_list[gain.index(max(gain))]
            maxfeature = k

    return maxgain,bestsplite,maxfeature

def CreateTree(data,label_lisan,label_lianxu): # 计算决策树的主代码
    global labels_word,M
    classList = [example[-1] for example in data] # 返回当前数据集下标签列所有值
    if classList.count(classList[0]) == len(classList):
        return classList[0]   #当类别完全相同时则停止继续划分，直接返回该类的标签
    
    if len(label_lisan) == 0 and len(label_lianxu) == 0: # 标签为空
        return leaf(data)

    gain1,feature1 = best_gain(data,label_lisan)
    gain2,splite,feature2 = continuing_value(data,label_lianxu)
    if gain1>=gain2:
        feature = feature1
        label_lisan.remove(feature)
    else:
        feature = feature2

    Tree = {labels_word[feature]:{}} # 当前数据集选取最好的特征存储在bestFeat中
    num = len(label_value[feature]) # 某个标签有几个值
    for i in range(num):
        son_data = []

        if feature > M:  #labels_lisan[-1]:
            for j in data:
                if j[feature] >= splite and i==0: 
                    son_data.append(j)
                    if len(son_data) == 0:
                        return leaf(data)
                    else:
                        Tree[labels_word[feature]][str('>='+str(splite))] = CreateTree(son_data,label_lisan,label_lianxu)

                elif j[feature] < splite and i==1: 
                    son_data.append(j)
                    if len(son_data) == 0:
                        return leaf(data)
                    else:
                        Tree[labels_word[feature]][str('<'+str(splite))] = CreateTree(son_data,label_lisan,label_lianxu)

        else:
            for j in data:
                if j[feature] == label_value[feature][i]:son_data.append(j)

            if len(son_data) == 0:
                return leaf(data)

            else:
                Tree[labels_word[feature]][label_value[feature][i]] = CreateTree(son_data,label_lisan,labels_lianxu)

    return Tree


if __name__ == '__main__':  
    Tree = CreateTree(dataSet,labels_lisan,labels_lianxu)
    TreePlot.createPlot(Tree)


        
    