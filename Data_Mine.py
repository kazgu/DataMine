import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations,permutations
import random
from IPython.display import HTML, display

import os
current_path=os.path.abspath('.')

def apriori(coldt1,coldt2,SUPmin=0.2,CONFmin=0.2):
    user_items_dict={}
    for user,item in zip(coldt1,coldt2):
        if user not in user_items_dict:
            user_items_dict[user]=[item]
        else:
            user_items_dict[user]+=[item]
    T=[]
    for user in user_items_dict:
        print(user,user_items_dict[user])
        T.append(set(user_items_dict[user]))
    def get_candidates():
        candidate_itemset=set()
        for t in T:
            for k in range(1,len(t)+1):
                itemset=set(combinations(t,k))
                candidate_itemset=candidate_itemset.union(itemset)
        return candidate_itemset
    def get_support(X):
        X=set(X)
        support=0
        for t in T:
            if t>=X:
                support=support+1/len(T)
        return round(support,2)
    def get_supports(Xs):
        supports=[]
        for X in Xs:
            supports.append(get_support(X))

        print('\n',"*"*20,'候选子项集',"*"*20,'\n')
        
        for candidate,support in zip(Xs,supports):
            print(set(candidate),'支持度:',support)
        return supports
    
    def get_freq_itemset(candidates,supports):
        print('\n',"*"*20,'频繁项集',"*"*20,'\n')
        freq_itemsets=[]
        for candidate,support in zip(candidates,supports):
            if support>SUPmin:
                print(set(candidate),'支持度:',support)
                freq_itemsets.append(candidate)
        return freq_itemsets

    def get_rules(freq_itemsets):
        strong_rule=[]
        pre_rules=list(permutations(freq_itemsets,2))
        rule_supports=[]
        for rule in pre_rules:
            support=0
            front_itemset=set(rule[0])
            back_itemset=set(rule[1])
            for t in T:
                if t>=front_itemset and t>=back_itemset and not front_itemset&back_itemset:
                    support+=1/len(T)
            if support>0:
                rule_supports.append([rule,support])

        print('\n',"*"*20,'关联规则',"*"*20,'\n')
        for rule,support in rule_supports:
            sup_X_Y=round(support,2)
            sup_X=get_support(set(rule[0]))
            conf_X_Y=round(sup_X_Y/sup_X,2)
            if sup_X_Y>SUPmin and conf_X_Y>CONFmin: 
                print('强关联规则:','%s=>%s'%(set(rule[0]),set(rule[1])),'支持度:',sup_X_Y,'可信度',conf_X_Y)
                strong_rule.append([rule,sup_X_Y,conf_X_Y])
            else:
                print('%s=>%s'%(set(rule[0]),set(rule[1])),'支持度:',sup_X_Y,'可信度',conf_X_Y)
    candidates=get_candidates()
    supports=get_supports(candidates)
    freq_itemsets=get_freq_itemset(candidates,supports)
    get_rules(freq_itemsets)

def kmeans(dt1,dt2,k=3):
    dataset=[[d[0],d[1]] for d in zip(dt1,dt2)]
    center=[random.choice(dataset) for _ in range(k)]
    def get_dist(dataset,center):
        def distEclud(vecA, vecB):
            return np.sqrt(np.sum(np.power(np.array(vecA) - np.array(vecB), 2)))
        dists=[]
        for dd in dataset:
            dlist=[]
            for i,cc in enumerate(center):
                if cc is not None:
                    dist=distEclud(dd,cc)
                    print('dist(',dd,',',cc,')=',dist)
                    dlist.append(dist)
#             print(dlist)
            dists.append([dd,dlist])
        return dists

    def divide_to_cluster(dists,k):
        cluster=[[] for _ in range(k)]
        for dd,dist in dists:
            cid=dist.index(min(dist))
            cluster[cid].append(dd)
        print('划分到聚类中心')
        for i,c in enumerate(cluster):
            print(i+1,',',c)
        return cluster

    def get_new_center(cluster):
        new_center=[[] for _ in range(len(cluster))]
        for cid,c in enumerate(cluster):
            print(c)
            if c is not None and c:
                for t in range(len(c[0])):
                    tmp=[z[t] for z in c]
                    new_center[cid].append(np.mean(tmp))
        print('新聚类中心')
        for i,cc in enumerate(new_center):
            print('C%s'%(i+1),cc)
        return sorted(new_center)

    for i in range(100):
        print('*'*30,'第%s次迭代'%(i+1),'*'*30)
        #第一步
        print('第一步')
        plt.scatter(np.array(dataset)[:,0],np.array(dataset)[:,1])
        plt.scatter(np.array(center)[:,0],np.array(center)[:,1])
        plt.show()

        #第二步
        print('第二步')
        dists=get_dist(dataset,center)

        #第三步
        print('第三步')
        new_cluster=divide_to_cluster(dists,k)

        #第四步
        print('第四步')
        new_center=get_new_center(new_cluster)
        if center==new_center:
            break
        else:
            for ik in range(len(new_center)):
                if new_center[ik]:
                    center[ik]=new_center[ik]
                else:
                    center[ik]=random.choice(dataset)
                    
def decision_tree(data,prop_cols,classcol):                    
    def get_column_prurity(data,col,classcol):
        classvcounts=data[classcol].value_counts()
    #     print(classvcounts)
        def gini(cond):
            leng=len(data[data[col]==cond])
    #         print('leng',leng)
            psum=[]
            itp=[]
            for indx,val in zip(classvcounts.index,classvcounts):
                condset=data[(data[col]==cond) & (data[classcol]==indx)]
    #             print(len(condset))
                p=(len(condset)/leng)*(len(condset)/leng)
                itp.append([indx,round(p,2)])
                psum.append(p)
            return round(1-sum(psum),2),itp
        vcounts=data[col].value_counts()
        leng=len(data[col])
    #     print(vcounts)
        gini_dict={}
        prob_dict={}
        col_prurity=0
        for indx,val in zip(vcounts.index,vcounts):
            gini_dict[indx],itp=gini(indx)
            prob_dict[indx]=itp
            col_prurity+=(val/leng)*gini_dict[indx]
        col_prurity=round(col_prurity,2)
        print('%s_纯度'%col,col_prurity,gini_dict,prob_dict)
        return col_prurity,gini_dict,col,prob_dict

    def get_all_prurity(result,colnum,data,prop_cols,classcol):# 递归
        if len(prop_cols)<=0:
            return 0
        pcol=[get_column_prurity(data,p,classcol) for p in  prop_cols]
        pgini=[pc[0] for pc in pcol]
        minindex=pgini.index(min(pgini))
        padd='        '
        print('Node %s'%(pcol[minindex][2]))

        minpcol= sorted(pcol[minindex][1].items(),key=lambda x:x[1],reverse=True)
        result.append(pcol[minindex])
    #     print(minpcol[1],'        ',minpcol[0],flush=True,end='')

        get_all_prurity(result,colnum,data[data[prop_cols[minindex]]==minpcol[0][0]],[prop_cols[i] for i in range(len(prop_cols)) if i!=minindex],classcol)

    def to_graph(result):
        mark='ABCDEFGHIJK'
        i=0
        last=None
        last2=None
        output=[]
        for re in result:
            minpcol= sorted(re[1].items(),key=lambda x:x[1],reverse=True)
            leaf=sorted(re[3][minpcol[1][0]],key=lambda x:x[1],reverse=True) 
            fro=mark[i]
            i+=1
            to=mark[i]
            if last:
                output.append('%s -->|%s| %s[%s]'%(last,last2,fro,re[2]))
            output.append('%s[%s:%s] -->|%s| %s[%s:%s]'%(fro,re[2],re[0],minpcol[1][0],to,leaf[0][0],leaf[0][1]))
            last=fro
            last2=minpcol[0][0]
            i+=1
            if re==result[-1]:
                output.append('%s[%s:%s] -->|%s| %s[%s:%s]'%(fro,re[2],re[0],minpcol[0][0],mark[i],leaf[1][0],leaf[1][1]))

        html_code='''
        <html>
        <head>
        <title>决策树</title>
        </head>
          <body>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({startOnLoad:true});</script>

            <div class="mermaid" style="text-align:center;">
                graph TD
                %s
            </div>
          </body>
        </html>'''%('\n'.join(output).replace('(','_').replace(')','_'))
        with open('decision_tree.html','w',encoding='utf-8') as f:
            f.write(html_code.strip())
        display(HTML('<h2>结果:file:///%s/decision_tree.html</h2>'%(current_path)))
    result=[]
    get_all_prurity(result,len(prop_cols),data,prop_cols,classcol)
    to_graph(result)

def to_quantify(df,cols,cond=None):
    for col in df.columns:
        if col not in cols:
            continue
        if cond and col in cond:
            df[col]=df[col].apply(cond[col])
        else:
            qt={t:i for i,t in enumerate(df[col].value_counts().index)}
            print(qt)
            df[col]=df[col].apply(lambda x: qt[x])
    return df
                    
