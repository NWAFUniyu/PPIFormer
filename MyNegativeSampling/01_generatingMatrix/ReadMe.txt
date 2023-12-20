This step was quoted from DeNovo(doi: 10.1093/bioinformatics/btv737). 
In this step, Global alignment (GlobalAlign) was performed using the 
Needleman-Wunsch algorithm with the BLOSUM30 matrix to capture distant 
similarities.


BacteriaHost(Human).mat：与细菌相互作用的人类宿主蛋白列表矩阵
BacteriaHumanDistance30.mat：与细菌相互作用的人类宿主蛋白相似性矩阵



1：

for循环
if 这一行百分之八十的值都大于0.8,那么不要这一行