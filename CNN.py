#import library######################################################################
import torch
import torch.nn as nn # 신경망들이 포함됨
import torch.optim as optim # 최적화 알고리즘들이 포함됨
import torch.nn.init as init # 텐서에 초기값을 줌

import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴

import numpy as np
import matplotlib.pyplot as plt


#set hyper parameter#################################################################
batch_size = 100 #한 epoch를 batch_size로 나누어 실행하는 횟수 = iteration
learning_rate = 0.0002 #기울기가 0점으로 이동하는 크기 (학습진행속도)
num_epoch = 10 #1에폭 = forward/backward 과정 1번 거침. #전체 데이터셋을 몇 번 볼까~

#load MNIST Data
mnist_train = datasets.MNIST(root="../Data/", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = datasets.MNIST(root="../Data/", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
"""
●train : 지금 정의하는게 학습용인지, 테스트용인지 정의
●transform : 데이터에 어떠한 변형을 줄건지 정의
    transforms.ToTensor() : 모델에 넣어주기 위해 텐서 변환해줌
●target_transform : 라벨(클래스)에 어떠한 변형을 줄 것인가 정의
●download : 앞에 지정해준 경로에 해당 데이터가 없을 시 다운로드 하도록 정의
"""

#Define Loaders#########################################################################
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

"""
학습 및 배치로 모델에 넣어주기 위한 툴. 앞에서 정의한 데이터셋을 DataLoader에 넣어주면, 
우리가 정의해준 조건에 맞게 보델을 Train, Inference할때 데이터를 Load해주게 됨. 
●shuffle : 데이터 섞어줄지 지정하는 파라미터
●num_workers : 데이터 묶을때 사용할 프로세서의 개수
●drop_last : 묶고 남은 데이터를 버릴것인가 지정하는 파라미터
"""

#Define CNN(Base) Model###################################################################
class CNN(nn.Module):
    def __init__(self):
        #super는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            # [100,1,28,28] -> [100,16,24,24]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),

            # [100,16,24,24] -> [100,32,20,20]
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5),
            nn.ReLU(),

            # [100,32,20,20] -> [100,32,10,10]
            nn.MaxPool2d(kernel_size=2,stride=2),

                  # [100,32,10,10] -> [100,64,6,6]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            
            # [100,64,6,6] -> [100,64,3,3]
            nn.MaxPool2d(kernel_size=2,stride=2)          
        )
        self.fc_layer = nn.Sequenfial(
            # [100,64*3*3] -> [100,100]
            nn.Linear(64*3*3,100),                                              
            nn.ReLU(),
            # [100,100] -> [100,10]
            nn.Linear(100,10)  
        )

    def forward(self,x):
        #self.layer에서 정의한 연산 수행
        out = self.layer(x)
        #view함수를 이용해서 텐서의 형태를 [100,나머지]로 변환
        out = out.view(batch_size, -1)
        #self.fc_layer 정의한 연산 수행
        out = self.fc_layer(out)
        return out
#device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device) #정의한 모델 객체를 선언후, device에 올려주기

#모델 학습용 손실, 최적화 함수 정의. 후에 Adam으로 감사줌으로써 
#모델의 파라미터들을 사전에 정의한 learning_rate로 업데이트
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train Model########################################################################
