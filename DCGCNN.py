import torch
import numpy as np
import torch.nn as nn


class GCNN(nn.Module):

    def __init__(self, inputSize, dataSize, classSize, trainData,classData, bSize):
        super(GCNN, self).__init__()
        self.trainingData = torch.Tensor(dataSize, inputSize).float().cuda()
        self.classOfData = torch.IntTensor(dataSize).float().cuda()

        self.ymax = torch.Tensor(1).fill_(0.9).cuda()
        self.divergeVals = torch.Tensor(dataSize,classSize).cuda()
        self.trainingData = trainData.clone().float().cuda()
        self.classOfData = classData.clone().int()

        self.variance = torch.Tensor(1).fill_(1).cuda()
        self.bsize = bSize
        #self.vectorvar = torch.Tensor(bSize,dataSize).fill_(1).cuda()
        self.classSize = classSize
        self.dataSize = dataSize
        self.inputSize = inputSize
        self.gradVar = 0
        self.varUpRate = 0
        self.isVectorVar = 0
        self.debug = 0
        self.denominator = torch.zeros(bSize,classSize, requires_grad=False).cuda()
        #self.vectorvar = vvals.float().cuda()
        #print(self.vectorvar)
        self.calculateClassNorm()
        self.calculateDiverge()
        self.loadVariance()
        

    def forward(self, x):
        
        self.output = torch.Tensor(self.bsize,self.classSize).fill_(0).cuda()
        self.distance = torch.Tensor(self.dataSize, self.inputSize).fill_(0).cuda()
        self.sqrdistance = torch.Tensor(self.dataSize, self.inputSize).fill_(0).cuda()

        self.patternOut = torch.zeros(x.size()[0],self.dataSize).cuda()
        self.uout = torch.zeros(self.classSize)
   
        #--if self.isVectorVar==1:
        #--    self.sqrvar = torch.mul(self.vectorvar,self.vectorvar)
        #--    self.twosqrvar = self.sqrvar*2
        #--else: 
        #--   self.sqrvar = self.variance*self.variance
        #--    self.twosqrvar = 2* self.sqrvar

        #print(x.size())
        self.distance = torch.bmm(torch.Tensor(self.bsize,self.dataSize,1).fill_(1).cuda(),x)
        #print(self.distance)
        #print(self.distance.size())
        #input()
        self.distance = torch.sub(self.distance,self.trainingData) 
        #print(self.distance)
        #print(self.distance.size())
        #input()
        
        self.sqrdistance = torch.mul(self.distance,self.distance)
        #print(self.sqrdistance)
        #print(self.sqrdistance.size())
        #input()
        

        self.sqrdistance = torch.div(self.sqrdistance,self.inputSize)
        
        self.sqrdistance = torch.sum(self.sqrdistance,2)
        
        #self.debug = self.debug + 1
        #if self.debug % 500 == 0:
        #    print(self.sqrdistance)
        #    print(self.sqrdistance.size())
        #    input()
        
        #print(self.inputSize*self.twosqrvar)
        #print((self.inputSize*self.twosqrvar).size())

        
        #--self.patternOut = torch.div(self.sqrdistance,self.twosqrvar)
        
        #print(self.patternOut)
        #print(self.patternOut.size())
        #input()
        
        self.patternOut = torch.exp(-self.sqrdistance)
        #print(self.patternOut)
        #print(self.patternOut.size())
        #input()
        
        #self.denominator = torch.sum(self.patternOut,1)
        #print(self.denominator)
        #print(self.denominator.size())
        #input()
        self.output = torch.matmul(self.patternOut.unsqueeze(1),self.divergeVals) 
        #print(self.output)
        #print(self.output.size())
        #input()
        
        #self.normdiv = torch.sum(self.divergeVals,1)
        #print(self.normdiv)
        #print(self.normdiv.size())
        #input()
        self.output = self.output.squeeze(1)
        #print(self.output)
        #print(self.output.size())
        #input()

        self.output = self.output + 1e-12
        
        self.denominator = torch.sum(self.output,1)
        #print(self.denominator)
        #print(self.denominator.size())
        #input()
        self.output = torch.div(self.output,self.denominator.unsqueeze(1))
        #print(self.output)
        #print(self.output.sum(1))
        #print(self.output.size())
        #input()
        
        self.output = torch.log(self.output)

        #print(self.output)
        #print(self.output.sum(1))
        #print(self.output.size())
        #input()
        return self.output



        
    def calculateClassNorm(self):
        self.classNormVals = torch.zeros(self.classSize).cuda()
        for i in range(0,self.dataSize):
            self.classNormVals[self.classOfData[i]] = self.classNormVals[self.classOfData[i]] + 1
        self.classNormVals = torch.div(torch.ones(self.classSize).cuda(), self.classNormVals)

        self.classNorm = torch.zeros(self.dataSize).cuda()
        for i in range(0,self.dataSize):
            self.classNorm[i] = self.classNormVals[self.classOfData[i]]
        return 

    def calculateDiverge(self):
        #print(self.divergeVals.size())
        for i in range(0,self.dataSize):
            for j in range(0,self.classSize):
                if self.classOfData[i]==j: 
                    #self.divergeVals[:,i,j].fill_(0.9*torch.exp(0.9-self.ymax[0]))
                    self.divergeVals[i,j].fill_(1)
                else:
                    #self.divergeVals[:,i,j].fill_(0.1*torch.exp(0.1-self.ymax[0]))
                    self.divergeVals[i,j].fill_(0)
                    
        #print(self.divergeVals)
        #print(self.divergeVals.size())
        #input()

        return

    def loadVariance(self):
        #self.vectorvar = torch.ones(1,self.dataSize)
        return




