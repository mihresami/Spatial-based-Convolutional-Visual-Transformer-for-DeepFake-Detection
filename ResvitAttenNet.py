from torchvision.models import resnet18,resnet50
# from torchvision.models.vision_transformer import vit_b_16
from VisionTransformers import vit_b_16_defined,vit_l_16_defined
# from VisionTransformers import vit_l_16
import torch.nn as nn
from transformers import ViTForImageClassification
import torch
from vit_pytorch import ViT
from typing import OrderedDict
from attention import SelfAttention
from pytorch_model_summary import summary
from torchvision import models
class RvitattenNet(nn.Module):
    def __init__(self,seq_length,hidden_dim,layers_dim:dict) -> None:
        super(RvitattenNet,self).__init__()
        self.model = resnet18(weights='ResNet18_Weights.DEFAULT') 
        # self.res = resnet18(pretrained=True) 
        # self.model = nn.Sequential(*list(self.res.children())[:-2])
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.vitmodel = vit_b_16_defined(pretrained=True)
        # self.vitmodel = vit_l_16_defined(pretrained=False)
        self.layers_dim = layers_dim
        # self.vitmodel.pos_embedding = nn.Parameter(torch.empty(1, seq_length+1+196, hidden_dim).normal_(std=0.02)) 
        for param in self.vitmodel.parameters():
            param.requires_grad = False

        self.vitmodel.encoder.pos_embedding = nn.Parameter(torch.empty(1, seq_length+49+1, hidden_dim).normal_(std=0.02)) # 233 197(conv1)
        self.vitmodel.heads = nn.Sequential(nn.Linear(hidden_dim,2))
        self.attention = nn.ModuleList([SelfAttention(input_dim) for _,input_dim in enumerate(self.layers_dim)])
        # self.vitmodel.class_token = nn.Parameter(torch.zeros(1,1,hidden_dim))
        input_feat = self.model.fc.in_features
        self.model.fc = nn.Linear(input_feat,2)
        self.linear1 = nn.Linear(512,hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.feature3 = None
        self.gradients = None


    def forward(self,x):
        input_img = x
        B,C,H,W = x.shape
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        attention_input = x.reshape(x.size(0),x.size(1),x.size(2)*x.size(3))
        attention1 = self.attention[0]
        out = attention1(attention_input).view(*x.shape)
        x = out + x
        x = self.model.layer3(x)
        x  = self.model.layer4(x)
        # attention_input = x.reshape(x.size(0),x.size(1),x.size(2)*x.size(3))
        # attention2 = self.attention[1]
        # out = attention2(attention_input).view(*x.shape)
        # x = x + out
        x = x.view(x.size(0),x.size(1),-1)

        out = x.permute(0,2,1)
        out = self.linear1(out)
        x = self.vitmodel._process_input(input_img)
        x = torch.cat((out,x),dim=1)
        n = x.shape[0]
        batch_class_token = self.vitmodel.class_token.expand(n,-1,-1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vitmodel.encoder(x)
        x = x[:,0]
        # feature  = x
        x = self.vitmodel.heads(x)
        x = self.softmax(x)
        return x



# model = RvitattenNet(seq_length=196,hidden_dim=768,layers_dim=[784,49])
# # model.load_state_dict(torch.load('weight-FF/RvitFFnewattentTF-19-acc0.9963047619047619.pt'), strict=False)
# x = torch.randn(2,3,224,224)
# # print(summary(model,x,show_input= True))
# # print(model)
# out = model(x)
# print(out)

# x = torch.randn(2,2)
# y = torch.randn(2,2)
# print(x,"\n", y)
# print((x*y),"\n",x+y)
