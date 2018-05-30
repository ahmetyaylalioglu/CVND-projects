import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first = True)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        outofLSTM, _ = self.lstm(embeddings)
        #outofLSTM2, _ = self.lstm(embeddings)
        outputs = self.linear(outofLSTM)
        return outputs
        

    def sample(self, inputs):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #This parameters not acceptable as self input,if use as function parameter error occured in notebook files
        result_ids = []
        prediction = None
        max_len = 20
        states = None 
        features = inputs

        for i in range(max_len):
            if(prediction != 1): #If I doesn't use this line; captions have more than 1 <end> token
                
                outofLSTM, states = self.lstm(features, states)
                output = self.linear(outofLSTM)
                _, predicted = output.max(2)
                prediction = predicted.item()
                result_ids.append(prediction)
                features = self.embed(predicted)

        return result_ids