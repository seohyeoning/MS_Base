import torch
import torch.nn as nn
import random

class DeepConvNet(nn.Module):
    def __init__(self, n_classes,input_ch,input_time,
                 batch_norm=True):
        super(DeepConvNet, self).__init__()

        self.batch_norm = batch_norm
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv1d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv1d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                
                nn.Conv1d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv1d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv1d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv1d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha,
                               affine=True, eps=1e-5, track_running_stats=True),
                nn.Conv1d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),

                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv1d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),

                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv1d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
  
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv1d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
        
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        x = x[0].unsqueezE(dim=1).permute(0,1,2,3) ################ only use EEG (bs, 1, c, s)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output=self.clf(output) 
        return output


# if __name__ == '__main__':
#     # from ptflops import get_model_complexity_info

#     # model = DeepConvNet(2,32,600,10, True)
#     model = DeepConvNet(3,28,600,10, True)
#     print(model)

#     from pytorch_model_summary import summary

#     print(summary(model, torch.zeros((1, 1, 32,600)), show_input=False))

#     # macs, params = get_model_complexity_info(model, dummy_size, as_strings=True, print_per_layer_stat=True, verbose=True)  
#     # print('computational complexity: ', macs)
#     # print('number of parameters: ', params)