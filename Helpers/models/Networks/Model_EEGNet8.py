"""
Simple EEGNet
based on "~" paper
"""
from pickle import TRUE
import torch
import torch.nn as nn

class EEGNet8(nn.Module):
    def __init__(self, num_classes, input_ch, input_time, track_running=True):
        super(EEGNet8, self).__init__()
        
        self.n_classes = num_classes
        freq = input_time #################### num_seg = frequency*window size

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, freq//3), stride=1, bias=False, padding=(1 , freq//4)),
            nn.BatchNorm2d(8, track_running_stats=track_running),
            nn.Conv2d(8, 16, kernel_size=(input_ch, 1), stride=1, groups=8),
            nn.BatchNorm2d(16, track_running_stats=track_running),
            nn.ELU(),
            # nn.AdaptiveAvgPool2d(output_size = (1,265)),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 16 , kernel_size=(1,freq//6),padding=(0,freq//6), groups=16),
            nn.Conv2d(16, 16, kernel_size=(1,1)),
            nn.BatchNorm2d(16, track_running_stats=track_running),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25),
            )
    
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        x = x[0].unsqueeze(dim=1).permute(0,1,2,3) ############# only use EEG (bs, 1, c, s)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output=self.clf(output) 
        return output

# if __name__ == '__main__':
#     model = EEGNet8(2, 32, 600, 10, True) # n_classes, n_channel, n_timewindow
#     # pred = model(torch.zeros(50, 1, 20, 250))
#     print(model)
#     from pytorch_model_summary import summary

#     print(summary(model, torch.rand((1, 1, 32, 600)), show_input=True))
#     # model input = torch.rand((1,1,32,600))
#     # batch size, channel, eeg electrodes, time window 