import torch
from torch.nn import Module, Conv2d, Parameter, Softmax

class PAM_Module(Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            explanation: do this method in each channel
            inputs:
                x: input feature maps(B x C x H x W)
            return:
                out: attention value + input feature
                attention: B x (H x W) x (H x W)
        """

        m_batchsize,C , height, width = x.size()

        # (premute)Rearrange the order (0,1,2) --> (0,2,1)
        # for example:
        # before proj_query: 32 x 32 x 196
        # after  proj_query: 32 x 196 x 32

        proj_query = self.query_conv(x)
        proj_query = proj_query.view(m_batchsize, -1, width*height).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key = proj_key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        proj_value = proj_value.view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # use mask
        # [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        # print self.gamma

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    ''''Channel attention module'''
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs:
                 x: input feature maps( B x C x H x W)
            return:
                out: attention value + input feature
                attention: B x C x C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        # print self.gamma

        return out

if __name__=="__main__":
    inputs = torch.Tensor(32, 256, 14, 14)
    print (inputs.size())
    model = PAM_Module(256)
    out = model(inputs)
    print (out.size())