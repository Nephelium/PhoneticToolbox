"""Frozen inference architecture adapted from the project research models; no training imports."""

import torch

from torch import nn

from torch.nn import functional as F

class ParameterRegressor(nn.Module):
    def __init__(self):
        super().__init__();layers=[];inc=1
        for width in [16,24,32,48]:
            layers.extend([nn.Conv2d(inc,width,3,2,1),nn.GroupNorm(4,width),nn.SiLU()]);inc=width
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((3,4)))
        self.head=nn.Sequential(nn.Linear(580,96),nn.SiLU(),nn.Dropout(.1),nn.Linear(96,2))

    def forward(self,x,m):
        return self.head(torch.cat([self.encoder(x).flatten(1),m],1))

def physical(z):
    return torch.pow(2.,z)*z.new_tensor([10.,60.])

class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(c), ChannelNorm(c)
        self.expand = nn.Conv2d(c,c*2,1)
        self.depthwise = nn.Conv2d(c*2,c*2,3,padding=1,groups=c*2)
        self.attention, self.project = nn.Conv2d(c,c,1), nn.Conv2d(c,c,1)
        self.ffn1, self.ffn2 = nn.Conv2d(c,c*2,1), nn.Conv2d(c,c,1)
        self.beta = nn.Parameter(torch.zeros(1,c,1,1))
        self.gamma = nn.Parameter(torch.zeros(1,c,1,1))

    @staticmethod
    def gate(x):
        a,b = x.chunk(2,1)
        return a*b

    def forward(self, x, mask):
        z = self.gate(self.depthwise(self.expand(self.norm1(x))))
        mean = (z*mask).sum((2,3),keepdim=True)/mask.sum((2,3),keepdim=True).clamp_min(1)
        x = x + self.beta*self.project(z*self.attention(mean))
        return x + self.gamma*self.ffn2(self.gate(self.ffn1(self.norm2(x))))

class ConditionalNAFNet(nn.Module):
    """Initial log-Mel + support + frequency + 3 values + 3 known flags.

    Masked channel attention excludes padding. Global attention still requires
    measuring full versus tiled inference differences; tiling is approximate.
    """
    def __init__(self, num_mels=100, width=24, depths=(2,2,4), middle=4):
        super().__init__()
        self.num_mels = num_mels
        self.factor = 2**len(depths)
        self.intro = nn.Conv2d(9,width,3,padding=1)
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        c = width
        for count in depths:
            self.enc.append(nn.ModuleList([NAFBlock(c) for _ in range(count)]))
            self.down.append(nn.Conv2d(c,c*2,2,stride=2))
            c *= 2
        self.middle = nn.ModuleList([NAFBlock(c) for _ in range(middle)])
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for count in reversed(depths):
            self.up.append(nn.Sequential(nn.Conv2d(c,c*2,1),nn.PixelShuffle(2)))
            c //= 2
            self.dec.append(nn.ModuleList([NAFBlock(c) for _ in range(count)]))
        self.out = nn.Conv2d(width,1,3,padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, base, conditions, coverage, valid=None):
        b,f,t = base.shape
        if f != self.num_mels or conditions.shape != (b,6):
            raise ValueError('Frontend or condition shape mismatch')
        freq = torch.linspace(0,1,f,device=base.device,dtype=base.dtype)[None,None,:,None].expand(b,1,f,t)
        cond = conditions[:,:,None,None].expand(b,6,f,t)
        support = coverage[:,None,:,None].expand(b,1,f,t)
        x = torch.cat([base[:,None]/5,support,freq,cond],1)
        mask = torch.ones_like(base[:,None]) if valid is None else valid[:,None].expand(b,f,t)[:,None].to(base.dtype)
        pad = (0,(-t)%self.factor,0,(-f)%self.factor)
        x = F.pad(x,pad,mode='replicate')
        mask = F.pad(mask,pad)
        x = self.intro(x)
        skips, masks = [], []
        for blocks, down in zip(self.enc,self.down):
            for block in blocks:
                x = block(x,mask)
            skips.append(x)
            masks.append(mask)
            x = down(x)
            mask = F.avg_pool2d(mask,2)
        for block in self.middle:
            x = block(x,mask)
        for up, blocks, skip, mask in zip(self.up,self.dec,reversed(skips),reversed(masks)):
            x = up(x)+skip
            for block in blocks:
                x = block(x,mask)
        return base + self.out(x)[:,0,:f,:t]

REPRESENTATION='initial-mel-restoration-visible-v1'

class VisibleMelNAFNet(ConditionalNAFNet):
    representation=REPRESENTATION
    def forward(self,base,conditions,coverage,valid=None):
        prediction=super().forward(base,conditions,coverage,valid)
        # A boundary filter has no fully observed target. Preserve its analytic
        # in-band contribution rather than allowing an unsupervised residual.
        prediction=torch.where(coverage[:,:,None]>=.9999,prediction,base)
        return torch.where(coverage[:,:,None]>0,prediction,prediction.new_tensor(1e-5).log())
