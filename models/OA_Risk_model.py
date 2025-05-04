import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class convBlock(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Feedforward(nn.Module):
    def __init__(self, inplace, outplace):
        super().__init__()

        self.conv1 = convBlock(inplace, outplace, kernel_size=3, padding=1)
        self.conv2 = convBlock(outplace, outplace, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class POELatent(nn.Module):
    """
    Adapted from: https://github.com/Li-Wanhua/POEs
    """
    def __init__(self, num_feat=2048):
        super().__init__()

        self.emd = nn.Sequential(
            nn.Linear(num_feat, num_feat),
            # nn.ReLU(True),
        )
        self.var = nn.Sequential(
            nn.Linear(num_feat, num_feat),
            # nn.BatchNorm1d(num_feat, eps=0.001, affine=False),
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x, max_t=50, use_sto=True):
        emb = self.emd(x)
        log_var = self.var(x)
        sqrt_var = torch.exp(log_var * 0.5)
        if use_sto:
            rep_emb = emb[None].expand(max_t, *emb.shape)
            rep_sqrt_var = sqrt_var[None].expand(max_t, *sqrt_var.shape)
            norm_v = torch.randn_like(rep_emb).cuda()
            sto_emb = rep_emb + rep_sqrt_var * norm_v
            drop_emb = self.drop(sto_emb)
            # logit = self.final(drop_emb)
        else:
            drop_emb = self.drop(emb)
            # logit = self.final(drop_emb)
        return drop_emb, emb, log_var


class BaselineModel(nn.Module):
    def __init__(self, arch='resnet18'):
        super(BaselineModel, self).__init__()
        # create model
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
        # print(model)
        if 'densenet' in arch:
            num_feat = model.classifier.in_features
            # model.classifier = nn.Identity()
        elif 'resnet' in arch:
            num_feat = model.fc.in_features
            # model.fc = nn.Identity()
        elif 'vgg' in arch:
            num_feat = model.classifier[-1].in_features
            # model.classifier[-1] = nn.Identity()
        elif 'convnext' in arch:
            num_feat = model.classifier[-1].in_features
            # model.classifier[-1] = nn.Identity()
        elif 'efficientnet' in arch:
            num_feat = model.classifier[-1].in_features

        if 'efficientnet' in arch or 'convnext' in arch:
            self.model = []
            for name, module in model.named_children():
                if name == 'avgpool':
                    continue
                if name == 'classifier':
                    continue
                self.model.append(module)
        else:
            self.model = []
            for name, module in model.named_children():
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    continue
                if isinstance(module, nn.Linear):
                    continue
                self.model.append(module)

        self.model = nn.Sequential(*self.model)
        self.num_feat = num_feat

    def forward(self, x):
        # x = torch.cat([x, x, x], dim=1)
        return self.model(x)

    def get_num_feat(self):
        return self.num_feat


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Simple_AttentionPool(nn.Module):
    """
    Pool to learn an attention over the slices
    Adapted from: https://github.com/reginabarzilaygroup/Sybil
    """
    def __init__(self, **kwargs):
        super(Simple_AttentionPool, self).__init__()
        self.attention_fc = nn.Linear(kwargs['num_chan'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.norm = nn.LayerNorm(kwargs['num_dim'])

    def forward(self, x):
        '''
        args:
            - x: tensor of shape (B, C, N)
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, C)
                + output['hidden']: tensor (B, C)
        '''
        output = {}
        B, C, W, H = x.shape
        # spatially_flat_size = (*x.size()[:2], -1)  # B, C, N

        spatially_flat_size = (B, C, -1)
        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1, 2))  # B, N, 1

        attention_map = self.norm(self.logsoftmax(attention_scores.transpose(1, 2)).view(B, -1)).view(B, 1, W, H)
        output['attention_map'] = attention_map
        attention_scores = self.softmax(attention_scores.transpose(1, 2))  # B, 1, N

        x = x * attention_scores  # B, C, N
        output['hidden'] = torch.sum(x, dim=-1)
        return output


class ContinuousPosEncoding(nn.Module):
    """
    Adapted from: https://github.com/reginabarzilaygroup/Sybil
    """
    def __init__(self, dim=256, drop=0.1, maxtime=240):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        position = torch.arange(0, maxtime, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(maxtime, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, xs, times):
        ys = xs
        times = times.long()
        for b in range(xs.shape[1]):
            # xxxxx = self.pe[times[b]]
            ys[:, b] += self.pe[times[b]]
        return self.dropout(ys)


class OA_BreaCR(nn.Module):
    def __init__(self, args, mtp=False):
        super(OA_BreaCR, self).__init__()
        # create model
        model = BaselineModel(arch=args.arch)
        num_feat = model.get_num_feat()
        self.num_feat = num_feat
        self.model = model
        # num_feat = num_feat * 2 if mtp else num_feat
        self.final = nn.Sequential(
            nn.Linear(num_feat, args.num_output_neurons),
            # nn.ReLU(),
        )  # output layer
        # self.pooling = nn.AdaptiveMaxPool2d((1,1))
        # self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.pooling = Simple_AttentionPool(
            num_chan=num_feat,
            conv_pool_kernel_size=7,
            stride=1,
            num_dim=int(args.img_size[0]/32) * int(args.img_size[1]/32),
        )
        if 'POE' in args.model_method:
            self.POE = True
            self.POELatent = POELatent(num_feat=num_feat)
        else:
            self.POE = False

        shape = [int(args.img_size[0]/32), int(args.img_size[1]/32)]
        self.reg_transformer = SpatialTransformer(shape)
        self.flew = Feedforward(inplace=2, outplace=2)
        # self.inter = Feedforward(inplace=num_feat*4, outplace=num_feat)
        self.pos_encoding = ContinuousPosEncoding(dim=num_feat, drop=0.2)
        # self.mlp = FeedForward_linear(dim=num_feat*3, hidden_dim=num_feat, dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(num_feat*3, num_feat),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(num_feat, num_feat),)

        self.final_single = nn.Sequential(
            nn.Linear(num_feat, args.num_output_neurons),
            # nn.ReLU(),
        )  # output layer

        self.difference_single = nn.Sequential(
            nn.Linear(num_feat, args.num_output_neurons),
            # nn.ReLU(),
        )  # output layer

    def forward(self, target_x, prior_x=None, time=None, **kwargs):

        mask = kwargs['mask'] if 'mask' in kwargs else None
        prior_mask = kwargs['prior_mask'] if 'prior_mask' in kwargs else None

        if mask is None:
            x = torch.cat([target_x, target_x, target_x], dim=1)
        else:
            x = torch.cat([(mask-0.5)*2, target_x*mask, target_x], dim=1)

        if prior_mask is None:
            prior_x = torch.cat([prior_x, prior_x, prior_x], dim=1)
        else:
            prior_x = torch.cat([(prior_mask - 0.5) * 2, prior_x, prior_x * prior_mask], dim=1)

        x = self.model(x)
        prior_x = self.model(prior_x)

        hidden_x = self.pooling(x)
        hidden_prior_x = self.pooling(prior_x)

        attention_map_x = hidden_x['attention_map']
        attention_map_prior_x = hidden_prior_x['attention_map']

        b, c, w, h = attention_map_x.shape
        # x_ = x.view(b * c, 1, w, h)
        # prior_x_ = prior_x.view(b * c, 1, w, h)
        # transform into flow field
        x_prior_x_ = torch.cat([attention_map_x, attention_map_prior_x], dim=1)
        flow_field = self.flew(x_prior_x_)
        target_x_source_ = self.reg_transformer(attention_map_prior_x, flow_field)
        # prior_x_source_ = self.reg_transformer(x_, -flow_field)
        loss = self.compute_reg_loss(attention_map_x, target_x_source_)

        moved_prior_x = self.reg_transformer(prior_x, flow_field)
        difference = torch.abs(x - moved_prior_x)
        hidden_difference = self.pooling(difference)

        x_hidden_feat = hidden_x['hidden']
        logit_current = self.final_single(x_hidden_feat)

        prior_x_hidden_feat = hidden_prior_x['hidden']
        logit_prior = self.final_single(prior_x_hidden_feat)

        differencehidden_feat = hidden_difference['hidden'].view(1, b, -1)
        differencehidden_feat = self.pos_encoding(differencehidden_feat, time).view(b, -1)
        logit_difference = self.difference_single(differencehidden_feat)

        x = torch.cat([x_hidden_feat, prior_x_hidden_feat, differencehidden_feat], dim=1)
        # flow_field = flow_field.view(b, c*2, w, h)
        # attention = torch.cat([flow_field, x, prior_x], dim=1)
        # attention = self.pooling(self.inter(attention)).view(1, b, -1)
        # x = self.pooling(x).view(b, -1)
        # prior_x = self.pooling(prior_x).view(b, -1)
        # x = torch.cat([x_hidden_feat, x, prior_x_hidden_feat], dim=1)
        x = self.mlp(x)

        if self.POE:
            max_t = kwargs['max_t'] if 'max_t' in kwargs else 50
            use_sto = kwargs['use_sto'] if 'use_sto' in kwargs else True
            x, emb, log_var = self.POELatent(x, max_t=max_t, use_sto=use_sto)
        else:
            emb, log_var = None, None

        logit = self.final(x)
        # return logit, emb, log_var, loss
        return {
            'final': logit,
            'current': logit_current,
            'prior': logit_prior,
            'difference': logit_difference,
            'emb_final': emb,
            'log_var_final': log_var,
            'loss': loss,
            'attention_map': attention_map_x,
            'attention_map_prior': attention_map_prior_x,
            'attention_map_difference': hidden_difference['attention_map'],
            'flow_field': flow_field,
        }

    def compute_reg_loss(self, x, target_x_source):
        loss_t1 = torch.mean((x - target_x_source) ** 2)
        # loss_t2 = torch.mean((prior_x_ - prior_x_source_) ** 2)
        return loss_t1 * 1e-2
