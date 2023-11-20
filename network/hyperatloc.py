import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from network.att import AttentionBlock


class PoseRegressorHyper(nn.Module):
    """
    The hyper-networks-based regression head
    This module receives both the input vector to process and the weights for its regression layers
    """

    def __init__(self, decoder_dim, hidden_dim, output_dim, hidden_scale=1.0):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the output dimension
        hidden_scale: (float) Determines the ratio between the input and the hidden layers' dimensions
        """
        super().__init__()
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_scale = hidden_scale

    @staticmethod
    def batched_linear_layer(x, wb):
        """
        Explicit implementation of a batched linear regression
        x: (Tensor) the input tensor to process
        wb: (Tensor) The weights and bias of the regression layer
        """
        # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
        one = torch.ones(*x.shape[:-1], 1, device=x.device)
        linear_res = torch.matmul(torch.cat([x, one], dim=-1).unsqueeze(1), wb)
        return linear_res.squeeze(1)

    @staticmethod
    def _swish(x):
        """
        Implementation of the Swish activation layer
        Reference: "Searching for Activation Functions" [https://arxiv.org/abs/1710.05941v2]
        """
        return x * F.sigmoid(x)

    def forward(self, x, weights):
        """
        Forward pass
        x: (Tensor) the input tensor to process
        weights: (Dict) A dictionary holding the weights and biases of the input, hidden and output regression layers
        """
        # Regressing over the input layer
        if 'w_h1' in weights:
            x = self._swish(self.batched_linear_layer(x, weights.get('w_h1').view(weights.get('w_h1').shape[0],
                                                                                  (self.decoder_dim + 1),
                                                                                  self.hidden_dim)))
        # Regressing over all hidden layers
        for index in range(len(weights.keys()) - 2):
            if f'w_h{index + 2}' in weights:
                x = self._swish(self.batched_linear_layer(x,
                                                          weights.get(f'w_h{index + 2}').view(
                                                              weights.get(f'w_h{index + 2}').shape[0],
                                                              (self.hidden_dim + 1),
                                                              (int(self.hidden_dim * self.hidden_scale)))))
        # Regressing over the output layer
        if 'w_o' in weights:
            x = self.batched_linear_layer(x, weights.get('w_o').view(weights.get('w_o').shape[0],
                                                                     (int(self.hidden_dim * self.hidden_scale) + 1),
                                                                     self.output_dim))
        return x


class HyperAtLoc(nn.Module):
    def __init__(self, droprate=0.5, pretrained=True, feat_dim=2048):
        super(HyperAtLoc, self).__init__()
        self.droprate = droprate
        self._backbone_dim = feat_dim

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.att = AttentionBlock(feat_dim)
        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        # =========================================
        # Hyper networks
        # =========================================
        self.hyper_dim_t = 256 # config.get('hyper_dim_t')
        self.hyper_in_t_proj = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_t)
        self.hyper_in_t = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_t)
        self.hyper_in_t_fc_2 = nn.Linear(in_features=self.hyper_dim_t, out_features=self.hyper_dim_t)
        self.hypernet_t_fc_h2 = nn.Linear(self.hyper_dim_t, 3 * (self.hyper_dim_t + 1))

        self.hyper_dim_rot = 512 # config.get('hyper_dim_rot')
        self.hyper_in_rot_proj = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_rot)
        self.hyper_in_rot = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_rot)
        self.hyper_in_rot_fc_2 = nn.Linear(in_features=self.hyper_dim_rot, out_features=self.hyper_dim_rot)
        self.hypernet_rot_fc_h2 = nn.Linear(self.hyper_dim_rot, 3 * (self.hyper_dim_rot + 1))

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

        # =========================================
        # Regressor Heads
        # =========================================
        # (1) Hyper-networks' regressors for position (t) and orientation (rot)
        self.regressor_hyper_t = PoseRegressorHyper(self.hyper_dim_t, self.hyper_dim_t, 3, hidden_scale=1.0)
        self.regressor_hyper_rot = PoseRegressorHyper(self.hyper_dim_rot, self.hyper_dim_rot, 3, hidden_scale=1.0)

    @staticmethod
    def _swish(x):
        return x * F.sigmoid(x)

    def forward(self, x):
        ##################################################
        # Backbone Forward Pass
        ##################################################
        x = self.feature_extractor(x)
        x = F.relu(x)

        x = self.att(x.view(x.size(0), -1))


        ##################################################
        # Hyper-networks Forward Pass
        ##################################################
        t_input = self.hyper_in_t_proj(x)
        hyper_in_h2 = self._swish(self.hyper_in_t_fc_2(t_input))
        hyper_w_t_fc_h2 = self.hypernet_t_fc_h2(hyper_in_h2)

        rot_input = self.hyper_in_rot_proj(x)
        hyper_in_h2 = self._swish(self.hyper_in_rot_fc_2(rot_input))
        hyper_w_rot_fc_h2 = self.hypernet_rot_fc_h2(hyper_in_h2)

        self.w_t = {'w_o': hyper_w_t_fc_h2}
        self.w_rot = {'w_o': hyper_w_rot_fc_h2}

        ##################################################
        # Regression Forward Pass
        ##################################################
        # (1) Hyper-network's regressors
        p_x_hyper = self.regressor_hyper_t(self.hyper_in_t(x), self.w_t)
        p_q_hyper = self.regressor_hyper_rot(self.hyper_in_rot(x), self.w_rot)

        # (2) Trained regressors
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        ##################################################
        # Output
        ##################################################
        xyz = torch.add(xyz, p_x_hyper)
        wpqr = torch.add(wpqr, p_q_hyper)

        est_pose = torch.cat((xyz, wpqr), dim=1)
        return est_pose