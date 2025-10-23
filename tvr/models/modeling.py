import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import CrossModel, Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
from .psi_module import Agent
import numpy as np
from torch.distributions import Normal, kl_divergence, Independent


allgather = AllGather.apply
allgather2 = AllGather2.apply


def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class GARE(nn.Module):
    def __init__(self, config):
        super(GARE, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        # -------------------------------------
        # psi module, to predict increments Delta, also can be seen as Dirac delta function
        # implemented as a pair-wise parallelization cross-attention Transformer module
        self.agent = Agent()
        self.psi = self.agent.psi
        # -------------------------------------


        width = int(transformer_width // self.config.center)
        self.weight_fc = nn.Sequential(
            nn.Linear(2 * width, 4 * width), nn.ReLU(inplace=True),
            nn.Linear(4 * width, 1))
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn(config)
        
        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()
                
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.load_state_dict(new_state_dict, strict=False)  # only update new state (seqTransf/seqLSTM/tightTransf)
        ## <=== End of initialization trick

        self.global_step = 0

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0, old_policy=None):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, cls = self.get_text_video_feat(text_ids, text_mask, video, video_mask, global_step=global_step, shaped=True)

        self.global_step = global_step
        if self.training:
            if torch.cuda.is_available():  # batch merge here
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat.contiguous(), self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                cls = allgather(cls, self.config)
                torch.distributed.barrier()  # force sync

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            M_t2v_logits, M_v2t_logits, reg_loss = self.get_similarity_logits(text_feat, cls, video_feat,
                                                                    text_mask, video_mask, global_step=global_step, shaped=True)
            
            M_loss_t2v = self.loss_fct(M_t2v_logits * logit_scale)
            M_loss_v2t = self.loss_fct(M_v2t_logits * logit_scale)
            M_loss = (M_loss_t2v + M_loss_v2t) / 2
            
            loss = M_loss + reg_loss
            return loss, M_loss, reg_loss
        else:
            return None

    def similarity(self, text_feat, cls, video_feat, text_mask, video_mask, global_step):
        v_feat = video_feat # (b,v,dim)
        v_weight = torch.einsum('ad,bvd->abv', [cls, video_feat])
        v_weight = torch.softmax(v_weight / self.config.temp, dim=-1)
        v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])
        video_feat_pooled = torch.einsum('abv,bvd->abd', [v_weight, video_feat])
        video_feat_mean = v_feat.mean(1).unsqueeze(0) # ->(1,b,dim)

        a, b = cls.size(0), video_feat.size(0)

        _cls = cls.unsqueeze(1).repeat(1,b,1) # (a,b,dim)
        delta = video_feat_mean - _cls
        delta, _ = self.psi(delta.unsqueeze(0), v_feat.unsqueeze(0)) #
        delta = delta.squeeze(0)
        cls = _cls + delta # (a,b,dim)

        cls, video_feat = cls.contiguous(), video_feat.contiguous()
        t_feat = cls
        v_feat = video_feat_pooled

        _t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        _v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.einsum('abd,abd->ab', [_t_feat, _v_feat])

        reg_loss = 0.0
        if self.training:
            # ---------------------------------------------------------------------------
            # Relaxation of Variational Information Bottleneck Compression Term
            # this OP is to make sure the deteriministic posterior q_\psi(delta | t, v)
            # (i.e., Dirac distribution function) to circumvent the sigularity,
            # thus avoid the KL divergence to be infinite
            mu = delta.mean(dim=0)
            sigma = delta.std(dim=0) + 1e-6
            target_dist = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
            estimated_dist = Normal(mu, sigma)
            kl_div = kl_divergence(estimated_dist, target_dist).mean()
            beta = self.config.beta
            relaxed_vib_loss = beta * kl_div
            # ---------------------------------------------------------------------------


            lambda_dir, lambda_epsilon = self.config.lambda_dir, self.config.lambda_epsilon
            dir_loss = self.direction_diversity_loss(delta, alpha=self.config.alpha)
            norm_loss = self.norm_based_epsilon_regularization_loss(delta, lambda_lower=self.config.lambda_lower) # hard margin 49.1

            t_anchor_reg_loss = lambda_dir * dir_loss + lambda_epsilon * norm_loss
            reg_loss = relaxed_vib_loss +t_anchor_reg_loss
            if self.config.local_rank==0 and global_step%50==0:
                print(f't dir loss: {lambda_dir * dir_loss}, t rad loss: {lambda_epsilon * norm_loss}')
                print(f'kl loss: {relaxed_vib_loss}')


        return retrieve_logits, retrieve_logits.T, reg_loss


    # Direction Diversity Regularization for Text Anchors
    def direction_diversity_loss(self, delta, alpha=1):
        B_t, B_v, D = delta.shape
        delta_dir = F.normalize(delta, dim=2)  # [B_t, B_v, D]

        loss = 0.0
        for i in range(B_t):
            z = delta_dir[i]
            sim = torch.matmul(z, z.T)
            dist = 1 - sim
            loss += torch.log(torch.exp(-alpha * dist).mean() + 1e-8)

        return loss / B_t

    # Norm-Based Regularization of Trust-Region Radii for Text Anchors
    def norm_based_epsilon_regularization_loss(self, delta, lambda_lower=1):
        B_t, B_v, D = delta.shape
        dist = delta.norm(p=2, dim=2)
        var = dist.var(dim=1)
        loss = -var.mean()
        return loss  if loss >= -lambda_lower else 0


    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        cls, text_feat = cls.float(), text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        cls = cls.view(bs_pair, -1, cls.size(-1)).squeeze(1)
        return text_feat, cls

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))

        if self.config.datatype == 'msvd':
            return video_feat

        video_feat = self.agg_video_feat(video_feat, video_mask, self.agg_module)
        return video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, global_step=0, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, cls = self.get_text_feat(text_ids, text_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, video_feat, cls

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    # TODO: if you try to finetune the model on MSVD, comment this function in `get_video_feat`
    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        return video_feat


    def get_similarity_logits(self, text_feat, cls, video_feat, text_mask, video_mask, global_step=0, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        M_t2v_logits, M_v2t_logits, reg_loss= self.similarity(text_feat, cls, video_feat, text_mask, video_mask, global_step)
        

        return M_t2v_logits, M_v2t_logits, reg_loss

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
