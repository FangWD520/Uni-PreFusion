import copy


import torch
import numpy as np
from projects.mmdet3d_plugin.models.utils.mamba_compat import MambaBlock
from mmdet.core import multi_apply
from mmdet3d.models.builder import HEADS, build_loss
import torch.nn as nn
from einops import repeat, rearrange

from projects.mmdet3d_plugin.models.utils import TransformerDecoderLayer, gen_sineembed_for_position
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class ConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ConvFuser, self).__init__()
        self.fuser = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x1, x2):
        assert x2.size() == x2.size()
        # print('###x1:', x1.shape)
        # print('###x2:', x2.shape)
        x = torch.cat([x1, x2], dim=1)
        y = self.fuser(x)
        return y


@HEADS.register_module()
class AnchorPlannerHead(nn.Module):
    def __init__(
            self,
            planning_anchor=None,
            in_channels=128 * 3,
            hidden_channel=128,
            # config for Transformer
            num_decoder_layers=3,
            decoder_layer=dict(),
            num_heads=8,
            bn_momentum=0.1,
            bias='auto',
            # loss
            loss_plan_cls=dict(),
            loss_plan_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
            loss_ego_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
            use_rank_loss=False,
            loss_plan_rank=dict(type='CrossEntropyLoss', loss_weight=1.0),
            train_reranker=False,
            loss_plan_rerank=dict(type='CrossEntropyLoss', loss_weight=1.0),
            # independent planning reranker
            use_reranker=False,
            reranker_hidden=None,
            reranker_layers=2,
            reranker_heads=4,
            reranker_use_risk=True,
            # ego status conditioning mode
            use_input_ego_status=False,
            # inference-time trajectory selection
            use_rescore=False,
            rescore_cfg=None,
            use_safety_rerank=False,
            use_oracle_selection=False,
            oracle_metric='L2',
            # others
            train_cfg=None,
            test_cfg=None,
    ):
        super(AnchorPlannerHead, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.hidden_channel = hidden_channel
        self.bn_momentum = bn_momentum
        self.fut_steps = 6
        self.ego_fut_mode = 3
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_rescore = use_rescore
        self.use_safety_rerank = use_safety_rerank
        default_rescore_cfg = dict(
            det_weight=2.0,
            occ_weight=1.0,
            map_weight=1.0,
            cls_weight=1.0,
            topk=6,
            safe_det_thresh=0.05,
            safe_occ_thresh=0.2,
            safe_map_thresh=0.5,
            keep_top1_if_safe=True,
            max_cls_drop=0.2,
            allow_low_score_safe=False,
            fallback='top1',
            det_score_thr=0.2,
            safety_margin=0.3,
            clearance_margin=1.0,
            ego_size=[4.08, 1.73],
            occ_threshold=0.5,
            map_drivable_label=0,
            footprint_resolution=0.5,
            pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        )
        if rescore_cfg is not None:
            default_rescore_cfg.update(rescore_cfg)
        self.use_oracle_selection = use_oracle_selection
        self.oracle_metric = oracle_metric
        self.use_input_ego_status = use_input_ego_status


        self.rescore_cfg = default_rescore_cfg

        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_cls = build_loss(loss_plan_cls)
        self.loss_ego_reg = build_loss(loss_ego_reg)
        self.use_rank_loss = use_rank_loss
        self.loss_plan_rank = build_loss(loss_plan_rank)

        self.train_reranker = train_reranker
        self.use_reranker = use_reranker
        self.reranker_use_risk = reranker_use_risk
        self.loss_plan_rerank = build_loss(loss_plan_rerank)


        self.fuser = ConvFuser(self.in_channels, self.hidden_channel)

        decoder = list()
        for i in range(self.num_decoder_layers):
            decoder.append(TransformerDecoderLayer(
                                hidden_channel,
                                num_heads,
                                1024)
            )
        self.decoder = nn.ModuleList(decoder)

        planning_anchor = np.load(planning_anchor)
        self.planning_anchor = nn.Parameter(
            torch.tensor(planning_anchor, dtype=torch.float32),
            requires_grad=False,
        )

        self.planning_anchor_encoder = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.ego_encoder = nn.Sequential(
            nn.Linear(10, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        # Prediction Head
        ego_fut_decoder = list()
        for i in range(2):
            ego_fut_decoder.append(nn.Linear(self.hidden_channel, self.hidden_channel))
            ego_fut_decoder.append(nn.ReLU())

        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.reg = nn.Linear(self.hidden_channel, self.fut_steps * 2)
        self.cls = nn.Linear(self.hidden_channel, 1)

        ego = list()
        for i in range(2):
            ego.append(nn.Linear(self.hidden_channel, self.hidden_channel))
            ego.append(nn.ReLU())
        ego.append(nn.Linear(self.hidden_channel, 10))
        self.ego = nn.Sequential(*ego)

        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.ego_info = MLN(3, f_dim=self.hidden_channel)

        kernel_size = tuple([int(x / 2) for x in [180, 180]])
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channel),
            nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size),
        )

        self.ego_anchor = nn.Parameter(
            torch.tensor([[0, 0.5, -1.84 + 1.56 / 2, np.log(4.08), np.log(1.73), np.log(1.56), 1, 0, 0, 0, 0],], dtype=torch.float32),
            requires_grad=False,
        )

        self.ego_anchor_encoder = nn.Sequential(
            nn.Linear(11, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel, bias=False),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.reranker_enabled = self.train_reranker or self.use_reranker
        if self.reranker_enabled:
            reranker_hidden = reranker_hidden or hidden_channel
            self.rerank_feat_proj = nn.Sequential(
                nn.Linear(hidden_channel + 10, reranker_hidden),
                nn.ReLU(),
                nn.LayerNorm(reranker_hidden),
            )
            rerank_layer = nn.TransformerEncoderLayer(
                d_model=reranker_hidden,
                nhead=reranker_heads,
                dim_feedforward=reranker_hidden * 4,
                dropout=0.0,
                batch_first=True,
                activation='gelu')
            self.rerank_encoder = nn.TransformerEncoder(
                rerank_layer, num_layers=reranker_layers)
            self.rerank_traj_encoder = nn.Sequential(
                nn.Linear(self.fut_steps * 2, reranker_hidden),
                nn.ReLU(),
                nn.LayerNorm(reranker_hidden),
                nn.Linear(reranker_hidden, reranker_hidden),
            )
            self.rerank_score_encoder = nn.Sequential(
                nn.Linear(1, reranker_hidden),
                nn.ReLU(),
                nn.LayerNorm(reranker_hidden),
            )
            self.rerank_risk_encoder = nn.Sequential(
                nn.Linear(3, reranker_hidden),
                nn.ReLU(),
                nn.LayerNorm(reranker_hidden),
            )
            self.rerank_head = nn.Sequential(
                nn.Linear(reranker_hidden, reranker_hidden),
                nn.ReLU(),
                nn.Linear(reranker_hidden, 1),
            )
            if not self.reranker_use_risk:
                for param in self.rerank_risk_encoder.parameters():
                    param.requires_grad = False




        self.init_weights()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward_single(self, feats, aux_feats, ego_status, gt_ego_fut_cmd, gt_trajs, img_metas):
        feats = self.fuser(feats, aux_feats)
        bev_pos = self.bev_pos.repeat(feats.shape[0], 1, 1).to(feats.device)

        gt_ego_fut_cmd_feat = gt_ego_fut_cmd.to(feats.dtype).unsqueeze(-1)
        gt_ego_fut_cmd_feat = rearrange(gt_ego_fut_cmd_feat, 'b c l -> b l c')
        ego_anchor = self.ego_anchor.repeat(feats.shape[0], 1)

        ego_query = self.ego_feature_encoder(feats)
        ego_query = ego_query.unsqueeze(1).squeeze(-1).squeeze(-1)
        ego_query = self.query_encoder(ego_query)

        feats = feats.flatten(-2)
        for i in range(self.num_decoder_layers):
            ego_query = self.decoder[i](rearrange(ego_query, 'b l c -> b c l'), feats, None, bev_pos)
            ego_query = rearrange(ego_query, 'b c l -> b l c')

        ego_anchor_embed = self.ego_anchor_encoder(ego_anchor)
        ego_anchor_embed = ego_anchor_embed.unsqueeze(1)
        ego_status_query = ego_query + ego_anchor_embed
        outputs_ego_status = self.ego(ego_status_query)

        planning_anchor = self.planning_anchor.unsqueeze(0).repeat(len(feats), 1, 1, 1, 1)[:, :, :, -1:].flatten(-2)
        planning_anchor = gen_sineembed_for_position(planning_anchor, self.hidden_channel)
        planning_anchor = self.planning_anchor_encoder(planning_anchor)
        planning_anchor = rearrange(planning_anchor, 'b l k c -> b (l k) c')

        if self.use_input_ego_status:
            ego_status_source = ego_status.to(outputs_ego_status.dtype)
        else:
            ego_status_source = outputs_ego_status.squeeze(1)
        ego_status_feat = self.ego_encoder(ego_status_source).unsqueeze(1)
        ego_query = ego_query + planning_anchor + ego_status_feat
        gt_ego_fut_cmd_feat = repeat(gt_ego_fut_cmd_feat, 'b l c -> b (l k) c', k=ego_query.shape[1])
        ego_query = self.ego_info(ego_query, gt_ego_fut_cmd_feat)

        ego_query = self.ego_fut_decoder(ego_query)
        outputs_ego_trajs = self.reg(ego_query)
        outputs_ego_trajs_cls = self.cls(ego_query)

        outputs_ego_trajs = rearrange(outputs_ego_trajs, 'b (m k) (n c) -> b m k n c', k=self.planning_anchor.shape[1], m=self.ego_fut_mode, n=self.fut_steps)
        outputs_ego_trajs_cls = rearrange(outputs_ego_trajs_cls.flatten(-2), 'b (m k) -> b m k', k=self.planning_anchor.shape[1], m=self.ego_fut_mode)
        ego_fut_query = rearrange(ego_query, 'b (m k) c -> b m k c', k=self.planning_anchor.shape[1], m=self.ego_fut_mode)

        res = {
            'ego_fut_preds': outputs_ego_trajs,
            'ego_fut_preds_cls': outputs_ego_trajs_cls,
            'ego_fut_query': ego_fut_query,
            'ego_status_preds': outputs_ego_status,
            'ego_fut_cmd': gt_ego_fut_cmd,
        }
        return [res]

    def forward(self, feats, map_feats, ego_status, gt_ego_fut_cmd, gt_trajs=None, img_metas=None):
        if not isinstance(feats, list):
            feats = [feats]
        if not isinstance(map_feats, list):
            map_feats = [map_feats]
        res = multi_apply(self.forward_single, feats, map_feats, [ego_status], [gt_ego_fut_cmd], [gt_trajs], [img_metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_best_reg(self, pred_trajs, pred_trajs_cls, gt_trajs, ego_fut_cmd, reg_weight):
        batch_size, num_pred, mode, ts, d = pred_trajs.shape
        bs_indices = torch.arange(batch_size, device=pred_trajs.device)
        cmd = ego_fut_cmd.argmax(dim=-1)
        pred_trajs = pred_trajs[bs_indices, cmd].unsqueeze(1)
        best_cls = pred_trajs_cls[bs_indices, cmd].unsqueeze(1)

        reg_preds_cum = pred_trajs.cumsum(dim=-2)
        reg_gt_cum = gt_trajs.cumsum(dim=-2)
        dist = torch.linalg.norm(reg_gt_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
        dist = dist * reg_weight
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        target_cls = mode_idx

        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
        best_reg = torch.gather(pred_trajs, 2, mode_idx).squeeze(1)
        return best_cls, best_reg, target_cls

    def get_oracle_rank_target(self, pred_trajs, gt_trajs, ego_fut_cmd, reg_weight):
        batch_size, num_cmd, num_mode, ts, d = pred_trajs.shape
        bs_indices = torch.arange(batch_size, device=pred_trajs.device)
        cmd = ego_fut_cmd.argmax(dim=-1)
        pred_trajs = pred_trajs[bs_indices, cmd]
        reg_preds_cum = pred_trajs.cumsum(dim=-2)
        reg_gt_cum = gt_trajs.cumsum(dim=-2).squeeze(1)
        weight = reg_weight.squeeze(1).squeeze(1)
        dist = torch.linalg.norm(reg_gt_cum.unsqueeze(1) - reg_preds_cum, dim=-1)
        dist = dist * weight.unsqueeze(1)
        valid_count = weight.sum(dim=-1).clamp(min=1.0)
        dist = dist.sum(dim=-1) / valid_count.unsqueeze(-1)
        target = torch.argmin(dist, dim=-1)
        return target

    def encode_reranker_features(self, traj_candidates, cls_scores, ego_query_feat,
                                 ego_status_pred, results_det=None, results_occ=None,
                                 results_map=None):
        k = traj_candidates.shape[0]
        traj_feat = self.rerank_traj_encoder(traj_candidates.reshape(k, -1))
        cls_feat = self.rerank_score_encoder(cls_scores[:, None])
        ego_context = torch.cat([ego_query_feat.expand(k, -1), ego_status_pred.expand(k, -1)], dim=-1)
        ego_feat = self.rerank_feat_proj(ego_context)
        rerank_feat = traj_feat + cls_feat + ego_feat

        if self.reranker_use_risk:
            det_risk = self.det_collision_risk(traj_candidates, results_det) if results_det is not None else cls_scores.new_zeros(k)
            occ_risk = self.occ_risk(traj_candidates, results_occ) if results_occ is not None else cls_scores.new_zeros(k)
            map_risk = self.map_risk(traj_candidates, results_map) if results_map is not None else cls_scores.new_zeros(k)
            risk_feat = self.rerank_risk_encoder(torch.stack([det_risk, occ_risk, map_risk], dim=-1))
            rerank_feat = rerank_feat + risk_feat
        return rerank_feat

    def reranker_forward(self, traj_candidates, cls_scores, ego_query_feat,
                         ego_status_pred, results_det=None, results_occ=None,
                         results_map=None):
        rerank_feat = self.encode_reranker_features(
            traj_candidates, cls_scores, ego_query_feat, ego_status_pred,
            results_det, results_occ, results_map)
        rerank_feat = self.rerank_encoder(rerank_feat.unsqueeze(0)).squeeze(0)
        rerank_logits = self.rerank_head(rerank_feat).squeeze(-1)
        return rerank_logits

    def loss(self, preds_dicts, metas, gt_ego_fut_trajs, gt_ego_fut_masks, gt_ego_fut_cmd, gt_ego_status):
        ego_fut_preds = preds_dicts[0][0]["ego_fut_preds"]
        ego_fut_cmd = preds_dicts[0][0]["ego_fut_cmd"]
        ego_fut_query = preds_dicts[0][0].get('ego_fut_query', None)
        ego_fut_preds_cls = preds_dicts[0][0]["ego_fut_preds_cls"]
        ego_status_preds = preds_dicts[0][0]['ego_status_preds']

        gt_ego_fut_trajs = gt_ego_fut_trajs.unsqueeze(1)

        loss_plan_l1_weight = gt_ego_fut_masks[:, None, None, :]

        cls_pred, reg_pred, cls_target = self.get_best_reg(ego_fut_preds, ego_fut_preds_cls, gt_ego_fut_trajs, ego_fut_cmd, loss_plan_l1_weight)
        loss_plan_reg = self.loss_plan_reg(
            reg_pred.reshape(-1, self.fut_steps * 2),
            gt_ego_fut_trajs.reshape(-1, self.fut_steps * 2),
            loss_plan_l1_weight.repeat(1, 1, 1, reg_pred.shape[-1]).reshape(-1, self.fut_steps * 2)
        )

        loss_plan_cls_weight = loss_plan_l1_weight.squeeze(-1).any(dim=-1)
        loss_plan_cls = self.loss_plan_cls(cls_pred.flatten(end_dim=1), cls_target.flatten(end_dim=1), loss_plan_cls_weight.flatten(end_dim=1))

        loss_ego_reg = self.loss_ego_reg(ego_status_preds.reshape(-1, 10), gt_ego_status.reshape(-1, 10))
        if self.use_rank_loss:
            bs_indices = torch.arange(ego_fut_preds.shape[0], device=ego_fut_preds.device)
            cmd = ego_fut_cmd.argmax(dim=-1)
            rank_logits = ego_fut_preds_cls[bs_indices, cmd]
            rank_target = self.get_oracle_rank_target(
                ego_fut_preds, gt_ego_fut_trajs, ego_fut_cmd, loss_plan_l1_weight)
            rank_weight = loss_plan_cls_weight.squeeze(1).float()
            loss_plan_rank = self.loss_plan_rank(
                rank_logits, rank_target, weight=rank_weight)


        loss = dict()
        if self.train_reranker and ego_fut_query is not None:
            bs_indices = torch.arange(ego_fut_preds.shape[0], device=ego_fut_preds.device)
            cmd = ego_fut_cmd.argmax(dim=-1)
            rank_target = self.get_oracle_rank_target(
                ego_fut_preds, gt_ego_fut_trajs, ego_fut_cmd, loss_plan_l1_weight)
            rerank_logits = []
            for b in range(ego_fut_preds.shape[0]):
                traj_candidates = ego_fut_preds[b, cmd[b]].cumsum(dim=-2)
                cls_scores = ego_fut_preds_cls[b, cmd[b]]
                rerank_logits.append(self.reranker_forward(
                    traj_candidates, cls_scores,
                    ego_fut_query[b, cmd[b]],
                    ego_status_preds[b, 0]))
            rerank_logits = torch.stack(rerank_logits, dim=0)
            rank_weight = loss_plan_cls_weight.squeeze(1).float()
            loss_plan_rerank = self.loss_plan_rerank(
                rerank_logits, rank_target, weight=rank_weight)


        loss['planning_loss_plan_reg'] = loss_plan_reg
        loss['planning_loss_plan_cls'] = loss_plan_cls
        loss['planning_loss_ego_reg'] = loss_ego_reg
        if self.use_rank_loss:
            loss['planning_loss_plan_rank'] = loss_plan_rank

        if self.train_reranker and ego_fut_query is not None:
            loss['planning_loss_plan_rerank'] = loss_plan_rerank


        return loss

    def get_bboxes(self, preds_dicts, metas, results_det=None, results_occ=None,
                   results_map=None, **data):
        ego_fut_preds = preds_dicts[0][0]['ego_fut_preds']
        ego_fut_cmd = preds_dicts[0][0]['ego_fut_cmd']
        ego_fut_preds_cls = preds_dicts[0][0]["ego_fut_preds_cls"].sigmoid()
        ego_fut_query = preds_dicts[0][0].get('ego_fut_query', None)
        ego_status_preds = preds_dicts[0][0].get('ego_status_preds', None)
        select = ego_fut_cmd.argmax(dim=-1)

        gt_ego_fut_trajs = data.get('gt_ego_fut_trajs', None)
        gt_ego_fut_masks = data.get('gt_ego_fut_masks', None)

        ret_layer = []
        for i in range(len(ego_fut_preds)):
            plan_select = select[i]
            plan_reg = ego_fut_preds[i].cumsum(dim=-2)
            traj_candidates = plan_reg[plan_select]
            plan_cls = ego_fut_preds_cls[i][plan_select]

            oracle_info = None
            if self.use_oracle_selection:
                if gt_ego_fut_trajs is None:
                    raise ValueError('Oracle planning selection requires gt_ego_fut_trajs in test data.')
                gt_traj_i = gt_ego_fut_trajs[i]
                gt_mask_i = gt_ego_fut_masks[i] if gt_ego_fut_masks is not None else None
                mode_idx, oracle_info = self.oracle_select_candidate(
                    traj_candidates, gt_traj_i, gt_mask_i, plan_cls)
            else:
                det_i = results_det[i] if results_det is not None and len(results_det) > i else None
                occ_i = results_occ[i] if results_occ is not None and len(results_occ) > i else None
                map_i = results_map[i] if results_map is not None and len(results_map) > i else None
                if self.use_reranker and ego_fut_query is not None and ego_status_preds is not None:
                    rerank_logits = self.reranker_forward(
                        traj_candidates, plan_cls,
                        ego_fut_query[i, plan_select],
                        ego_status_preds[i, 0],
                        det_i, occ_i, map_i)
                    mode_idx = rerank_logits.argmax(dim=-1)
                elif self.use_safety_rerank:
                    mode_idx = self.safety_rerank_candidates(
                        traj_candidates, plan_cls, det_i, occ_i, map_i)
                else:
                    if self.use_rescore:
                        plan_cls = self.rescore_candidates(
                            traj_candidates, plan_cls, det_i, occ_i, map_i)
                    mode_idx = plan_cls.argmax(dim=-1)

            selected_traj = traj_candidates[mode_idx]
            if oracle_info is not None:
                selected_traj = selected_traj.clone()
                selected_traj.oracle_info = oracle_info
            ret_layer.append(selected_traj)

        assert len(ret_layer) == 1

        return ret_layer

    def oracle_select_candidate(self, traj_candidates, gt_traj, gt_mask=None, cls_scores=None):
        gt_traj = gt_traj.to(traj_candidates.device).cumsum(dim=-2)
        if gt_mask is None:
            gt_mask = traj_candidates.new_ones(traj_candidates.shape[1])
        else:
            gt_mask = gt_mask.to(traj_candidates.device).float()
        valid = gt_mask > 0
        if valid.sum() == 0:
            valid = torch.ones_like(valid, dtype=torch.bool)
        diff = traj_candidates - gt_traj.unsqueeze(0)
        l2 = torch.linalg.norm(diff, dim=-1)
        masked_l2 = l2[:, valid].mean(dim=-1)
        valid_idx = torch.where(valid)[0]
        final_l2 = torch.linalg.norm(
            traj_candidates[:, valid_idx[-1]] - gt_traj[valid_idx[-1]].unsqueeze(0), dim=-1
        ) if valid_idx.numel() > 0 else masked_l2
        if self.oracle_metric.upper() == 'FDE':
            oracle_cost = final_l2
        else:
            oracle_cost = masked_l2
        mode_idx = oracle_cost.argmin(dim=-1)
        oracle_info = dict(
            oracle_mode_idx=int(mode_idx.item()),
            oracle_minade=float(masked_l2[mode_idx].item()),
            oracle_minfde=float(final_l2[mode_idx].item()) if final_l2.ndim > 0 else float(oracle_cost[mode_idx].item()),
        )
        if cls_scores is not None:
            oracle_info['oracle_cls_score'] = float(cls_scores[mode_idx].item())
        return mode_idx, oracle_info

    def rescore_candidates(self, traj_candidates, cls_scores, results_det=None,
                           results_occ=None, results_map=None):
        cfg = self.rescore_cfg
        scores = cls_scores.float().clone()
        if results_det is not None and cfg.get('det_weight', 0) > 0:
            scores = scores - cfg['det_weight'] * self.det_collision_risk(
                traj_candidates, results_det)
        if results_occ is not None and cfg.get('occ_weight', 0) > 0:
            scores = scores - cfg['occ_weight'] * self.occ_risk(
                traj_candidates, results_occ)
        if results_map is not None and cfg.get('map_weight', 0) > 0:
            scores = scores - cfg['map_weight'] * self.map_risk(
                traj_candidates, results_map)
        return scores

    def safety_rerank_candidates(self, traj_candidates, cls_scores, results_det=None,
                                  results_occ=None, results_map=None):
        cfg = self.rescore_cfg
        num_candidates = traj_candidates.shape[0]
        topk = min(int(cfg.get('topk', num_candidates)), num_candidates)
        topk_idx = cls_scores.topk(topk, largest=True).indices
        topk_trajs = traj_candidates[topk_idx]
        topk_cls = cls_scores[topk_idx]

        det_risk = self.det_collision_risk(topk_trajs, results_det) if results_det is not None else topk_cls.new_zeros(topk)
        occ_risk = self.occ_risk(topk_trajs, results_occ) if results_occ is not None else topk_cls.new_zeros(topk)
        map_risk = self.map_risk(topk_trajs, results_map) if results_map is not None else topk_cls.new_zeros(topk)
        total_risk = (
            cfg.get('det_weight', 0.0) * det_risk +
            cfg.get('occ_weight', 0.0) * occ_risk +
            cfg.get('map_weight', 0.0) * map_risk
        )

        safe_mask = torch.ones_like(topk_cls, dtype=torch.bool)
        safe_mask &= det_risk <= cfg.get('safe_det_thresh', 0.05)
        safe_mask &= occ_risk <= cfg.get('safe_occ_thresh', 0.2)
        safe_mask &= map_risk <= cfg.get('safe_map_thresh', 0.5)

        if cfg.get('keep_top1_if_safe', True) and safe_mask[0]:
            return topk_idx[0]

        if not cfg.get('allow_low_score_safe', False):
            max_drop = cfg.get('max_cls_drop', 0.2)
            safe_mask &= topk_cls >= (topk_cls[0] - max_drop)

        if safe_mask.any():
            candidate_idx = torch.where(safe_mask)[0]
            safe_scores = cfg.get('cls_weight', 1.0) * topk_cls[candidate_idx] - total_risk[candidate_idx]
            best_local = candidate_idx[safe_scores.argmax()]
            return topk_idx[best_local]

        fallback = cfg.get('fallback', 'top1')
        if fallback == 'risk_only':
            best_local = total_risk.argmin()
        elif fallback == 'risk_then_score':
            norm_risk = total_risk / total_risk.clamp(min=1e-6).max()
            combined = cfg.get('cls_weight', 1.0) * topk_cls - norm_risk
            best_local = combined.argmax()
        else:
            best_local = topk_cls.argmax()
        return topk_idx[best_local]

    def det_collision_risk(self, traj_candidates, results_det):
        if not isinstance(results_det, dict):
            return traj_candidates.new_zeros(traj_candidates.shape[0])
        boxes = results_det.get('boxes_3d', None)
        scores = results_det.get('scores_3d', None)
        if boxes is None:
            return traj_candidates.new_zeros(traj_candidates.shape[0])
        box_tensor = boxes.tensor.to(traj_candidates.device) if hasattr(boxes, 'tensor') \
            else torch.as_tensor(boxes, device=traj_candidates.device)
        if box_tensor.numel() == 0:
            return traj_candidates.new_zeros(traj_candidates.shape[0])
        keep = None
        if scores is not None:
            keep = scores.to(traj_candidates.device) > self.rescore_cfg.get('det_score_thr', 0.2)
            box_tensor = box_tensor[keep]
        if box_tensor.numel() == 0:
            return traj_candidates.new_zeros(traj_candidates.shape[0])

        agent_centers, agent_yaws, agent_sizes = self.get_agent_future_boxes(box_tensor, results_det, keep)
        ego_centers = traj_candidates
        ego_yaws = self.get_yaw(ego_centers).squeeze(-1)
        ego_size = traj_candidates.new_tensor(self.rescore_cfg.get('ego_size', [4.08, 1.73]))
        ego_sizes = ego_size.view(1, 1, 2).expand(ego_centers.shape[0], ego_centers.shape[1], 2)

        collision = self.oriented_box_collision(
            ego_centers, ego_yaws, ego_sizes, agent_centers, agent_yaws, agent_sizes)
        hard_risk = collision.float().amax(dim=(1, 2))
        clearance = self.oriented_box_clearance_penalty(
            ego_centers, agent_centers, ego_sizes, agent_sizes)
        return torch.maximum(hard_risk, clearance)

    def get_agent_future_boxes(self, box_tensor, results_det, keep=None):
        device = box_tensor.device
        num_agents = box_tensor.shape[0]
        centers = box_tensor[:, None, :2].expand(num_agents, self.fut_steps, 2).clone()
        yaws = box_tensor[:, 6:7].expand(num_agents, self.fut_steps).clone()
        sizes = box_tensor[:, None, 3:5].expand(num_agents, self.fut_steps, 2).clone()

        trajs = results_det.get('trajs_3d', None)
        trajs_cls = results_det.get('trajs_cls_3d', None)
        if keep is not None and trajs is not None:
            keep_for_traj = keep.to(trajs.device) if trajs.is_cuda else keep.detach().cpu()
            trajs = trajs[keep_for_traj]
            if trajs_cls is not None:
                keep_for_cls = keep.to(trajs_cls.device) if trajs_cls.is_cuda else keep.detach().cpu()
                trajs_cls = trajs_cls[keep_for_cls]

        if trajs is not None:
            trajs = trajs.to(device)
            if trajs.ndim == 4:
                if trajs_cls is not None:
                    mode = trajs_cls.to(device).sigmoid().argmax(dim=-1)
                else:
                    mode = trajs.new_zeros((num_agents,), dtype=torch.long)
                gather_idx = mode[:, None, None, None].expand(-1, 1, trajs.shape[-2], trajs.shape[-1])
                centers = torch.gather(trajs, 1, gather_idx).squeeze(1)[..., :2]
            elif trajs.ndim == 3:
                centers = trajs[..., :2]
            if centers.shape[1] != self.fut_steps:
                centers = centers[:, :self.fut_steps]
                if centers.shape[1] < self.fut_steps:
                    pad = centers[:, -1:].expand(-1, self.fut_steps - centers.shape[1], -1)
                    centers = torch.cat([centers, pad], dim=1)
            yaws = self.get_yaw(centers, start_yaw=box_tensor[:, 6:7]).squeeze(-1)
        return centers, yaws, sizes

    def oriented_box_collision(self, ego_centers, ego_yaws, ego_sizes,
                               agent_centers, agent_yaws, agent_sizes):
        ego_corners = self.box_corners_2d(ego_centers, ego_yaws, ego_sizes)[:, :, None]
        agent_corners = self.box_corners_2d(agent_centers, agent_yaws, agent_sizes).permute(1, 0, 2, 3).unsqueeze(0)

        ego_axes = self.box_axes(ego_yaws).unsqueeze(2)
        rel_agent_to_ego = agent_corners - ego_centers.unsqueeze(2).unsqueeze(3)
        proj_agent_on_ego = torch.einsum('ktan d, ktam d -> ktan m', rel_agent_to_ego, ego_axes).abs()
        ego_half = ego_sizes.unsqueeze(2).unsqueeze(3) / 2 + self.rescore_cfg.get('safety_margin', 0.3)
        inside_ego = (proj_agent_on_ego <= ego_half).all(dim=-1).any(dim=-1)

        agent_axes = self.box_axes(agent_yaws).permute(1, 0, 2, 3).unsqueeze(0)
        rel_ego_to_agent = ego_corners - agent_centers.permute(1, 0, 2).unsqueeze(0).unsqueeze(3)
        proj_ego_on_agent = torch.einsum('ktan d, ktam d -> ktan m', rel_ego_to_agent, agent_axes).abs()
        agent_half = agent_sizes.permute(1, 0, 2).unsqueeze(0).unsqueeze(3) / 2 + self.rescore_cfg.get('safety_margin', 0.3)
        inside_agent = (proj_ego_on_agent <= agent_half).all(dim=-1).any(dim=-1)
        return inside_ego | inside_agent

    def oriented_box_clearance_penalty(self, ego_centers, agent_centers, ego_sizes, agent_sizes):
        center_dist = torch.cdist(ego_centers.reshape(-1, 2), agent_centers.reshape(-1, 2)).view(
            ego_centers.shape[0], ego_centers.shape[1], agent_centers.shape[0], agent_centers.shape[1])
        diag_ego = torch.linalg.norm(ego_sizes, dim=-1)[:, :, None, None] / 2
        diag_agent = torch.linalg.norm(agent_sizes, dim=-1)[None, None] / 2
        clearance = center_dist - diag_ego - diag_agent
        min_clearance = clearance.amin(dim=(1, 2, 3))
        margin = self.rescore_cfg.get('clearance_margin', 1.0)
        return torch.clamp((margin - min_clearance) / max(margin, 1e-6), min=0.0, max=1.0)

    @staticmethod
    def box_axes(yaw):
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        axis_x = torch.stack([cos_yaw, sin_yaw], dim=-1)
        axis_y = torch.stack([-sin_yaw, cos_yaw], dim=-1)
        return torch.stack([axis_x, axis_y], dim=-2)

    @staticmethod
    def box_corners_2d(centers, yaws, sizes):
        half = sizes / 2
        base = torch.stack([
            torch.stack([half[..., 0], half[..., 1]], dim=-1),
            torch.stack([half[..., 0], -half[..., 1]], dim=-1),
            torch.stack([-half[..., 0], -half[..., 1]], dim=-1),
            torch.stack([-half[..., 0], half[..., 1]], dim=-1),
        ], dim=-2)
        axes = AnchorPlannerHead.box_axes(yaws)
        rotated = torch.einsum('...cd,...ed->...ce', base, axes)
        return rotated + centers.unsqueeze(-2)


    def occ_risk(self, traj_candidates, results_occ):
        occ = results_occ.get('occ', None) if isinstance(results_occ, dict) else results_occ
        if occ is None:
            return traj_candidates.new_zeros(traj_candidates.shape[0])
        occ = torch.as_tensor(occ, device=traj_candidates.device)
        if occ.ndim == 3:
            occ_bev = (occ > 0).float().amax(dim=-1)
        elif occ.ndim == 2:
            occ_bev = (occ > 0).float()
        elif occ.ndim == 4:
            occ_bev = occ.float().softmax(dim=-1)[..., 1:].sum(dim=-1).amax(dim=-1)
        else:
            return traj_candidates.new_zeros(traj_candidates.shape[0])
        footprint_risk = self.trajectory_footprint_risk(occ_bev, traj_candidates)
        center_risk = self.sample_bev_map(occ_bev, traj_candidates).max(dim=-1)[0]
        return torch.maximum(footprint_risk, center_risk)

    def map_risk(self, traj_candidates, results_map):
        seg = results_map.get('seg', None) if isinstance(results_map, dict) else results_map
        if seg is None:
            return traj_candidates.new_zeros(traj_candidates.shape[0])
        seg = torch.as_tensor(seg, device=traj_candidates.device).float()
        # SegMapHead returns sigmoid probabilities with shape [C, H, W].
        # Channel 0 is drivable_area in the default UniLION config.
        if seg.ndim == 3:
            drivable_label = self.rescore_cfg.get('map_drivable_label', 0)
            drivable = seg[int(drivable_label)].clamp(0, 1)
        elif seg.ndim == 2:
            drivable = seg.clamp(0, 1)
        else:
            return traj_candidates.new_zeros(traj_candidates.shape[0])
        center_values = self.sample_bev_map(drivable, traj_candidates)
        footprint_values = self.trajectory_footprint_values(drivable, traj_candidates)
        return 1.0 - torch.minimum(center_values.mean(dim=-1), footprint_values.mean(dim=-1))

    def sample_bev_map(self, bev_map, traj_candidates):
        h, w = bev_map.shape[-2:]
        pc_range = self.rescore_cfg.get('pc_range', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
        x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
        ix = ((traj_candidates[..., 0] - x_min) / max(x_max - x_min, 1e-6) * (w - 1)).round().long()
        iy = ((traj_candidates[..., 1] - y_min) / max(y_max - y_min, 1e-6) * (h - 1)).round().long()
        ix = ix.clamp(0, w - 1)
        iy = iy.clamp(0, h - 1)
        return bev_map[iy, ix]

    def trajectory_footprint_values(self, bev_map, traj_candidates):
        offsets = self.ego_footprint_offsets(traj_candidates.device, traj_candidates.dtype)
        yaws = self.get_yaw(traj_candidates).squeeze(-1)
        axes = self.box_axes(yaws)
        rotated_offsets = torch.einsum('od,kted->ktoe', offsets, axes)
        footprint_points = traj_candidates[:, :, None, :] + rotated_offsets
        values = self.sample_bev_map(bev_map, footprint_points.reshape(traj_candidates.shape[0], -1, 2))
        return values.view(traj_candidates.shape[0], traj_candidates.shape[1], -1).mean(dim=-1)

    def trajectory_footprint_risk(self, bev_map, traj_candidates):
        values = self.trajectory_footprint_values(bev_map, traj_candidates)
        return values.max(dim=-1)[0]

    def ego_footprint_offsets(self, device, dtype):
        length, width = self.rescore_cfg.get('ego_size', [4.08, 1.73])
        step = self.rescore_cfg.get('footprint_resolution', 0.5)
        xs = torch.arange(-length / 2, length / 2 + 1e-6, step, device=device, dtype=dtype)
        ys = torch.arange(-width / 2, width / 2 + 1e-6, step, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(xs, ys)
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    def get_yaw(self, traj, start_yaw=0, static_dis_thresh=0.5):
        yaw = traj.new_zeros(traj.shape[:-1])
        yaw[..., 1:-1] = torch.atan2(
            traj[..., 2:, 1] - traj[..., :-2, 1],
            traj[..., 2:, 0] - traj[..., :-2, 0],
        )
        yaw[..., -1] = torch.atan2(
            traj[..., -1, 1] - traj[..., -2, 1],
            traj[..., -1, 0] - traj[..., -2, 0],
        )
        if torch.is_tensor(start_yaw):
            start_yaw_value = start_yaw.to(device=traj.device, dtype=traj.dtype).squeeze(-1)
        else:
            start_yaw_value = traj.new_tensor(start_yaw)
        yaw[..., 0] = start_yaw_value
        # for static object, estimated future yaw would be unstable
        start = traj[..., 0, :]
        end = traj[..., -1, :]
        dist = torch.linalg.norm(end - start, dim=-1)
        mask = dist < static_dis_thresh
        start_yaw = yaw[..., 0].unsqueeze(-1)
        yaw = torch.where(
            mask.unsqueeze(-1),
            start_yaw,
            yaw,
        )
        return yaw.unsqueeze(-1)


class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out
