import torch
import torch.nn as nn
from model import build_model
from data_loader2 import dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Solver(object):
    def __init__(self, args):

        self.args = args

        '''Define models'''
        self.feature_extractor = build_model(model='feature extractor')
        self.rpn = build_model(model='region proposal')
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(6272, 1028), nn.ReLU(),
                                                   nn.Linear(1028, 128), nn.ReLU()])
        # self.roi_head_classifier = nn.Sequential(*[nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
        #                                            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
        #                                            nn.AvgPool2d(7)])
        self.cls_loc = nn.Linear(128, 11*4)
        self.score = nn.Linear(128, 11)

        self.feature_extractor.cuda()
        self.rpn.cuda()
        self.roi_head_classifier.cuda()
        self.cls_loc.cuda()
        self.score.cuda()


        '''
        Define Optimizers
        '''
        '''for image restoration'''

        self.opt_rpn = optim.SGD(list(self.feature_extractor.parameters())+list(self.rpn.parameters()), lr=0.001, momentum=0.9)

        # self.opt_roi = optim.Adam(list(self.feature_extractor.parameters()) + list(self.roi_head_classifier.parameters())
        #                           + list(self.cls_loc.parameters()) + list(self.score.parameters()), lr=0.0001)
        self.opt_roi = optim.SGD(list(self.feature_extractor.parameters()) + list(self.roi_head_classifier.parameters())
                                  + list(self.cls_loc.parameters()) + list(self.score.parameters()), lr=0.001, momentum=0.9)


        '''sub sample rate'''
        self.sub_sample = 2
        '''anchor'''
        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1, 2]
        self.pos_iou_threshold = 0.7
        self.neg_iou_threshold = 0.3
        '''For generating proposals'''
        self.nms_thresh = 0.7
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_post_nms = 300
        self.min_size = 16


        '''
        Load Datasets
        '''
        '''inlier train dataset'''
        self.train_dataset = dataset(dataset='detection mnist', mode='train')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, shuffle=False)

        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((7, 7))

    '''Reset gradients'''
    def reset_grad(self):
        self.opt_rpn.zero_grad()
        self.opt_roi.zero_grad()

    '''Train'''
    def train(self, epoch, mode):
        self.feature_extractor.train()
        self.rpn.train()
        self.roi_head_classifier.train()
        self.cls_loc.train()
        self.score.train()

        device = 'cuda' if self.args.cuda else 'cpu'

        '''
        Define Loss array
        '''
        losses = {}
        losses['rpn'] = []
        losses['roi'] = []
        losses['total'] = []

        ##################################################################################
        for batch_idx, (im_data, gt_boxes, labels) in enumerate(tqdm(self.train_loader, desc='Epoch: '+str(epoch))):
            im_data, gt_boxes, labels = im_data.to(device), np.squeeze(gt_boxes), labels
            gt_boxes = np.squeeze(gt_boxes.numpy())
            labels = np.squeeze(labels.numpy())
            im_h, im_w = im_data.size()[2], im_data.size()[3]

            '''Process anchor'''
            '''Make Anchor boxes and label them based on thresholds'''
            anchors, anchor_labels, anchor_locations = self.process_anchor(im_h=im_h, im_w=im_w, bbox=gt_boxes, sub_sample=self.sub_sample,
                                                                  ratios=self.anchor_ratios, scales=self.anchor_scales,
                                                                  pos_iou_threshold=self.pos_iou_threshold,
                                                                  neg_iou_threshold=self.neg_iou_threshold)


            '''RPN network forward'''
            out_feature = self.feature_extractor(im_data)
            pred_anchor_locs, pred_cls_scores = self.rpn(out_feature)

            '''reformat to align with our anchor targets'''
            pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
            pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
            objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
            pred_cls_scores = pred_cls_scores.view(1, -1, 2)

            '''RPN loss'''
            rpn_loc = pred_anchor_locs[0]
            rpn_score = pred_cls_scores[0]

            gt_rpn_loc = torch.from_numpy(anchor_locations).type(torch.FloatTensor).cuda()
            gt_rpn_score = torch.from_numpy(anchor_labels).cuda()

            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)

            pos = gt_rpn_score > 0
            mask = pos.unsqueeze(1).expand_as(rpn_loc)
            mask_loc_preds = rpn_loc[mask].view(-1, 4)
            mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

            x = torch.abs(mask_loc_targets - mask_loc_preds)
            rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))

            rpn_lambda = 2.
            N_reg = (gt_rpn_score > 0).float().sum()
            rpn_loc_loss = rpn_loc_loss.sum() / N_reg

            rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)


            '''anchor proposal'''
            roi, after_delta = self.anchor_proposal(img_size=(im_h, im_w), anchors=anchors,
                                       pred_anchor_locs=pred_anchor_locs[0].cpu().data.numpy(),
                                       objectness_score=objectness_score[0].cpu().data.numpy(),
                                       pre_nms_thresh=self.n_train_pre_nms,
                                       post_nms_thresh=self.n_train_post_nms)


            '''iou of each ground truth object with the region proposals, 
            We will use the same code we have used in Anchor boxes to calculate the ious'''
            sample_roi, gt_roi_labels, gt_roi_locs = self.random_choice_roi(roi=roi, gt_boxes=gt_boxes, labels=labels,
                                                                            n_sample=32, pos_ratio=0.5)

            ##########################################################
            rois = torch.from_numpy(sample_roi).float()
            roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
            roi_indices = torch.from_numpy(roi_indices).float()

            indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
            xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]

            indices_and_rois = xy_indices_and_rois.contiguous()

            output = []
            rois = indices_and_rois.data.float()
            rois[:, 1:].mul_(1 / self.sub_sample)  # Subsampling ratio
            rois = rois.long()
            num_rois = rois.size(0)
            for i in range(num_rois):
                roi = rois[i]
                im_idx = roi[0]
                im = out_feature.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
                output.append(self.adaptive_max_pool(im))

            output = torch.cat(output, 0)
            k = output.view(output.size(0), -1)

            '''fast rcnn forward'''
            k = self.roi_head_classifier(k)
            roi_cls_loc = self.cls_loc(k)
            roi_cls_score = self.score(k)

            '''Fast RCNN loss'''
            gt_roi_locs = torch.from_numpy(gt_roi_locs).type(torch.FloatTensor).cuda()
            gt_roi_labels = torch.from_numpy(np.float32(gt_roi_labels)).long().cuda()

            roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_labels, ignore_index=-1)
            n_sample = roi_cls_loc.shape[0]
            roi_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_labels]

            ################################################################
            roi_pos = gt_roi_labels > 0
            roi_mask = roi_pos.unsqueeze(1).expand_as(roi_loc)

            roi_mask_loc_preds = roi_loc[roi_mask].view(-1, 4)
            roi_mask_loc_targets = gt_roi_locs[roi_mask].view(-1, 4)

            x = torch.abs(roi_mask_loc_targets - roi_mask_loc_preds)
            roi_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))

            roi_lambda = 10.
            N_reg = (gt_roi_labels > 0).float().sum() + 1
            roi_loc_loss = roi_loc_loss.sum() / N_reg

            #################################################################
            roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)

            losses['rpn'].append(rpn_loss.item())
            losses['roi'].append(roi_loss.item())

            if mode == 'rpn':
                print("training rpn")
                print(rpn_loc_loss)
                print(rpn_cls_loss)
                total_loss = rpn_loss
                total_loss.backward()
                self.opt_rpn.step()

            elif mode == 'frcnn':
                print("training frcnn")
                print(roi_cls_loss)
                print(roi_loc_loss)
                total_loss = roi_loss
                total_loss.backward()
                self.opt_roi.step()

            losses['total'].append(total_loss.item())

            scores = F.softmax(roi_cls_score, 1).cpu().data.numpy()
            box_deltas = roi_loc.cpu().data.numpy()
            im = im_data.cpu().data.numpy()

            self.reset_grad()

            # self.visualize(image=im, scores=scores[:, 1:], boxes=sample_roi, box_deltas=box_deltas)

        return losses



    '''Test'''
    def test(self):
        self.feature_extractor.eval()
        self.rpn.eval()
        self.roi_head_classifier.eval()
        self.cls_loc.eval()
        self.score.eval()

        device = 'cuda' if self.args.cuda else 'cpu'

        '''
        Define Loss array
        '''
        ##################################################################################
        with torch.no_grad():
            for batch_idx, (im_data, gt_boxes, labels) in enumerate(tqdm(self.train_loader, desc='Test')):
                im_data, gt_boxes, labels = im_data.to(device), np.squeeze(gt_boxes), labels
                gt_boxes = np.squeeze(gt_boxes.numpy())
                labels = np.squeeze(labels.numpy())
                im_h, im_w = im_data.size()[2], im_data.size()[3]

                '''Process anchor'''
                '''Make Anchor boxes and label them based on thresholds'''
                anchors, anchor_labels, anchor_locations = self.process_anchor(im_h=im_h, im_w=im_w, bbox=gt_boxes, sub_sample=self.sub_sample,
                                                                      ratios=self.anchor_ratios, scales=self.anchor_scales,
                                                                      pos_iou_threshold=self.pos_iou_threshold,
                                                                      neg_iou_threshold=self.neg_iou_threshold)


                '''RPN network forward'''
                out_feature = self.feature_extractor(im_data)
                pred_anchor_locs, pred_cls_scores = self.rpn(out_feature)

                '''reformat to align with our anchor targets'''
                pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
                pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
                objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
                pred_cls_scores = pred_cls_scores.view(1, -1, 2)

                '''RPN loss'''
                rpn_loc = pred_anchor_locs[0]
                rpn_score = pred_cls_scores[0]

                gt_rpn_loc = torch.from_numpy(anchor_locations).type(torch.FloatTensor).cuda()
                gt_rpn_score = torch.from_numpy(anchor_labels).cuda()

                rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)

                pos = gt_rpn_score > 0
                mask = pos.unsqueeze(1).expand_as(rpn_loc)
                mask_loc_preds = rpn_loc[mask].view(-1, 4)
                mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

                x = torch.abs(mask_loc_targets - mask_loc_preds)
                rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))

                rpn_lambda = 10.
                N_reg = (gt_rpn_score > 0).float().sum()
                rpn_loc_loss = rpn_loc_loss.sum() / N_reg

                rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)


                '''anchor proposal'''
                roi, after_delta = self.anchor_proposal(img_size=(im_h, im_w), anchors=anchors,
                                           pred_anchor_locs=pred_anchor_locs[0].cpu().data.numpy(),
                                           objectness_score=objectness_score[0].cpu().data.numpy(),
                                           pre_nms_thresh=self.n_test_pre_nms,
                                           post_nms_thresh=self.n_test_post_nms)


                '''iou of each ground truth object with the region proposals, 
                We will use the same code we have used in Anchor boxes to calculate the ious'''
                sample_roi, gt_roi_labels, gt_roi_locs = self.random_choice_roi(roi=roi, gt_boxes=gt_boxes,
                                                                                labels=labels, n_sample=32, pos_ratio=1.)
                # sample_roi, gt_roi_labels, gt_roi_locs = self.choice_roi(roi=roi, gt_boxes=gt_boxes,
                #                                                                 labels=labels)

                ##########################################################
                rois = torch.from_numpy(sample_roi).float()
                roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
                roi_indices = torch.from_numpy(roi_indices).float()

                indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
                xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]

                indices_and_rois = xy_indices_and_rois.contiguous()

                output = []
                rois = indices_and_rois.data.float()
                rois[:, 1:].mul_(1 / self.sub_sample)  # Subsampling ratio
                rois = rois.long()
                num_rois = rois.size(0)
                for i in range(num_rois):
                    roi = rois[i]
                    im_idx = roi[0]
                    im = out_feature.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
                    output.append(self.adaptive_max_pool(im))

                output = torch.cat(output, 0)
                k = output.view(output.size(0), -1)

                '''fast rcnn forward'''
                k = self.roi_head_classifier(k)
                roi_cls_loc = self.cls_loc(k)
                roi_cls_score = self.score(k)

                '''Fast RCNN loss'''
                gt_roi_locs = torch.from_numpy(gt_roi_locs).type(torch.FloatTensor).cuda()
                gt_roi_labels = torch.from_numpy(np.float32(gt_roi_labels)).long().cuda()

                roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_labels, ignore_index=-1)
                n_sample = roi_cls_loc.shape[0]
                roi_loc = roi_cls_loc.view(n_sample, -1, 4)
                roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_labels]

                ################################################################
                roi_pos = gt_roi_labels > 0
                roi_mask = roi_pos.unsqueeze(1).expand_as(roi_loc)

                roi_mask_loc_preds = roi_loc[roi_mask].view(-1, 4)
                roi_mask_loc_targets = gt_roi_locs[roi_mask].view(-1, 4)

                x = torch.abs(roi_mask_loc_targets - roi_mask_loc_preds)
                roi_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))

                roi_lambda = 10.
                N_reg = (gt_roi_labels > 0).float().sum() + 1
                roi_loc_loss = roi_loc_loss.sum() / N_reg

                #################################################################
                roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)


                scores = F.softmax(roi_cls_score, 1).cpu().data.numpy()
                box_deltas = roi_loc.cpu().data.numpy()
                im = im_data.cpu().data.numpy()

                self.visualize(image=im, scores=scores[:, 1:], boxes=sample_roi, box_deltas=box_deltas)


    def fill_anchor_base(self, anchor_base=None, sub_sample=None, anchor_ratios=None, anchor_scales=None):
        ctr_y = sub_sample / 2.
        ctr_x = sub_sample / 2.
        # print(ctr_y, ctr_x)
        # Out: (8, 8)
        for i in range(len(anchor_ratios)):
            for j in range(len(anchor_scales)):
                h = sub_sample * anchor_scales[j] * np.sqrt(anchor_ratios[i])
                w = sub_sample * anchor_scales[j] * np.sqrt(1. / anchor_ratios[i])

                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = ctr_y - h / 2.
                anchor_base[index, 1] = ctr_x - w / 2.
                anchor_base[index, 2] = ctr_y + h / 2.
                anchor_base[index, 3] = ctr_x + w / 2.
        return anchor_base

    def full_anchor(self, size, sub_sample=None, ratios=None, scales=None):
        fe_size = (size // sub_sample)
        ctr_x = np.arange(sub_sample, (fe_size+1) * sub_sample, sub_sample)
        ctr_y = np.arange(sub_sample, (fe_size+1) * sub_sample, sub_sample)
        ctr = np.zeros([len(ctr_x)*len(ctr_y),2])
        index = 0
        for x in range(len(ctr_x)):
            for y in range(len(ctr_y)):
                ctr[index, 1] = ctr_x[x] - sub_sample/2
                ctr[index, 0] = ctr_y[y] - sub_sample/2
                index += 1
        anchors = np.zeros([len(ratios)*len(scales)*len(ctr_x)*len(ctr_y), 4])
        index = 0
        for c in ctr:
            ctr_y, ctr_x = c
            for i in range(len(ratios)):
                for j in range(len(scales)):
                    h = sub_sample * scales[j] * np.sqrt(ratios[i])
                    w = sub_sample * scales[j] * np.sqrt(1. / ratios[i])
                    anchors[index, 0] = ctr_y - h / 2.
                    anchors[index, 1] = ctr_x - w / 2.
                    anchors[index, 2] = ctr_y + h / 2.
                    anchors[index, 3] = ctr_x + w / 2.
                    index += 1

        return anchors

    def iou_calc(self, bbox=None, anchor_valid=None):
        ious = np.empty((len(anchor_valid), 2), dtype=np.float32)
        ious.fill(0)
        for num1, i in enumerate(anchor_valid):
            ya1, xa1, ya2, xa2 = i
            anchor_area = (ya2 - ya1) * (xa2 - xa1)
            for num2, j in enumerate(bbox):
                yb1, xb1, yb2, xb2 = j
                box_area = (yb2 - yb1) * (xb2 - xb1)
                inter_x1 = max([xb1, xa1])
                inter_y1 = max([yb1, ya1])
                inter_x2 = min([xb2, xa2])
                inter_y2 = min([yb2, ya2])

                if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                    inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                    iou = inter_area / (anchor_area + box_area - inter_area)
                else:
                    iou = 0.
                ious[num1, num2] = iou
        return ious

    def process_anchor(self, im_h=None, im_w=None, bbox=None, sub_sample=None, ratios=None, scales=None, pos_iou_threshold=None, neg_iou_threshold=None):
        anchors = self.full_anchor(size=im_h, sub_sample=sub_sample, ratios=ratios, scales=scales)
        '''label -1 with labels which are out of image'''
        anchor_valid_index = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= im_h) &
            (anchors[:, 3] <= im_w)
        )[0]
        label = np.empty((len(anchor_valid_index),), dtype=np.int32)
        label.fill(-1)
        anchor_valid = anchors[anchor_valid_index]

        ious = self.iou_calc(bbox=bbox, anchor_valid=anchor_valid)

        '''for each anchor, which box is closer'''
        gt_argmax_ious = ious.argmax(axis=0)

        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]

        '''for each anchor, which box is closer'''
        argmax_ious = ious.argmax(axis=1)

        max_ious = ious[np.arange(len(ious)), argmax_ious]

        # find number of anchor which has biggest iou between bbox
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        '''Assign anchor labels by thresholding'''
        label[max_ious < neg_iou_threshold] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= pos_iou_threshold] = 1

        '''To make dataset balance'''
        pos_ratio = 0.5
        n_sample = 256
        n_pos = pos_ratio * n_sample
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = n_sample * np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        max_iou_bbox = bbox[argmax_ious]

        '''Convert [y1,x1,y2,x2] => [dy, dx, dh, dw]'''
        height = anchor_valid[:, 2] - anchor_valid[:, 0]
        width = anchor_valid[:, 3] - anchor_valid[:, 1]
        ctr_y = anchor_valid[:, 0] + 0.5 * height
        ctr_x = anchor_valid[:, 1] + 0.5 * width
        base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
        base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
        base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
        base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

        eps = np.finfo(height.dtype).eps
        height = np.maximum(height, eps)
        width = np.maximum(width, eps)
        dy = (base_ctr_y - ctr_y) / height
        dx = (base_ctr_x - ctr_x) / width
        dh = np.log(base_height / height)
        dw = np.log(base_width / width)
        anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()

        '''anchor labels'''
        anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
        anchor_labels.fill(-1)
        anchor_labels[anchor_valid_index] = label

        '''anchor location => [dy, dx, dh, dw]'''
        anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
        anchor_locations.fill(0)
        anchor_locations[anchor_valid_index, :] = anchor_locs


        return anchors, anchor_labels, anchor_locations


    def non_max_suppression_fast(self, boxes, probs, overlap_thresh):
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = probs.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= overlap_thresh)[0]
            order = order[inds + 1]
        return keep


    def anchor_proposal(self, img_size, anchors, pred_anchor_locs, objectness_score, pre_nms_thresh, post_nms_thresh):
        anc_height = anchors[:, 2] - anchors[:, 0]
        anc_width = anchors[:, 3] - anchors[:, 1]
        anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
        anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

        # print(anc_height.shape)

        pred_anchor_locs_numpy = pred_anchor_locs
        objectness_score_numpy = objectness_score
        dy = pred_anchor_locs_numpy[:, 0::4]
        dx = pred_anchor_locs_numpy[:, 1::4]
        dh = pred_anchor_locs_numpy[:, 2::4]
        dw = pred_anchor_locs_numpy[:, 3::4]
        ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
        ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
        h = np.exp(dh) * anc_height[:, np.newaxis]
        w = np.exp(dw) * anc_width[:, np.newaxis]

        '''Convert [ctr_x, ctr_y, h, w] to [y1, x1, y2, x2] format'''
        '''ROI'''
        roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=anchors.dtype)
        # print(roi.shape)
        roi[:, 0::4] = ctr_y - 0.5 * h
        roi[:, 1::4] = ctr_x - 0.5 * w
        roi[:, 2::4] = ctr_y + 0.5 * h
        roi[:, 3::4] = ctr_x + 0.5 * w

        # img_size = (im_h, im_w)  # Image size
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        after_delta = roi

        '''Remove predicted boxes with either height or width < threshold.'''
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= self.min_size) & (ws >= self.min_size))[0]
        roi = roi[keep, :]

        score = objectness_score_numpy[keep]

        '''sort by object prop'''
        order = score.ravel().argsort()[::-1]

        order = order[:pre_nms_thresh]
        roi_total = roi
        roi = roi[order, :]

        '''Take Top pos_nms_topN'''
        keep = self.non_max_suppression_fast(boxes=roi_total, probs=score, overlap_thresh=self.nms_thresh)

        keep = keep[:post_nms_thresh]  # while training/testing , use accordingly
        roi = roi_total[keep]  # the final region proposals
        return roi, after_delta

    def random_choice_roi(self, roi, gt_boxes, labels, n_sample, pos_ratio):
        n_sample = n_sample
        pos_ratio = pos_ratio
        pos_iou_thresh = 0.6
        neg_iou_thresh_hi = 0.3
        neg_iou_thresh_lo = 0.0

        ious = self.iou_calc(bbox=gt_boxes, anchor_valid=roi)
        gt_assignment = ious.argmax(axis=1)
        max_iou = ious.max(axis=1)

        gt_roi_label = labels[gt_assignment]

        pos_index = np.where(max_iou >= pos_iou_thresh)[0]
        pos_roi_per_this_image = n_sample * pos_ratio
        pos_roi_per_this_image = int(min(pos_roi_per_this_image, pos_index.size))

        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < neg_iou_thresh_hi) &
                             (max_iou >= neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        gt_roi_labels = gt_roi_label[keep_index]
        gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0

        sample_roi = roi[keep_index]

        bbox_for_sampled_roi = gt_boxes[gt_assignment[keep_index]]

        height = sample_roi[:, 2] - sample_roi[:, 0]
        width = sample_roi[:, 3] - sample_roi[:, 1]
        ctr_y = sample_roi[:, 0] + 0.5 * height
        ctr_x = sample_roi[:, 1] + 0.5 * width

        base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
        base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
        base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
        base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width

        eps = np.finfo(height.dtype).eps
        height = np.maximum(height, eps)
        width = np.maximum(width, eps)

        dy = (base_ctr_y - ctr_y) / height
        dx = (base_ctr_x - ctr_x) / width
        dh = np.log(base_height / height)
        dw = np.log(base_width / width)

        gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()

        return sample_roi, gt_roi_labels, gt_roi_locs

    def visualize(self, image, scores, boxes, box_deltas):
        sorted_index = scores.ravel().argsort()[::-1]
        [idx1, idx2] = sorted_index[:2]
        idx = [idx1, idx2]
        print(np.max(scores))
        print(np.mean(scores))

        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        ctr_y = boxes[:, 0] + 0.5 * height
        ctr_x = boxes[:, 1] + 0.5 * width

        dy = box_deltas[:,0]
        dx = box_deltas[:,1]
        dh = box_deltas[:,2]
        dw = box_deltas[:,3]

        ctr_y_modified = dy * height + ctr_y
        ctr_x_modified = dx * width + ctr_x
        height_modified = np.exp(dh)*height
        width_modified = np.exp(dw)*width

        y1 = ctr_y_modified - height_modified / 2.
        x1 = ctr_x_modified - width_modified / 2.

        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(image), cmap='gray')
        for i in range(len(idx)):
            k = idx[i]
            xy = (x1[k//10], y1[k//10])

            width_tmp = width_modified[k//10]

            height_tmp = height_modified[k//10]

            rect = patches.Rectangle(xy=xy, width=width_tmp, height=height_tmp, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.annotate(str(k - k//10*10), xy=(xy[0], xy[1]), color='magenta')
        plt.savefig('result.png')

    def save_rpn(self):
        torch.save(self.feature_extractor.state_dict(), self.args.checkpoint_dir + '/feature_extractor.pth')
        torch.save(self.rpn.state_dict(), self.args.checkpoint_dir + '/rpn.pth')

    def save_frcnn(self):
        torch.save(self.feature_extractor.state_dict(), self.args.checkpoint_dir + '/feature_extractor.pth')
        torch.save(self.roi_head_classifier.state_dict(), self.args.checkpoint_dir + '/roi_head_classifier.pth')
        torch.save(self.cls_loc.state_dict(), self.args.checkpoint_dir + '/cls_loc.pth')
        torch.save(self.score.state_dict(), self.args.checkpoint_dir + '/score.pth')

    def load_rpn(self):
        self.feature_extractor.load_state_dict(
            torch.load(self.args.checkpoint_dir +'/feature_extractor.pth'))
        self.rpn.load_state_dict(
            torch.load(self.args.checkpoint_dir + '/rpn.pth'))

    def load_frcnn(self):
        self.feature_extractor.load_state_dict(
            torch.load(self.args.checkpoint_dir +'/feature_extractor.pth'))
        self.roi_head_classifier.load_state_dict(
            torch.load(self.args.checkpoint_dir + '/roi_head_classifier.pth'))
        self.cls_loc.load_state_dict(
            torch.load(self.args.checkpoint_dir + '/cls_loc.pth'))
        self.score.load_state_dict(
            torch.load(self.args.checkpoint_dir + '/score.pth'))













