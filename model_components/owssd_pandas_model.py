# ---
# PANDAS
# Copyright (C) 2023 NAVER Corp.
# CC BY-NC-SA 4.0 license
# ---

import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
from collections import defaultdict
import time
import pickle
from torch import optim
from torch.nn import functional as F
from torchvision.models.detection.faster_rcnn import _resnet_fpn_extractor, TwoMLPHead, MultiScaleRoIAlign
from torchvision.models import resnet50

from engine import evaluate_coco, evaluate_lvis
from .rcnn_components import RoIHeadsModified, get_box_features
from .faiss_kmeans import Faiss2SklearnKMeans
from .faster_rcnn_predictors import FasterRCNNPredictorOrig, FasterRCNNPredictorNCDMaskOrig, \
    FastRCNNPredictorClassAgnosticRegressor
from hungarian_matching import cluster_map_fn
from sklearn.metrics import recall_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import umap
from umap import plot as umap_plot
from pathlib import Path


"""
1) We are no longer planning to perform clustering using prototypes
2) That being said, there is no point in training any prototypes at all since we are using the VAE ensemble method
3) Have tried to modify code accordingly...and for the sake of the codebase consider prototype to be another way of calling VAE ensemble
"""


def pd_mat(input1, input2):
    return 0.5 * torch.cdist(input1, input2) ** 2

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.encoder1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder2 = nn.Linear(
            in_features=512, out_features=256
        )
        self.encoder3 = nn.Linear(
            in_features=256, out_features=128
        )
        self.encoder_mu = nn.Linear(
            in_features=128, out_features=128
        )
        self.encoder_logvar = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder1 = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder2 = nn.Linear(
            in_features=128, out_features=256
        )
        self.decoder3 = nn.Linear(
            in_features=256, out_features=512
        )
        self.decoder4 = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def encode(self, features):
        encode1_op = F.relu(self.encoder1(features))
        encode2_op = F.relu(self.encoder2(encode1_op))
        value = F.relu(self.encoder3(encode2_op))
        mu = self.encoder_mu(value)
        logvar = self.encoder_logvar(value)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def decode(self, z):
        decoder1_op = F.relu(self.decoder1(z))
        decoder2_op = F.relu(self.decoder2(decoder1_op))
        decoder3_op = F.relu(self.decoder3(decoder2_op))
        value = self.decoder4(decoder3_op)
        reconstructed = torch.sigmoid(value)
        return reconstructed

    def forward(self, features):
        mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def loss_function(reconstructed, features, mu, logvar):
    # Debugging: Check the range of reconstructed and features
    print("Reconstructed min:", reconstructed.min().item(), "max:", reconstructed.max().item())
    print("Features min:", features.min().item(), "max:", features.max().item())
    
    # Ensure values are in the [0, 1] range
    assert (reconstructed >= 0).all() and (reconstructed <= 1).all(), "Reconstructed values out of range"
    assert (features >= 0).all() and (features <= 1).all(), "Features values out of range"

    # Binary Cross Entropy Loss
    BCE = F.binary_cross_entropy(reconstructed, features, reduction='sum')
    
    # Kullback-Leibler Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total Loss
    return BCE + KLD

class OWSSDModel(object):
    """
    PANDAS NCD object detection model that uses prototype learning, that has been modified to use a VAE ensemble.
    Here each vae model is trained on one of the extracted features and then we use this enseble to find novelty as in OWSSD paper
    """

    def __init__(self, num_classes, num_base_classes, base_checkpoint, feature_dim=1024, device='cuda:1',
                 l2_normalize=False, ncd_checkpoint=None, num_clusters=21, background_classifier='softmax',
                 similarity_metric='invert_square', proba_norm='l1', prototype_init='cluster_all', class_names=None,
                 base_ids=None, novel_ids=None, dataset=None, dets_per_img=100, score_thresh=0.05,
                 output_dir=None, kmeans_n_init=10, kmeans_max_iter=1000, base_score_thresh=0.05,
                 base_dets_per_img=100, kmeans_seed=42, save_cluster_mapping=None, last_free_class_id=10000,
                 max_class_num=20000):

        # setup model parameters
        self.num_classes = num_classes
        self.num_base_classes = num_base_classes
        self.checkpoint = base_checkpoint
        self.device = device
        self.l2_normalize = l2_normalize
        self.feature_size = feature_dim
        self.num_clusters = num_clusters
        self.similarity_metric = similarity_metric
        self.proba_norm = proba_norm
        self.background_classifier = background_classifier
        self.ncd_checkpoint = ncd_checkpoint
        self.prototype_init = prototype_init # Not changing so as to not worry about the function call
        self.n_init = kmeans_n_init
        self.max_iter = kmeans_max_iter
        self.base_ids = base_ids
        self.novel_ids = novel_ids  # 1 based indices of novel classes
        self.dataset = dataset
        self.dets_per_img = dets_per_img
        self.score_thresh = score_thresh
        self.output_dir = output_dir
        self.base_score_thresh = base_score_thresh  # before clustering, use this threshold for boxes
        self.base_dets_per_img = base_dets_per_img  # before clustering, use this threshold for boxes
        self.kmeans_seed = kmeans_seed
        self.save_cluster_mapping = save_cluster_mapping

        print('Device: ', self.device)
        self.class_names = class_names
        if class_names is not None:
            print('Class Names Key: ', class_names)

        self.od_model = self.get_object_detection_model_fpn(num_base_classes, base_checkpoint, freeze_model=True)

        # initialize prototype array
        self.class_prototypes = torch.zeros(
            (num_classes - 1, feature_dim)).to(self.device)

        # initialize ncd prediction model and cluster map for evaluation
        self.ncd_prediction_model = None
        self.cluster_map = None

        # for hungarian assignment
        self.last_free_class_id = last_free_class_id  # maximum total number of classes in the dataset
        self.max_class_num = max_class_num  # can be used as a class ID for novel category that is guaranteed to be
        #                                      higher than the number of known classes; used to assign IDs to novel classes and to avoid
        #                                      IDs overlapping

    def initialize_model(self, discovery_loader=None, novel_loader=None, base_loader=None, test_loader=None):
        print("You have entered the initialize function of the OWSSD Pandas model - congratulations")
        if self.ncd_checkpoint is not None:
            # load in model from ckpt
            print('\nInitializing model using: NCD Checkpoint')
            print("NOT SURE WHY THIS RAN...NCD CHECKPOINT DEFAULT VALUE IS NONE...TERMINATING")
            exit()
            self.load_model(self.ncd_checkpoint)
        else:
            if self.prototype_init == 'gt_prototypes': # I suppose gt stands for ground truth protoypes
                print('\nInitializing model using: GT Prototypes')
                features_gt, labels_gt, _, _ = self.get_all_box_features_normalize(self.od_model, discovery_loader)
                print("WHY HAS THE CODE ENTERED THE GT PROTOTYPE ZONE INSTEAD OF VAE CALCULATION- TERMINATING")
                print("AS PER THE ORIGINAL PANDAS CODE, THIS PART WOULD SIGNIFY - prototype_init == gt_prototypes")
                print("AS PER OUT NEW METHOD WE NEED THE CODE TO CALCULATE THE BASE VAE ENSEMBLE FIRST")
                exit()
                gt_prototypes, labels_prototypes = self.compute_prototypes_from_features(features_gt, labels_gt)
                # since there might be missing labels or labels could be out of order,
                # let's put the prototypes in the correct spot in the original array
                for ii, lab in enumerate(labels_prototypes):
                    self.class_prototypes[lab - 1] = gt_prototypes[ii]
            else:
                print('\nInitializing model using: %s' % self.prototype_init)
                if self.prototype_init == 'pandas':
                    # cluster novel data and use gt base prototypes
                    self.initialize_model_cluster_novel_use_gt_owssd(self.od_model, base_loader, novel_loader, test_loader)
                else:
                    print('Extracting features...')
                    print("YOU HAVE ENTERED THE OWSSD PANDAS MODEL PYTHON FILE FOR MODEL INITIALIZATION")
                    print("THIS PART IS RUN FOR EXPERIMENTS THAT ARE NOT BASED ON PANDAS METHOD AS A COMPARISON")
                    print("If need to run experiment arises - edit owssd_pandas_model.py to remove this TERMINATION")
                    exit()
                    _, _, features, _ = self.get_all_box_features_normalize(self.od_model, discovery_loader,
                                                                            included_features=['rpn'])
                    # cluster all features
                    self.initialize_model_offline_clustering(features)
        self.num_prototypes = len(self.class_prototypes) # This line would technically be meaningless right since we have no prototypes?

        # after initialization, change the score threshold and max detections per image
        print('\nChanging score threshold and maximum detections per image...')
        self.od_model.roi_heads.score_thresh = self.score_thresh
        self.od_model.roi_heads.detections_per_img = self.dets_per_img

    def get_object_detection_model_fpn(self, num_base_classes, base_ckpt, freeze_model):
        backbone = resnet50(pretrained=False)
        backbone.out_channels = 2048
        out_channels = 256
        # box roi pool
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        print('\nUsing Class Agnostic Regressor...')
        box_predictor = FastRCNNPredictorClassAgnosticRegressor(self.feature_size, num_base_classes)

        # box head
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)
        roi_heads = RoIHeadsModified(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512, box_positive_fraction=0.25,
            bbox_reg_weights=None,
            box_score_thresh=self.base_score_thresh, box_nms_thresh=0.5, box_detections_per_img=self.base_dets_per_img,
            proba_norm=self.proba_norm)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            trainable_backbone_layers=3, num_classes=num_base_classes)
        model.backbone = _resnet_fpn_extractor(backbone, 3, norm_layer=nn.BatchNorm2d)
        model.roi_heads = roi_heads

        if base_ckpt is None:
            print('Base checkpoint not loaded!!!')
        else:
            print('\nLoading checkpoint for object detection model from: %s' % base_ckpt)
            base_ckpt = torch.load(base_ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(base_ckpt['model'])

        model = model.to(self.device)
        model.eval()

        if freeze_model:
            print('Freezing model')
            # freeze Faster RCNN
            for name, param in model.named_parameters():
                param.requires_grad = False

        return model

    # def compute_prototypes_from_features(self, features_gt, labels_gt):
    #     # compute gt prototypes
    #     print('Computing GT prototypes...')
    #     gt_prototypes = []
    #     labels_for_prototypes = []
    #     for l in range(self.num_classes - 1):
    #         curr_lab = l + 1
    #         ixs = torch.where(labels_gt == curr_lab)[0]
    #         if len(ixs) == 0:
    #             print('No features for label %d found.' % curr_lab)
    #         else:
    #             curr_feat = features_gt[ixs, :]
    #             gt_prototypes.append(torch.mean(curr_feat, dim=0).unsqueeze(0))
    #             labels_for_prototypes.append(curr_lab)  # labels are 1-based here
    #     gt_prototypes = torch.cat(gt_prototypes, dim=0)
    #     return gt_prototypes, labels_for_prototypes

    def initialize_model_cluster_novel_use_gt_owssd(self, model, base_loader, novel_loader, test_loader):
        # pandas method: gt prototypes for base classes and clusters for novel data
        # owssd_pandas method:
        """
        1) Train an Ensemble on the base classes, save the VAE to pth files
        2) TODO: Pending...
        """
        print('Extracting features...')
        
        feat_file = Path("features_gt.pt")
        if feat_file.exists():
            features_gt = torch.load('features_gt.pt')
            labels_gt = torch.load('labels_gt.pt')
        else:
            features_gt, labels_gt, _, _ = self.get_all_box_features_normalize(model, base_loader)
            torch.save(features_gt, 'features_gt.pt')
            torch.save(labels_gt, 'labels_gt.pt')

        print("base loader len: ", len(base_loader))
        print("novel loader len: ", len(novel_loader))
        print("features_gt: ", len(features_gt))
        print("labels_gt: ", len(labels_gt))
        
        thresholds = {}
        for class_name in self.class_names.keys(): # Imp to note that this for loop allows us to iterate over class keys names....therefore we will be able to train individual vae's
            if self.class_names[class_name][1] == 'base':
                class_indices = (labels_gt == class_name).nonzero().squeeze()
                class_labels_gt = torch.index_select(labels_gt, 0, class_indices)
                class_features_gt = torch.index_select(features_gt, 0, class_indices)
                class_features_gt = class_features_gt.to(self.device)
                print("class_labels_gt: ", len(class_labels_gt))
                print("class_features_gt: ", len(class_features_gt))
                
                X_train, X_test, y_train, y_test = train_test_split(class_features_gt,
                                                    class_labels_gt,
                                                    test_size=0.33,
                                                    random_state=42)

                X_train = normalize_data(X_train)
                X_test = normalize_data(X_test)

                vae_model = VAE(input_shape=1024).to(self.device)
                optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)
                epochs = 40 # Stabilizes around 35 epochs
                train_losses, val_losses = [], []

                for epoch in range(epochs):
                    vae_model.train()
                    optimizer.zero_grad()
                    reconstructed, mu, logvar = vae_model(X_train)
                    train_loss = loss_function(reconstructed, X_train, mu, logvar)
                    train_loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        vae_model.eval()
                        reconstructed_val, mu_val, logvar_val = vae_model(X_test)
                        val_loss = loss_function(reconstructed_val, X_test, mu_val, logvar_val)

                    train_losses.append(train_loss.item())
                    val_losses.append(val_loss.item())

                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.8f}, Val Loss: {val_losses[-1]:.8f}")

                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(f'vae_training_curves/vae_{class_name}_loss_curves.jpg', format='jpg', dpi=300, bbox_inches='tight')
                plt.close()

                # Compute the threshold for anomaly detection
                # I suppose this threshold is going to be on a per class basis
                train_losses = []
                with torch.no_grad():
                    for feat_gt in class_features_gt:
                        normalize_feat_gt = normalize_data(feat_gt.unsqueeze(0))
                        reconstructed, mu, logvar = vae_model(normalize_feat_gt)
                        loss = loss_function(reconstructed, normalize_feat_gt, mu, logvar)
                        train_losses.append(loss.item())

                threshold = np.mean(train_losses) + np.std(train_losses)
                print("Threshold: ", threshold)
                thresholds[class_name] = threshold    
                torch.save(vae_model.state_dict(), f"VAE_model_checkpoints/voc_class_{class_name}.pth")
        
        print("CONGRATUILATION...TERMINATING")
        exit()

        feat_test_file = Path("features_test.pt")
        if feat_test_file.exists():
            features_test = torch.load('features_test.pt')
            labels_test = torch.load('labels_test.pt')
        else:
            features_test, labels_test, _, _ = self.get_all_box_features_normalize(model, test_loader)
            torch.save(features_test, 'features_test.pt')
            torch.save(labels_test, 'labels_test.pt')
        
        print("features_test: ", len(features_test))

        preds = []
        binary = []
        for i in range(len(features_test)):
            if i % 100 == 0:
                print("At feature ", i)
            feature_test = features_test[i].to(self.device)
            label_test = labels_test[i].item()
            if self.class_names[label_test][1] == 'base':
                binary.append(0)
            else:
                binary.append(1)
            print("label_test: ", label_test)
            
            inl = []
            for class_name in self.class_names.keys():
                if self.class_names[class_name][1] == 'base':
                    print("class_name: ", class_name)
                    vae_model = VAE(input_shape=1024).to(self.device)
                    vae_model.load_state_dict(torch.load(f"VAE_model_checkpoints/voc_class_{class_name}.pth"))
                    vae_model = vae_model.to(self.device)
                    vae_model.eval()
                    with torch.no_grad():
                        reconstructed, mu, logvar = vae_model(feature_test.unsqueeze(0))
                        loss = loss_function(reconstructed, feature_test.unsqueeze(0), mu, logvar)
                    print("score: ", loss.item())
                    if loss < thresholds[class_name]:
                        inl.append(1)
                    else:
                        inl.append(0)

            if 1 in inl:
                preds.append(0)
            else: 
                preds.append(1)

        print("len binary: ", len(binary))
        print("len predictions: ", len(preds))

        print("binary: ", binary)
        print("predictions: ",preds)

        tpr = recall_score(binary, preds)
        tnr = recall_score(binary, preds, pos_label = 0) 
        fpr = 1 - tnr
        fnr = 1 - tpr
        print("fpr :", fpr)
        print("fnr :", fnr) 

        auroc = metrics.roc_auc_score(binary, preds)
        print("auroc binary :", auroc)
        print("area under curve (auc) binary: ", auroc)
        print("fpr binary:", fpr)
        print("tpr binary :", tpr)
            
        _, _, features_rpn_novel, _ = self.get_all_box_features_normalize(model, novel_loader, included_features=['rpn'])
        print("features_rpn_novel: ", len(features_rpn_novel))
        # labels_prototypes is 1-based
        #gt_prototypes, labels_prototypes = self.compute_prototypes_from_features(features_gt, labels_gt)
        #self.initialize_model_offline_clustering(features_rpn_novel)

        # use base gt prototypes AND novel rpn clusters
        #self.class_prototypes = torch.cat([gt_prototypes.to(self.device), self.class_prototypes], dim=0)
        #self.base_ids = labels_prototypes  # set the labels that correspond with the gt prototypes
        #self.num_base_classes = len(labels_prototypes)

    def initialize_model_offline_clustering(self, features):
        print('Performing clustering...')
        features = features.numpy()
        print('about to cluster %d features...' % len(features))
        start_time = time.time()
        feature_size = features.shape[1]
        kmeans = Faiss2SklearnKMeans(feature_size, k=self.num_clusters, nredo=self.n_init, niter=self.max_iter,
                             seed=self.kmeans_seed)
        kmeans.fit(features)
        # assign prototypes as cluster centers
        self.class_prototypes = torch.from_numpy(kmeans.centroids).to(self.device)
        print('time to cluster: ', (time.time() - start_time))

    def extract_normalize_bbox_features_batch(self, model, images, targets_, included_features):
        # extracts bounding box features from a mini-batch (images, targets)
        with torch.no_grad():
            images = list(image.to(self.device) for image in images)
            labels_gt = [targets_[ii]['labels'] for ii in range(len(targets_))]
            labels_gt = torch.cat(labels_gt)
            targets = []
            for t in targets_:
                d = {}
                for k, v in t.items():
                    if torch.is_tensor(v):
                        d[k] = v.to(self.device)
                    elif k in ['size', 'image_id', 'height', 'width']:
                        d[k] = v
                targets.append(d)

            if 'gt' in included_features:
                box_feats_gt, _, _ = get_box_features(model, images, targets, return_logits=True)
                if self.l2_normalize:
                    box_feats_gt = torch.nn.functional.normalize(box_feats_gt)
            else:
                box_feats_gt = None

            if 'rpn' in included_features:
                outputs_ = model(images)
                box_feats_rpn, logits_rpn, regress_coords_rpn = get_box_features(model, images, outputs_,
                                                                                 return_logits=True)
                labels_rpn = torch.argmax(logits_rpn, dim=1)

                if self.l2_normalize:
                    box_feats_rpn = torch.nn.functional.normalize(box_feats_rpn)
            else:
                box_feats_rpn = None
                labels_rpn = None
                regress_coords_rpn = None

            return box_feats_gt, labels_gt, box_feats_rpn, labels_rpn, regress_coords_rpn

    def get_all_box_features_normalize(self, model, loader, included_features=['gt', 'rpn']):
        gt_features_list = []
        gt_labels_list = []
        rpn_features_list = []
        rpn_labels_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                if i % 20 == 0:
                    print('%d/%d' % (i, len(loader)))

                box_feats_gt_, labels_gt_, box_feats_rpn, labels_rpn, _ = self.extract_normalize_bbox_features_batch(
                    model, images, targets, included_features)
                
                if box_feats_gt_ is not None:
                    gt_features_list.append(box_feats_gt_.cpu())
                if labels_gt_ is not None:
                    gt_labels_list.append(labels_gt_.cpu())
                if box_feats_rpn is not None:
                    rpn_features_list.append(box_feats_rpn.cpu())
                if labels_rpn is not None:
                    rpn_labels_list.append(labels_rpn.cpu())

        if gt_features_list != []:
            features_gt = torch.cat(gt_features_list, dim=0)
        else:
            features_gt = None
        if gt_labels_list != []:
            labels_gt = torch.cat(gt_labels_list, dim=0)
        else:
            labels_gt = None
        if rpn_features_list != []:
            features_rpn = torch.cat(rpn_features_list, dim=0)
        else:
            features_rpn = None
        if rpn_labels_list != []:
            labels_rpn = torch.cat(rpn_labels_list, dim=0)
        else:
            labels_rpn = None

        return features_gt, labels_gt, features_rpn, labels_rpn

    def remap_labels(self, labels):
        # map labels to consecutive values starting at 0
        v = np.unique(labels)
        labels2 = np.zeros_like(labels)
        mapping = {}
        for i in range(len(v)):
            mapping[i] = v[i]
            ix = np.where(labels == v[i])[0]
            labels2[ix] = i
        return labels2, mapping

    def get_ncd_prediction_model(self, test_loader):
        if self.ncd_prediction_model is None:
            self.ncd_prediction_model = self.make_ncd_prediction_model(test_loader=test_loader)
        return self.ncd_prediction_model

    def evaluate(self, loader):
        with torch.no_grad():
            self.get_ncd_prediction_model(loader)

            print(f'EVALUATING...')
            if self.dataset == 'lvis':
                evaluate_lvis(self.ncd_prediction_model, loader, device=self.device, class_names=self.class_names,
                              save_dir=self.output_dir)
            else:
                evaluate_coco(self.ncd_prediction_model, loader, device=self.device, class_names=self.class_names)

    def compute_closest_prototypes(self, feats, ixs=None, ix_shift=0):
        # ixs: torch tensor of indices of novel clusters

        if ixs is None:
            logits = self.compute_similarity(feats, self.class_prototypes)
        else:
            # only find nearest neighbors of subset
            logits = self.compute_similarity(feats, self.class_prototypes[ixs])
        values, indices = logits.max(dim=1)
        # convert current subset indices back to original indices
        indices += ix_shift
        return values, indices

    def compute_similarity(self, x, M, eps=1e-12):
        if self.similarity_metric == 'cosine':
            logits = F.linear(F.normalize(x, p=2, dim=1), \
                              F.normalize(M, p=2, dim=1))
        elif self.similarity_metric == 'dot_prod':
            M = M.transpose(1, 0)
            c = 0.5 * torch.sum(M * M, dim=0)
            logits = torch.matmul(x, M) - c
        else:
            dists = pd_mat(x, M.to(self.device))
            if self.similarity_metric == 'invert':
                logits = (1 / (dists + eps))
            elif self.similarity_metric == 'invert_square':
                logits = (1 / ((dists ** 2) + eps))
            else:
                raise NotImplementedError
        return logits

    def predict_scores(self, x):
        logits = self.compute_similarity(x, self.class_prototypes)
        return logits

    def make_ncd_prediction_model(self, test_loader=None):
        prediction_model = self.od_model

        # compute cluster mapping using hungarian matching (if not using oracle prototypes)
        if self.prototype_init != 'gt_prototypes':
            if self.cluster_map is None:
                if self.prototype_init == 'pandas':
                    cluster_map = self.get_class_mapping_hungarian(test_loader, self.num_classes - 1,
                                                                   save_mapping=self.save_cluster_mapping,
                                                                   gt_prototypes=True)
                else:
                    cluster_map = self.get_class_mapping_hungarian(test_loader, self.num_classes - 1,
                                                                   save_mapping=self.save_cluster_mapping)
                self.cluster_map = cluster_map
            else:
                cluster_map = self.cluster_map
        else:
            cluster_map = None

        # get predictor model (using appropriate background classifier)
        if self.background_classifier == 'softmax':
            predictor = FasterRCNNPredictorNCDMaskOrig(bbox_pred=prediction_model.roi_heads.box_predictor.bbox_pred,
                                                       prediction_model=self,
                                                       cluster_mapping=cluster_map, l2_normalize=self.l2_normalize,
                                                       cls_score_orig=prediction_model.roi_heads.box_predictor.cls_score,
                                                       num_classes=self.num_classes,
                                                       mask_type='orig_model',
                                                       device=self.device)
            orig_predictor = FasterRCNNPredictorOrig(bbox_pred=prediction_model.roi_heads.box_predictor.bbox_pred,
                                                     cls_score=prediction_model.roi_heads.box_predictor.cls_score,
                                                     num_classes=self.num_classes,
                                                     device=self.device)
            prediction_model.roi_heads.box_predictor_orig = orig_predictor
        elif self.background_classifier == 'none':
            predictor = FasterRCNNPredictorNCDMaskOrig(bbox_pred=prediction_model.roi_heads.box_predictor.bbox_pred,
                                                       prediction_model=self,
                                                       cluster_mapping=cluster_map, l2_normalize=self.l2_normalize,
                                                       cls_score_orig=prediction_model.roi_heads.box_predictor.cls_score,
                                                       num_classes=self.num_classes,
                                                       mask_type='none',
                                                       device=self.device)
        else:
            raise NotImplementedError
        prediction_model.roi_heads.box_predictor = predictor
        return prediction_model.to(self.device)

    def get_class_mapping_hungarian(self, data_loader, num_classes, save_mapping=None, gt_prototypes=False):
        if data_loader is None:
            return None
        # extract gt bbox features and find closest clusters
        print('computing cluster mapping...')
        cluster_dict = defaultdict(list)
        gt_labels_all = []
        cluster_ids_all = []
        for i, (images, targets) in enumerate(data_loader):

            box_feats_gt, labels_gt, _, _, _ = self.extract_normalize_bbox_features_batch(
                self.od_model, images, targets, included_features=['gt'])

            if gt_prototypes:
                cluster_indices = np.arange(self.num_base_classes,
                                            len(self.class_prototypes))  # clusters are always at the bottom of array
                ix_shift = self.num_base_classes  # shift subset indices by this amount to correspond to original self.class_prototypes
                vals, ixs = self.compute_closest_prototypes(box_feats_gt, ixs=cluster_indices,
                                                            ix_shift=ix_shift)
            else:
                vals, ixs = self.compute_closest_prototypes(box_feats_gt)
            for cluster_ix, gt_ix in zip(ixs, labels_gt):
                cluster_dict[cluster_ix].append(gt_ix.cpu().item())
                gt_labels_all.append(gt_ix.cpu().item())
                cluster_ids_all.append(cluster_ix.cpu().item())

        if gt_prototypes:
            # y_true and y_pred need to be filtered based on only novel class GT IDs
            y_true = np.array(gt_labels_all)
            y_pred = np.array(cluster_ids_all)
            ixs_filter = []
            for l in self.novel_ids:
                curr_ixs = np.where(y_true == l)[0]
                ixs_filter.extend(list(curr_ixs))
            ixs_filter = np.array(ixs_filter)
            y_true = y_true[ixs_filter]
            y_pred = y_pred[ixs_filter]

            y_true = y_true - 1  # shift labels by 1
        else:
            y_true = np.array(gt_labels_all) - 1
            y_pred = np.array(cluster_ids_all)
        y_true_mapped, label_mapping = self.remap_labels(y_true)

        # compute cluster mapping using hungarian matching
        cluster_map = cluster_map_fn(y_true_mapped, y_pred, self.last_free_class_id, self.max_class_num)
        mapping2 = np.arange((num_classes))
        mapping2[self.novel_ids - 1] = -1
        for i in cluster_map:
            # i = (cluster_id, label_id)
            if i[1] < num_classes:
                # remap i[1] back to original label value
                cls_val = label_mapping[i[1]]
                mapping2[cls_val] = i[0]

        if gt_prototypes:
            # label for base class corresponds to location of prototype (0-based)
            for proto_i, base_i in enumerate(self.base_ids):
                mapping2[base_i - 1] = proto_i  # making base_i 0-based

        m = mapping2.astype(np.int64)

        if save_mapping is not None:
            print('\nSaving cluster mapping to: %s' % os.path.join(save_mapping + '.pickle'))
            with open(os.path.join(save_mapping + '.pickle'), 'wb') as f:
                pickle.dump(m, f)
        return m

    def save_model(self, ckpt_file):
        """
        Save model parameters to torch file
        """
        print('\nSaving model to %s' % ckpt_file)
        d = dict()
        d['class_prototypes'] = self.class_prototypes.cpu()
        d['cluster_map'] = self.cluster_map
        d['base_ids'] = self.base_ids

        # save model out
        torch.save(d, ckpt_file)

    def load_model(self, ckpt_file):
        print('\nLoading model from %s' % ckpt_file)
        ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
        self.class_prototypes = ckpt['class_prototypes'].to(self.device)
        self.cluster_map = ckpt['cluster_map']

        if 'base_ids' in list(ckpt.keys()):
            self.base_ids = ckpt['base_ids']

        if self.save_cluster_mapping is not None:
            print('\nLoading cluster mapping from %s' % self.save_cluster_mapping)
            # if cluster map is saved in another file, load it here
            with open(self.save_cluster_mapping, 'rb') as pickle_file:
                cluster_map = pickle.load(pickle_file)
            self.cluster_map = cluster_map
