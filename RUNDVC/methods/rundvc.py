import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math

USE_FOCAL = False

class FocalLoss(nn.Module):
    def __init__(self,  reduction="mean" ,gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        input =  torch.clamp(input, 1e-9, 1-1e-9)
        cross_entropy = -target * torch.log(input)
        cross_entropy = cross_entropy.sum(-1)  # [N, label_channel]
        loss = ((1-input)**self.gamma) * target
        loss = loss.sum(-1)
        loss = loss * cross_entropy 
        if self.reduction == "none":
            return loss
        else:
            return loss.mean()

def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        ema_avg = lambda avg_model_param, model_param, num_averaged: \
             decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device=device, avg_fn=ema_avg)

class RUNDVC():
    """
    RUNDVC transfer learning for deep learning based variant callers
    """

    def __init__(self, encoder, classifier):
        """
        Arguments:
        ----------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)


        
        
        self.start_epoch = 0
        self.current_step = 0
        self.loaded_model = False
        self.best_val_loss=float('inf')


    def train(self, source_dataloader_weak, source_dataloader_strong, source_dataloader_test, 
              target_dataloader_weak, target_dataloader_strong, target_dataloader_test,
              param, save_path, target_labeled_dataloader=None, target_labeled_strong_dataloader=None):
        """
        Trains the model (encoder + classifier).

        Arguments:
        ----------
        source_dataloader_weak: PyTorch DataLoader
            DataLoader with source domain training data with weak augmentations.

        source_dataloader_strong: PyTorch DataLoader
            DataLoader with source domain training data with strong augmentations.

        target_dataloader_weak: PyTorch DataLoader
            DataLoader with target domain training data with weak augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE.

        target_dataloader_strong: PyTorch DataLoader
            DataLoader with target domain training data with strong augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE. 

        target_dataloader_test: PyTorch DataLoader
            DataLoader with target domain validation data, used for early stopping.

        param: dict
            Dictionary containing hyperparameters

        save_path: str
            Path to store model weights.

        Returns:
        --------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes. Multi-head
        """
        if torch.__version__.startswith("2."):
            import torch._dynamo as dynamo
            # torch._dynamo.config.verbose = True
            
            torch._dynamo.config.suppress_errors = True

            # torch._dynamo.reset()
            self.encoder = torch.compile(self.encoder)
            self.classifier = torch.compile(self.classifier)
        self.param = param
        # configure hyperparameters
        lr = param.initialLearningRate
        use_scheduler = param.use_scheduler
        tau = param.tau
        epochs = param.maxEpoch

        iters = min(len(source_dataloader_weak), len(source_dataloader_strong), len(target_dataloader_weak), len(target_dataloader_strong))
        iters = len(source_dataloader_weak)

        eval_interval = iters // param.evalInterval 

        # mu related stuff
        steps_per_epoch = iters
        total_steps = epochs * steps_per_epoch 
        current_step = 0 if self.current_step ==0 else iters*self.start_epoch

        scheduler = None

        from rangerlars import RangerLars

        opt_name=param.opt_name

        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                list(self.encoder.parameters()) + list(self.classifier.parameters()),
                lr=lr,
                momentum=param.momentum,
                weight_decay=param.l2RegularizationLambda,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, momentum=param.momentum, weight_decay=param.l2RegularizationLambda, eps=0.0316, alpha=0.9
            )
        elif opt_name=="rangerlars":
            optimizer = RangerLars(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, weight_decay=param.l2RegularizationLambda )
        elif opt_name=="radam":
            optimizer = torch.optim.RAdam(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, weight_decay=param.l2RegularizationLambda )
        else:
             raise RuntimeError("Invalid optimier")

        if param.use_scheduler:
            if param.lr_scheduler == "steplr":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param.lr_step_size, gamma=param.lr_gamma)
            elif param.lr_scheduler == "cosineannealinglr":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs , eta_min=param.lr_min
                )
            elif param.lr_scheduler == "exponentiallr":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param.lr_gamma)

            elif param.lr_scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=param.lr_gamma, patience=param.lr_step_size)
            
            elif param.lr_scheduler == "LambdaLR":
                # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 1e-4/lr if epoch < 2 else 1)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: pow(0.5, epoch//15) )
            else:
                raise RuntimeError(
                    f"Invalid lr scheduler '{param.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                    "are supported."
                )
        
        if self.loaded_model:
            optimizer.load_state_dict(self.m_pth["optimizer"])
            if param.use_scheduler and "lr_scheduler" in self.m_pth:
                scheduler.load_state_dict(self.m_pth["lr_scheduler"])

        self.USE_SWA = param.USE_SWA
        if param.USE_SWA:
            alpha = 1.0 - param.swa_ema_decay
            # alpha = min(1.0, alpha * adjust)
            self.feat_ema = ExponentialMovingAverage(self.encoder, device=self.device, decay=1.0 - alpha)
            self.cls_ema = ExponentialMovingAverage(self.classifier, device=self.device, decay=1.0 - alpha)

        # early stopping variables
        start_epoch = self.start_epoch
        best_loss = float('inf')
        patience = param.patience * param.evalInterval 
        bad_epochs = 0
        if current_step:
            self.history_writer = open( save_path+".Train_log.txt","a")
            self.config_writer = open( save_path+".config","a")
        else:
            self.history_writer = open( save_path+".Train_log.txt","w")
            self.config_writer = open( save_path+".config","w")
        
        self.history_writer.write(param.s2t)
        self.config_writer.write(param.config_write)
        self.config_writer.close()
        
        self.history_writer.flush()
        self.history = {'epoch_loss': [], 'loss_source': [], 'loss_target': [], 'Target_Test': []}
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter()

        def get_rundvc_loss(ls, lc, lspp, source_total, num_target_labeled, labels_target):
            
            source_batch_size = source_total // 2
            target_batch_size = (lc.size(0)-source_total)//2 - num_target_labeled


            if num_target_labeled:
                logits_target_labeled = lc[source_total: source_total+2*num_target_labeled]


            lsp = lc[:source_total]
            if param.USE_RLI:
                # perform random logit interpolation
                lambd = torch.rand_like(lsp).to(self.device)
                final_logits_source = (lambd * lsp) + ((1-lambd) * lspp)
            else:
                final_logits_source = lsp

            ## softmax for logits of weakly augmented source images
            logits_source_weak = final_logits_source[:source_batch_size]

            if True or num_target_labeled == 0 :
                # compute loss
                source_loss = self._compute_source_loss(logits_source_weak, final_logits_source[source_batch_size:], ls)
                if not param.USE_SSL and num_target_labeled:
                    source_loss += self._compute_target_labeled_loss(labels_target, logits_target_labeled, num_target_labeled).mean()
            else:
                source_loss = self._compute_source_loss(logits_source_weak[:-num_target_labeled], final_logits_source[source_batch_size:-num_target_labeled], ls[:-num_target_labeled])
                source_loss += self._compute_source_loss(logits_source_weak[-num_target_labeled:], final_logits_source[-num_target_labeled:], ls[-num_target_labeled:])


                
            if not param.USE_SSL:
                # mind the placeholders
                if param.UNIFY_MASK:
                    return source_loss, 0, torch.ones(1)
                else:
                    return source_loss, 0

            
            ## softmax for logits of weakly augmented target images
            logits_target = lc[source_total+2*num_target_labeled:]
            logits_target_weak = logits_target[:target_batch_size]
            logits_target_strong = logits_target[target_batch_size:]
            
            pseudolabels_target = F.softmax(logits_target_weak, 1)

            final_pseudolabels = pseudolabels_target.detach()
            
            if param.RELATIVE_THRESHOLD:
                # perform relative confidence thresholding
                pseudolabels_source = F.softmax(logits_source_weak, 1)
                row_wise_max, _ = torch.max(pseudolabels_source.detach(), dim=1)
                final_sum = torch.mean(row_wise_max, 0)
                
                ## define relative confidence threshold
                c_tau = tau * final_sum
                max_values, _ = torch.max(final_pseudolabels, dim=1)
                mask = (max_values >= c_tau).float()
            else:
                # Fixmatch fixed threshold
                max_probs, targets_u = torch.max(final_pseudolabels, dim=1)
                mask = max_probs.ge(param.tau).float()

            self.confident_num = sum(mask) / len(mask) * 100.0

            
            final_pseudolabels = torch.max(final_pseudolabels, 1)[1] # argmax
            
            if param.UNIFY_MASK:
                target_loss = self._compute_target_loss_nomask(final_pseudolabels, logits_target_strong)
                APPEND_TO_SOURCE = True
                if num_target_labeled:
                    if APPEND_TO_SOURCE:
                        source_loss += self._compute_target_labeled_loss(labels_target, logits_target_labeled, num_target_labeled).mean() / source_batch_size * num_target_labeled
                    else:
                        target_labeled_loss = self._compute_target_labeled_loss(labels_target, logits_target_labeled, num_target_labeled)
                        target_loss = torch.cat([target_loss, target_labeled_loss], 0)
                        mask = torch.cat([mask, torch.ones(num_target_labeled).float().to(self.device)], 0)
                return source_loss, target_loss, mask
            else:
                target_loss = self._compute_target_loss(final_pseudolabels, logits_target_strong, mask)
                if num_target_labeled:
                    target_labeled_loss = self._compute_target_labeled_loss(labels_target, logits_target_labeled, num_target_labeled, mask)
                return source_loss, target_loss

        self.confident_num = 0

        if target_labeled_dataloader:
            tl_dataloader = iter(target_labeled_dataloader)
            tl_s_dataloader = iter(target_labeled_strong_dataloader)
        # pbar = tqdm.tqdm(zip(source_dataloader_weak, source_dataloader_strong, target_dataloader_weak, target_dataloader_strong))
        sl_w_dataloader = iter(source_dataloader_weak)
        sl_s_dataloader = iter(source_dataloader_strong)
        if not self.param.FULL_LABEL:
            tul_w_dataloader = iter(target_dataloader_weak)
            tul_s_dataloader = iter(target_dataloader_strong)


        # training loop
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0

            # set network to training mode
            self.encoder.train()
            self.classifier.train()

            pbar = tqdm.tqdm(range(iters))
            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            # for (data_source_weak, labels_source), (data_source_strong, _), (data_target_weak, _), (data_target_strong, _) in pbar:
            for i in pbar:
                try:
                    data_source_weak, labels_source =  next(sl_w_dataloader)
                except StopIteration:
                    sl_w_dataloader = iter(source_dataloader_weak)
                    data_source_weak, labels_source = next(sl_w_dataloader)

                try:
                    data_source_strong, _ = next(sl_s_dataloader)
                except StopIteration:
                    sl_s_dataloader = iter(source_dataloader_strong)
                    data_source_strong, _ = next(sl_s_dataloader)

                if not self.param.FULL_LABEL:
                    try:
                        data_target_weak, _ = next(tul_w_dataloader)
                    except StopIteration:
                        tul_w_dataloader = iter(target_dataloader_weak)
                        data_target_weak, _ = next(tul_w_dataloader)

                    try:
                        data_target_strong, _ = next(tul_s_dataloader)
                    except StopIteration:
                        tul_s_dataloader = iter(target_dataloader_strong)
                        data_target_strong, _ = next(tul_s_dataloader)


                data_source_weak = data_source_weak.to(self.device)
                data_source_strong = data_source_strong.to(self.device)
                if not self.param.FULL_LABEL:
                    data_target_weak = data_target_weak.to(self.device)
                    data_target_strong = data_target_strong.to(self.device)
                
                for idx in range(len(labels_source)):
                    labels_source[idx] = labels_source[idx].to(self.device)

                num_target_labeled = 0
                if target_labeled_dataloader and  current_step % param.SSDA_ITER == 1:
                    try:
                        data_target_labeled, labels_target = next(tl_dataloader)
                    except StopIteration:
                        tl_dataloader = iter(target_labeled_dataloader)
                        data_target_labeled, labels_target = next(tl_dataloader)

                    try:
                        data_target_labeled_strong, _ = next(tl_s_dataloader)
                    except StopIteration:
                        tl_s_dataloader = iter(target_labeled_strong_dataloader)
                        data_target_labeled_strong, _ = next(tl_s_dataloader)

                    # merge target labeled to source data
                    for idx in range(len(labels_target)):
                        labels_target[idx] = labels_target[idx].to(self.device)
                        
                        # labels_source[idx] = torch.cat([labels_source[idx], labels_target[idx]],0)
                    data_target_labeled = data_target_labeled.to(self.device)
                    data_target_labeled_strong = data_target_labeled_strong.to(self.device)
                    
                    num_target_labeled = data_target_labeled.size(0)
                    
                    # data_source_weak = torch.cat( [data_source_weak, data_target_labeled] ,0)
                    # data_source_strong = torch.cat( [data_source_strong, data_target_labeled_strong] ,0)

                
                

                # concatenate data (in case of low GPU power this could be done after classifying with the model)
                if self.param.FULL_LABEL:
                    data_combined = torch.cat([data_source_weak, data_source_strong], 0)
                else:
                    if num_target_labeled == 0:
                        data_combined = torch.cat([data_source_weak, data_source_strong, data_target_weak, data_target_strong], 0)
                    else:
                        #  sw, ss, tuw, tus, tlw, tls
                        data_combined = torch.cat([data_source_weak, data_source_strong, data_target_labeled, data_target_labeled_strong, data_target_weak, data_target_strong], 0)
                source_combined = torch.cat([data_source_weak, data_source_strong], 0)

                # get source data limit (useful for slicing later)
                source_total = source_combined.size(0)

                # zero gradients
                optimizer.zero_grad()

                feature_combined = self.encoder(data_combined)
                # forward pass: calls the model once for both source and target and once for source only
                logits_combined = self.classifier(feature_combined)
                # logits_source_p = logits_combined[:source_total]

                target_batch_size = (data_combined.size(0)-source_total)//2 - num_target_labeled

                if param.USE_RLI:
                    # from https://github.com/yizhe-ang/AdaMatch-PyTorch/blob/main/trainers/adamatch.py
                    self._disable_batchnorm_tracking(self.encoder)
                    self._disable_batchnorm_tracking(self.classifier)
                    logits_source_pp = self.classifier(self.encoder(source_combined))
                    
                    self._enable_batchnorm_tracking(self.encoder)
                    self._enable_batchnorm_tracking(self.classifier)
                else:
                    logits_source_pp=[0,0,0,0]

                ## compute target loss weight (mu)
                pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
                step = torch.tensor(current_step, dtype=torch.float).to(self.device)

                if param.FIXED_mu:
                    mu = torch.tensor(param.mu ,dtype=torch.float).to(self.device)
                else:
                    schedule_epoch = 15
                    mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / (schedule_epoch*2*iters)  )) / 2
                    # set maximum mu
                    mu = torch.tensor(param.mu ,dtype=torch.float).to(self.device)* mu 
                
                if param.UNIFY_MASK:
                    # aggregates all mask info from four tasks. Use a unified masking for filtering confident pseudo-labels
                    source_loss, target_loss, final_mask = get_rundvc_loss(labels_source[0],logits_combined[0],
                            logits_source_pp[0],  source_total,num_target_labeled, labels_target[0] if num_target_labeled else None )
                    t_source_loss = source_loss 
                    t_target_loss = target_loss

                    for t_i in range(1,4):
                        source_loss, target_loss, tmp_mask = get_rundvc_loss(labels_source[t_i],logits_combined[t_i], 
                            logits_source_pp[t_i], source_total, num_target_labeled, labels_target[t_i] if num_target_labeled else None)
                        t_source_loss += source_loss 
                        t_target_loss += target_loss
                        final_mask *= tmp_mask
                    
                    t_target_loss = (t_target_loss * final_mask).mean()
                    self.confident_num = sum(final_mask) / len(final_mask) * 100.0
                else: # not Unified masking, masking is applied seperately to each task
                    source_loss, target_loss = get_rundvc_loss(labels_source[3],logits_combined[3],
                            logits_source_pp[3], source_total,num_target_labeled, labels_target[3] if num_target_labeled else None)
                    t_source_loss = source_loss 
                    t_target_loss = target_loss

                    # reverse order, to print confident_num of genotypes prediction task.
                    for t_i in [2,1,0]:
                        source_loss, target_loss = get_rundvc_loss(labels_source[t_i],logits_combined[t_i],
                            logits_source_pp[t_i], source_total, num_target_labeled, labels_target[t_i] if num_target_labeled else None)
                        t_source_loss += source_loss 
                        t_target_loss += target_loss

                if param.USE_CORAL:
                    coral_loss = param.CORAL_weight * coral(feature_combined[:source_total-num_target_labeled], feature_combined[source_total:])
                    # print("Loss value:",source_loss, coral_loss)
                else:
                    coral_loss = 0
                
                current_step += 1

                if param.clip_grad_norm:
                    loss = t_source_loss + coral_loss
                    # backpropagate and update weights 
                    loss.backward(retain_graph=True)
                    # clip gradient
                    if param.clip_grad_norm:
                        grad_norm_s = nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.classifier.parameters()), param.clip_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    self.encoder.zero_grad()
                    self.classifier.zero_grad()
                    
                    t_target_loss = t_target_loss
                    t_target_loss.backward()
                    # clip gradient
                    if param.clip_grad_norm:
                        grad_norm_t = nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.classifier.parameters()), mu * param.clip_grad_norm)
                else:
                    loss = t_source_loss +  mu *t_target_loss + coral_loss
                    loss.backward()

                optimizer.step()

                # logging
                if current_step % 100==99:

                    data_ = {
                        'basic1_c1_weight': torch.mean(torch.abs(self.encoder.basic1.conv1.weight.grad)).item(),
                        'layer1_c1_weight': torch.mean(torch.abs(self.encoder.layer1[0].conv1.weight.grad)).item(),
                        'layer1_c2_weight': torch.mean(torch.abs(self.encoder.layer1[0].conv2.weight.grad)).item(),
                        'basic2_c1_weight': torch.mean(torch.abs(self.encoder.basic2.conv1.weight.grad)).item(),
                        'layer2_c1__weight': torch.mean(torch.abs(self.encoder.layer2[0].conv1.weight.grad)).item(),
                        'layer2_c2__weight': torch.mean(torch.abs(self.encoder.layer2[0].conv2.weight.grad)).item(),
                        'basic3_c1_weight': torch.mean(torch.abs(self.encoder.basic3.conv1.weight.grad)).item(),
                        'layer3_c1__weight': torch.mean(torch.abs(self.encoder.layer3[0].conv1.weight.grad)).item(),
                        'layer3_c2__weight': torch.mean(torch.abs(self.encoder.layer3[0].conv2.weight.grad)).item(),
                        'cls_l5_1_weight': torch.mean(torch.abs(self.classifier.L5_1[0].weight.grad)).item(),
                        'cls_l5_2_weight': torch.mean(torch.abs(self.classifier.L5_2[0].weight.grad)).item(),
                        'cls_l5_3_weight': torch.mean(torch.abs(self.classifier.L5_3[0].weight.grad)).item(),
                    }
                    self.writer.add_scalars("Gradient Mean Value by Layer", data_, current_step)

                    data_ = {
                        'basic1_c1_weight': torch.var(torch.abs(self.encoder.basic1.conv1.weight.grad)).item(),
                        'layer1_c1_weight': torch.var(torch.abs(self.encoder.layer1[0].conv1.weight.grad)).item(),
                        'layer1_c2_weight': torch.var(torch.abs(self.encoder.layer1[0].conv2.weight.grad)).item(),
                        'basic2_c1_weight': torch.var(torch.abs(self.encoder.basic2.conv1.weight.grad)).item(),
                        'layer2_c1__weight': torch.var(torch.abs(self.encoder.layer2[0].conv1.weight.grad)).item(),
                        'layer2_c2__weight': torch.var(torch.abs(self.encoder.layer2[0].conv2.weight.grad)).item(),
                        'basic3_c1_weight': torch.var(torch.abs(self.encoder.basic3.conv1.weight.grad)).item(),
                        'layer3_c1__weight': torch.var(torch.abs(self.encoder.layer3[0].conv1.weight.grad)).item(),
                        'layer3_c2__weight': torch.var(torch.abs(self.encoder.layer3[0].conv2.weight.grad)).item(),
                        'cls_l5_1_weight': torch.var(torch.abs(self.classifier.L5_1[0].weight.grad)).item(),
                        'cls_l5_2_weight': torch.var(torch.abs(self.classifier.L5_2[0].weight.grad)).item(),
                        'cls_l5_3_weight': torch.var(torch.abs(self.classifier.L5_3[0].weight.grad)).item(),
                    }
                    self.writer.add_scalars("Gradient Variance Value by Layer", data_, current_step)


                    try:
                        self.writer.add_scalars( 'Loss', { 'source':t_source_loss.detach().item() ,'target': t_target_loss.detach().item(), 'coral_loss': coral_loss.detach().item() if param.USE_CORAL else 0 }, current_step)
                    except:
                        self.writer.add_scalars( 'Loss', { 'source':t_source_loss.detach().item() ,'target': t_target_loss , 'coral_loss': coral_loss.detach().item() if param.USE_CORAL else 0}, current_step)
                    
                    if param.clip_grad_norm:
                        self.writer.add_scalars('GradNorm',{'Norm-S':grad_norm_s, 'Norm-t': grad_norm_t}, current_step)

                    self.writer.add_scalars('Parameters',{'lr': optimizer.param_groups[-1]['lr'] , "mu":mu, "Grad": param.clip_grad_norm if param.clip_grad_norm else 0, "wd": param.l2RegularizationLambda },current_step)


                # swa: weight averaging
                if epoch >= param.swa_start_epoch and param.USE_SWA and (current_step % param.swa_step_size) == (param.swa_step_size - 1):
                    self.feat_ema.update_parameters(self.encoder)
                    self.cls_ema.update_parameters(self.classifier)

                if (current_step % eval_interval) == ( eval_interval-1):
                    
                    self.USE_SWA_MODEL_IN_EVAL = False
                    if epoch >= param.swa_start_epoch and  self.USE_SWA:
                        self.USE_SWA_MODEL_IN_EVAL = True

                    if epoch >= param.swa_start_epoch and  self.USE_SWA:
                        # copy BN from training model as a start point for BN updtae... 
                        self._copy_batchnorm(self.encoder, self.feat_ema)
                        self._copy_batchnorm(self.classifier, self.cls_ema)
                        self.feat_ema.train()
                        self.cls_ema.train()

                        eval_pbar = tqdm.tqdm ( zip(source_dataloader_weak, source_dataloader_strong, target_dataloader_weak, target_dataloader_strong) )
                        print("\nUpdating BN layers in EMA model")
                        # the batch normalization layer should be updated in the same data the model was trained on.
                        iter_bn = 0
                        for (inp,_), (inp2,_), (inp3,_), (inp4,_) in eval_pbar :
                            iter_bn += 1
                            if iter_bn > 250:
                                # we use 0.99 for BN momentum. Just update it with 500 iter...
                                break
                            inp = inp.to(self.device)
                            inp2 = inp2.to(self.device)
                            inp3 = inp3.to(self.device)
                            inp4 = inp4.to(self.device)
                            comb_inp = torch.cat( [inp, inp2, inp3, inp4] ,0)
                            self.cls_ema(self.feat_ema(comb_inp))
                        del eval_pbar

                    epoch_loss_source = self.evaluate(source_dataloader_test)
                    test_epoch_loss = self.evaluate(target_dataloader_test)
                    print('[Validation-Epoch {}/Step:{}] Best loss: {:.6f}; source loss: {:.6f}; Target domain loss: {:.6f};'.format(epoch+1, current_step, self.best_val_loss, epoch_loss_source, test_epoch_loss))
                    self.history_writer.write('[Validation-Epoch {}/Step: {}] Best loss: {:.6f}; source loss: {:.6f}; Target domain loss: {:.6f};\n'.format(epoch+1, current_step, self.best_val_loss, epoch_loss_source, test_epoch_loss))
                    self.history_writer.flush()
                    if epoch_loss_source < self.best_val_loss:
                        self.best_val_loss = epoch_loss_source
                        if self.USE_SWA:
                            torch.save({'encoder_weights': self.encoder.state_dict(),
                                    'classifier_weights': self.classifier.state_dict(),
                                    'ema_encoder_weights': self.feat_ema.state_dict(),
                                    'ema_classifier_weights': self.cls_ema.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "lr_scheduler": scheduler.state_dict() if scheduler else None,
                                    'epoch': epoch
                                    }, save_path+".best")
                        else:
                            torch.save({'encoder_weights': self.encoder.state_dict(),
                                    'classifier_weights': self.classifier.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "lr_scheduler": scheduler.state_dict() if scheduler else None,
                                    'epoch': epoch
                                    }, save_path+".best")
                        patience = param.patience * param.evalInterval 
                    else:
                        patience -= 1
                        if patience == 0:
                            break
                    self.writer.add_scalars( 'Validation', { 'source val loss': epoch_loss_source, 'target loss': test_epoch_loss}, current_step)

                    # if param.lr_scheduler == "plateau":
                    #     scheduler.step(epoch_loss_source)
                    
                # metrics
                running_loss += loss.detach().item()
                try:
                    pbar.set_postfix( {"Total Iters": iters, "epoch": epoch, "source loss": f"{t_source_loss.detach().item():.3f}", "target loss": f"{t_target_loss.detach().item():.3f}", "Mask Ratio":self.confident_num.detach().item()  ,"mu":mu.detach().item() } )
                    self.writer.add_scalars( 'Mask', { 'Ratio': self.confident_num.detach().item()}, current_step)
                except:
                    pbar.set_postfix( {"Total Iters": iters, "epoch": epoch, "source loss": f"{t_source_loss.detach().item():.3f}","mu":mu.detach().item()} )

            # get losses
            epoch_loss = running_loss / iters
            self.history['epoch_loss'].append(epoch_loss)
            if self.USE_SWA:
                torch.save({'encoder_weights': self.encoder.state_dict(),
                        'classifier_weights': self.classifier.state_dict(),
                        'ema_encoder_weights': self.feat_ema.state_dict(),
                        'ema_classifier_weights': self.cls_ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict() if scheduler else None,
                        'epoch': epoch
                        }, save_path+".epoch_{}".format(epoch))
            else:
                torch.save({'encoder_weights': self.encoder.state_dict() if not self.USE_SWA else self.feat_ema.state_dict(),
                            'classifier_weights': self.classifier.state_dict() if not self.USE_SWA else  self.cls_ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": scheduler.state_dict() if scheduler else None,
                            'epoch': epoch
                            }, save_path+".epoch_{}".format(epoch))

            # scheduler step
            if use_scheduler and scheduler is not None and param.lr_scheduler != "plateau":
                if param.lr_scheduler == "plateau":
                    scheduler.step(epoch_loss_source)
                else:
                    scheduler.step()

        best = torch.load(save_path)
        self.encoder.load_state_dict(best['encoder_weights'])
        self.classifier.load_state_dict(best['classifier_weights'])
        
        return self.encoder, self.classifier

    def evaluate(self, dataloader, return_lists_roc=False):
        """
        Evaluates model on `dataloader`.

        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.

        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.

        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.encoder.eval()
        self.classifier.eval()

        if self.USE_SWA:
            self.feat_ema.eval()
            self.cls_ema.eval()

        loss_f = nn.CrossEntropyLoss()
        total_loss = 0
        pbar = tqdm.tqdm ( dataloader)

        with torch.no_grad():
            for idx, (data, labels) in enumerate(pbar):
                # data = data.squeeze()
                data = data.to(self.device)
                
                for j in range(4):
                    labels[j] = torch.argmax(labels[j],dim=1).long()
                    labels[j] = labels[j].to(self.device)
                # predict
                if self.USE_SWA and self.USE_SWA_MODEL_IN_EVAL:
                    outputs = self.cls_ema(self.feat_ema(data))
                else:
                    outputs = self.classifier(self.encoder(data))
                
                
                loss = loss_f( outputs[0], labels[0] ) + loss_f( outputs[1], labels[1] ) \
                    + loss_f( outputs[2], labels[2] ) + loss_f( outputs[3], labels[3] )
                # append
                total_loss += loss.item() 
        total_loss /= (idx+1)

        # set network to evaluation mode
        self.encoder.train()
        self.classifier.train()
        
        if self.USE_SWA:
            self.feat_ema.train()
            self.cls_ema.train()

        return total_loss


    def load_model(self, chkpnt_fn):
        self.m_pth = torch.load(chkpnt_fn)
        self.encoder.load_state_dict(self.m_pth['encoder_weights'])
        self.classifier.load_state_dict(self.m_pth['classifier_weights'])
        self.loaded_model = True
        if 'epoch' in self.m_pth:
            self.start_epoch = self.m_pth['epoch'] + 1
            self.current_step = self.m_pth['epoch']
        return

    def prefix_load(self,source,target,prefix):

        pretrained_dict = {k: v for k, v in target.items() }

        for k, v in source.items():
            if k[len(prefix+"."):] in target:
                pretrained_dict[k[len(prefix+"."):]] = v
            elif k[len("_orig_mod"+"."):] in target:
                pretrained_dict[k[len("_orig_mod"+"."):]] = v
            elif k[len("module."):] in target:
                pretrained_dict[k[len("module."):]] = v
            elif k in target:
                # if "layer2.0.bn2.running_mean" in k:
                #     print("Loaded",v[:5])
                pretrained_dict[k] = v

            else:
                print("[EMA model loading] param only exists in chkpnt file:",k)
        return pretrained_dict

    def load_model_SWA(self, chkpnt_fn):
        self.m_pth = torch.load(chkpnt_fn)
        # self.encoder.load_state_dict(self.m_pth['encoder_weights'])
        # self.classifier.load_state_dict(self.m_pth['classifier_weights'])
        target_enc = self.encoder.state_dict()
        target_cls = self.classifier.state_dict()

        
        self.encoder.load_state_dict( self.prefix_load(self.m_pth['encoder_weights'], target_enc,"module" ) )
        self.classifier.load_state_dict( self.prefix_load(self.m_pth['classifier_weights'], target_cls ,"module") )
        
        self.loaded_model = True
        if 'epoch' in self.m_pth:
            self.start_epoch = self.m_pth['epoch'] + 1
            self.current_step = self.m_pth['epoch']
        return

    @staticmethod
    def _copy_batchnorm(source_model, dest_model):
        # iterate over source and destination model modules
        src_modules = source_model.named_modules()
        dst_modules = dest_model.named_modules()
        # ema model in dest_model have "module" at first, simple workaround..
        next(dst_modules)

        bn_copied = 0
        for (src_name, src_module), (dst_name, dst_module) in zip(src_modules, dst_modules):
            # if the source module is a batch norm layer, copy its state dict to the destination module
            if isinstance(src_module, nn.modules.batchnorm._BatchNorm) and isinstance(dst_module, nn.modules.batchnorm._BatchNorm):
                bn_copied += 1 
                dst_module.load_state_dict(src_module.state_dict())

        if bn_copied == 0:
            print("No BN copied!")
        else:
            print("{} BN copied!".format(bn_copied))

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        # loss_function = nn.CrossEntropyLoss() # default: `reduction="mean"`
        if USE_FOCAL:
            loss_function = FocalLoss()
        else:
            loss_function = nn.CrossEntropyLoss() # default: `reduction="mean"`
            labels = torch.argmax(labels,dim=1).long()
        weak_loss = loss_function(logits_weak, labels )
        strong_loss = loss_function(logits_strong, labels)

        #return weak_loss + strong_loss
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        # loss_function = nn.CrossEntropyLoss(reduction="none")
        if USE_FOCAL:
            loss_function = FocalLoss(reduction="none")
        else:
            loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach() # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)
        
        return (loss * mask).mean()

    @staticmethod
    def _compute_target_loss_nomask(pseudolabels, logits_strong):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        # loss_function = nn.CrossEntropyLoss(reduction="none")
        if USE_FOCAL:
            loss_function = FocalLoss(reduction="none")
        else:
            loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach() # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)
        
        return loss 
    @staticmethod
    def _compute_target_labeled_loss(labels, logits_strong, num_target_labeled, mask=None):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `labels` are treated as ground truth (standard SSL practice).
        """
        # loss_function = nn.CrossEntropyLoss(reduction="none")
        if USE_FOCAL:
            loss_function = FocalLoss(reduction="none")
        else:
            loss_function = nn.CrossEntropyLoss(reduction="none")
            labels = torch.argmax(labels,dim=1).long()
        loss = ( loss_function(logits_strong[:num_target_labeled], labels) + loss_function(logits_strong[num_target_labeled:], labels) )/2
        
        if mask:
            return (loss*mask).mean()
        else:
            return loss 