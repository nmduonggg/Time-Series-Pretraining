import os
import sys
from tqdm import tqdm
sys.path.append("..")

from loss import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from model import * 

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if training_mode == 'pre_train':
        print('Pretraining on source dataset')
        for epoch in range(1, config.num_epoch + 1):
            # Train and validate
            """Train. In fine-tuning, this part is also trained???"""
            train_loss = model_pretrain(model, model_optimizer, criterion, train_dl, config, device, training_mode)
            logger.debug('\nPre-training Epoch : '+str(epoch) + ' Train Loss : '+str(train_loss.item()))

        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", 'ckp_last.pt'))
        print('Pretrained model is stored at folder:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))

    """Fine-tuning and Test"""
    if training_mode != 'pre_train':
        """fine-tune"""
        print('Fine-tune on Fine-tuning set')
        performance_list = []
        total_f1 = []
        total_acc = []
        KNN_f1 = []
        global emb_finetune, label_finetune, emb_test, label_test

        for epoch in range(1, config.num_epoch + 1):
            logger.debug(f'\nEpoch : {epoch}')

            params, valid_loss, emb_finetune, label_finetune, acc = model_finetune(model, model_optimizer, valid_dl, config,
                                  device, training_mode, classifier=classifier, classifier_optimizer=classifier_optimizer)
            model = params['model']
            classifier = params['classifier']
            model_optimizer = params['model_optimizer']
            classifier_optimizer = params['classifier_optimizer']
            
            scheduler.step(valid_loss)

            # save best fine-tuning model""
            global arch
            arch = 'ecg2ecg'
            if len(total_acc) == 0 or acc > max(total_acc):
                print('Update fine-tuned model')
                os.makedirs('experiments_logs/finetunemodel/', exist_ok=True)
                torch.save(model.state_dict(), 'experiments_logs/finetunemodel/' + arch + '_model.pt')
                torch.save(classifier.state_dict(), 'experiments_logs/finetunemodel/' + arch + '_classifier.pt')
            total_acc.append(acc)
                
            logger.debug("MLP Training: Loss=%.4f | ACC=%.4f"% (valid_loss, acc*100))

            # evaluate on the test set
            """Testing set"""
            logger.debug('Test on Target datasts test set')
            # model.load_state_dict(torch.load('experiments_logs/finetunemodel/' + arch + '_model.pt'))
            # classifier.load_state_dict(torch.load('experiments_logs/finetunemodel/' + arch + '_classifier.pt'))
            test_loss, test_acc, emb_test, label_test, performance = model_test(model, test_dl, config, device, training_mode, classifier=classifier, classifier_optimizer=classifier_optimizer)
            performance_list.append(performance)
        
            logger.debug('MLP Testing: Loss=%.4f | ACC=%.4f'% (test_loss, test_acc*100))

            """Use KNN as another classifier; it's an alternation of the MLP classifier in function model_test. 
            Experiments show KNN and MLP may work differently in different settings, so here we provide both. """
            # train classifier: KNN
            # neigh = KNeighborsClassifier(n_neighbors=5)
            # neigh.fit(emb_finetune, label_finetune)
            # knn_acc_train = neigh.score(emb_finetune, label_finetune)
            # # print('KNN finetune acc:', knn_acc_train)
            # representation_test = emb_test.detach().cpu().numpy()

            # knn_result = neigh.predict(representation_test)
            # knn_result_score = neigh.predict_proba(representation_test)
            # one_hot_label_test = one_hot_encoding(label_test)
            # # print(classification_report(label_test, knn_result, digits=4))
            # # print(confusion_matrix(label_test, knn_result))
            # knn_acc = accuracy_score(label_test, knn_result)
            # precision = precision_score(label_test, knn_result, average='macro', )
            # recall = recall_score(label_test, knn_result, average='macro', )
            # F1 = f1_score(label_test, knn_result, average='macro')
            # auc = roc_auc_score(one_hot_label_test, knn_result_score, average="macro", multi_class="ovr")
            # prc = average_precision_score(one_hot_label_test, knn_result_score, average="macro")
            # result_log = ('KNN Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'%
            #       (knn_acc, precision, recall, F1, auc, prc))
            # # print(result_log)
            # logger.debug(result_log)
            # KNN_f1.append(F1)
        logger.debug("\n################## Best testing performance! #########################")
        performance_array = np.array(performance_list)
        best_performance = performance_array[np.argmax(performance_array[:,0], axis=0)]
        final_result_log = ('Best Testing MLP Performance: Acc=%.4f' % (best_performance[0]))
        logger.debug(final_result_log)
        # print(final_result_log)
        # knn_best_log = ('Best KNN F1', max(KNN_f1))
        # print(knn_best_log)
        # logger.debug(knn_best_log)

    logger.debug("\n################## Training is Done! #########################")

def model_pretrain(model, model_optimizer, criterion, train_loader, config, device, training_mode,):
    total_loss = []
    model.train()
    global loss, loss_t, loss_f, l_TF, loss_c, data_test, data_f_test

    # optimizer
    model_optimizer.zero_grad()

    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]
        aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        """Compute Pre-train loss"""
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                       config.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True

        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.5
        loss = lam*(loss_t + loss_f) + (1-lam)*l_TF

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    print('Pretraining: overall loss:{}, l_t: {}, l_f:{}, l_c:{}, l_TF:{}'.format(loss, loss_t, loss_f, loss_c, l_TF))

    ave_loss = torch.tensor(total_loss).mean()

    return ave_loss


def model_finetune(model, model_optimizer, val_dl, config, device, training_mode, classifier=None, classifier_optimizer=None):
    global labels, pred_numpy, fea_concat_flat
    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []  # it should be outside of the loop
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    feas = np.array([])
    
    model.train()
    if classifier is not None: 
        print("Use classifier")
        classifier.train()

    for data, labels, aug1, data_f, aug1_f in tqdm(val_dl, total=len(val_dl)):
        # print('Fine-tuning: {} of target samples'.format(labels.shape[0]))
        # print(labels.detach())
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)
        
        """if random initialization:"""
        model_optimizer.zero_grad()  # The gradients are zero, but the parameters are still randomly initialized.
        classifier_optimizer.zero_grad()  # the classifier is newly added and randomly initialized

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
        nt_xent_criterion = NTXentLoss_poly(device, config.target_batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), \
                        nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3) #


        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss_p = criterion(predictions, labels)
        
        lam = 0.0
        loss = loss_p + lam*(loss_t + loss_f + l_TF)

        acc_bs = torch.eq(labels, torch.softmax(predictions, dim=1).detach().argmax(dim=1)).float()
        # print(acc_bs.shape)
        onehot_label = F.one_hot(labels, num_classes=config.num_classes_target)
        pred_numpy = predictions.detach().cpu().numpy()

        total_acc.append(acc_bs.cpu())
        total_loss.append(loss.detach().item())
        
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        if training_mode != "pre_train":
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            feas = np.append(feas, fea_concat_flat.data.cpu().numpy())

    feas = feas.reshape([len(trgs), -1])  # produce the learned embeddings

    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    
    ave_loss = torch.tensor(total_loss).mean()
    ave_acc = (torch.cat(total_acc, dim=0)).mean()
    # ave_auc = torch.tensor(total_auc).mean()
    # ave_prc = torch.tensor(total_prc).mean()

    new_params = {
        'model': model,
        'classifier': classifier,
        'model_optimizer': model_optimizer,
        'classifier_optimizer': classifier_optimizer
    }
    print('Finetune: loss = %.4f| Acc=%.4f'% (ave_loss, ave_acc*100))

    return new_params, ave_loss, feas, trgs, ave_acc


def model_test(model,  test_dl, config,  device, training_mode, classifier=None, classifier_optimizer=None):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss() # the loss for downstream classifier
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels, _,data_f, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
            h_t, z_t, h_f, z_f = model(data, data_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions_test = classifier(fea_concat)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)
            
            loss = criterion(predictions_test, labels)
            acc_bs = torch.eq(labels, torch.softmax(predictions_test, dim=1).detach().argmax(dim=1)).float().mean()
            # onehot_label = F.one_hot(labels, num_classes=config.num_classes_target)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            
            # acc_bs = np.array([pred_numpy[i]==labels_numpy[i] for i in range(labels_numpy.shape[0])]).mean()
            total_acc.append(acc_bs)

            total_loss.append(loss.item())

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    performance = [total_acc * 100]
    emb_test_all = torch.concat(tuple(emb_test_all))
    return total_loss, total_acc, emb_test_all, trgs, performance
