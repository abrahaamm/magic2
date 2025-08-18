import os

import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optim_schedule import ScheduledOptim
from ..model import BERTDetector
from ..model import BERTClassifier

class BERTTuner:
    def __init__(self, bert_path: str,
                 tune_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert_path: BERT model path
        :param tune_dataloader: tune dataset dataloader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: tuning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT tuning, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = torch.load(bert_path, weights_only=False)
        # Initialize the BERT Language Model, with BERT model
        self.detector = BERTDetector(self.bert).to(self.device)

        # Distributed GPU tuning if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.detector, device_ids=cuda_devices)

        # Setting the tune dataloader
        self.tune_data = tune_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Binary Cross Entropy Loss
        pos_weight = torch.tensor([40])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def tune(self, epoch):
        """
        loop over the tune_data for tuning, backward operation is activated, and auto save the model every epoch

        :param epoch: current epoch index
        :return: None
        """

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.tune_data),
                              desc="EP_tune:%d" % epoch,
                              total=len(self.tune_data), disable=True)

        avg_loss = 0.0
        prediction = []
        label = []

        for i, data in data_iter:
            # 0. batch_data will be sent into the device (GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the attack detection
            detection_output = self.model.forward(data["eval_input"], data["segment_label"])

            # 2. Cross Entropy Loss of detection result
            loss = self.criterion(detection_output, data["eval_label"].float())

            # 3. backward and optimization only
            self.optim_schedule.zero_grad()
            loss.backward()
            self.optim_schedule.step_and_update_lr()

            # Evaluation of prediction
            avg_loss += loss.item()
            prediction.extend((detection_output > 0).int().cpu()) ## 调参数 
            label.extend(data['eval_label'].cpu())
    
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss": round(loss.item(), 6)
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP_tune: %d, avg_loss: %.6f, accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f" % (
            epoch, avg_loss / len(data_iter), accuracy_score(label, prediction), precision_score(label, prediction),
            recall_score(label, prediction), f1_score(label, prediction)))

    def save(self, epoch, file_path="output/bert_tuned.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which going to be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = os.path.join(file_path, "tune.ep%d" % epoch)
        torch.save(self.detector.cpu(), output_path)
        output_bert_path = os.path.join(file_path, "tune_bert.ep%d" % epoch)
        torch.save(self.bert.cpu(),  output_bert_path)
        self.bert.to(self.device)
        self.detector.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        print("EP:%d Model Bert Saved on : " %epoch, output_bert_path)
        return output_path


class BERTClassifierTuner:
    def __init__(self, bert_path: str,
                 tune_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        微调BERT用于多任务分类：Token[1] -> 二分类；Token[3] -> 五分类。
        """

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # 加载预训练的BERT模型
        self.bert = torch.load(bert_path, weights_only=False)
        self.classifier = BERTClassifier(self.bert).to(self.device)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for BERTClassifier" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.classifier, device_ids=cuda_devices)
        else:
            self.model = self.classifier

        self.tune_data = tune_dataloader
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # 两个任务的损失函数
        self.multi_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 160.0, 10240.0, 12800.0]).to(self.device))

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def tune(self, epoch):
        data_iter = tqdm.tqdm(enumerate(self.tune_data),
                              desc="EP_tune:%d" % epoch,
                              total=len(self.tune_data), disable=False)

        avg_loss = 0.0
        multi_preds, multi_labels = [], []

        for i, batch in data_iter:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            input_ids, segment_label = batch["eval_input"], batch["segment_label"]
            multi_label = batch["eval_label"].long()

            # forward
            multi_logits = self.model(input_ids, segment_label)

            # loss
            multi_loss = self.multi_criterion(multi_logits, multi_label)
            loss = multi_loss

            # optimize
            self.optim_schedule.zero_grad()
            loss.backward()
            self.optim_schedule.step_and_update_lr()

            # collect for metric
            avg_loss += loss.item()

            # Multi-class prediction
            multi_preds.extend(torch.argmax(multi_logits, dim=-1).cpu().tolist())
            multi_labels.extend(multi_label.cpu().tolist())

            if i % self.log_freq == 0:
                data_iter.write(f"[Epoch {epoch} Step {i}] Loss: {loss.item():.6f}")

        print(f"EP {epoch} | Loss: {avg_loss / len(data_iter):.6f}")
        print("Multi-class Task - Accuracy: %.4f, F1: %.4f" %
              (accuracy_score(multi_labels, multi_preds),
               f1_score(multi_labels, multi_preds, average='macro')))

    def save(self, epoch, file_path="output/bert_classifier_tuned.model"):
        """
        保存当前的分类模型
        """
        os.makedirs(file_path, exist_ok=True)
        output_path = os.path.join(file_path, f"tune.ep{epoch}")
        output_bert_path = os.path.join(file_path, f"tune_bert.ep{epoch}")

        torch.save(self.classifier.cpu(), output_path)
        torch.save(self.bert.cpu(), output_bert_path)

        self.bert.to(self.device)
        self.classifier.to(self.device)

        print("Model saved to:", output_path)
        print("BERT backbone saved to:", output_bert_path)
        return output_path