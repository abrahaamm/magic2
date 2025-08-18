import os
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader


class BERTTester:
    def __init__(self, detector_path: str,
                 test_dataloader: DataLoader = None,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param detector_path: BERT model path
        :param test_dataloader: test dataset dataloader [can be None]
        :param with_cuda: testing with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT testing, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Initialize the BERT Language Model, with BERT model
        self.detector = torch.load(detector_path, weights_only=False).to(self.device)

        # Distributed GPU testing if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.detector, device_ids=cuda_devices)

        # Setting the test dataloader
        self.test_data = test_dataloader

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.result = None

    def test(self, test_result_file):
        """
        loop over the test_data for testing, backward operation is activated, and auto save the model every epoch

        :return: None
        """

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP_test",
                              total=len(self.test_data), disable=True)

        prediction = []
        label = []
        detection_outputs = [] 
        linear1_outputs = [] 
        relu_outputs = [] 
        dropout_outputs = [] 
        linear2_outputs = [] 
        detection_threshold = 0
        print(f"Detection Threshold is : {detection_threshold}")
        for i, data in data_iter:
            # batch_data will be sent into the device (GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # forward the attack detection
            with torch.no_grad(): 
                detection_output, layer_output = self.detector.forward(data["eval_input"], data["segment_label"], return_hidden=True)
            linear1_outputs.extend(layer_output['linear1']) 
            relu_outputs.extend(layer_output['relu']) 
            dropout_outputs.extend(layer_output['dropout']) 
            linear2_outputs.extend(layer_output['linear2']) 
            # Evaluation of prediction
            prediction.extend((detection_output > detection_threshold).int().cpu())
            detection_outputs.extend(detection_output.cpu().detach())
            label.extend(data['eval_label'].cpu())

        # 假设 detection_output 是一个 numpy 数组或可以转换为 numpy 数组
        # 如果不是numpy数组，先转换
        if not isinstance(detection_outputs, np.ndarray):
            detection_outputs = np.array(detection_outputs)
        linear1_outputs = np.array(linear1_outputs) 
        relu_outputs = np.array(relu_outputs) 
        dropout_outputs = np.array(dropout_outputs) 
        linear2_outputs = np.array(linear2_outputs) 
        labels = np.array(label)
        # 保存为 .npy 文件
        np.save("detection_outputs.npy", detection_outputs)
        np.save("linear1_outputs.npy", linear1_outputs)
        np.save("relu_outputs.npy", relu_outputs) 
        np.save("dropout_outputs.npy", dropout_outputs) 
        np.save("linear2_outputs.npy", linear2_outputs)

        print("shape ", detection_outputs.shape, linear1_outputs.shape, relu_outputs.shape, dropout_outputs.shape, linear2_outputs.shape)
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels) 
        colors = ['blue', 'red'] 
        for lbl in unique_labels: 
            mask = (label == lbl) 
            plt.scatter(x = np.arange(len(detection_outputs))[mask], 
                        y = detection_outputs[mask], 
                        c = colors[int(lbl)], 
                        label = f'Label {lbl}', 
                        alpha = 0.5)
        # Add horizontal line at y=0
        plt.axhline(y=detection_threshold, color='black', linestyle='--', linewidth=1, label=f'y={detection_threshold}')
        plt.title('Detection Outputs by Label') 
        plt.xlabel('Sample Index') 
        plt.ylabel('Detection Output Value') 
        plt.legend() 
        plt.grid(True, linestyle='--', alpha=0.5) 
        plt.savefig('detection_outputs_scatter.png', dpi=300, bbox_inches='tight') 
        plt.show() 
        plt.close() 

        # print and save test results 
        self.result = "accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f" % (
            accuracy_score(label, prediction), precision_score(label, prediction),
            recall_score(label, prediction), f1_score(label, prediction))  
        with open(test_result_file, 'w') as f: 
            for predict_result in prediction: 
                f.write(f"{predict_result}\n")
        print(self.result)

    def save(self, file_path):
        """
        Saving the result

        :param file_path: result output path which going to be file_path
        :return: None
        """
        output_path = os.path.join(file_path, "test_result.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.result)


class BERTMultiTaskClassifierTester:
    def __init__(self, classifier_path: str,
                 test_dataloader: DataLoader = None,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.classifier = torch.load(classifier_path, weights_only=False).to(self.device)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.classifier, device_ids=cuda_devices)
        else:
            self.model = self.classifier

        self.test_data = test_dataloader
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.result = None

    def test(self, test_result_file):

        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP_test",
                              total=len(self.test_data), disable=True)

        multi_preds = []
        multi_labels = []

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                multi_output = self.model(
                    data["eval_input"],
                    data["segment_label"]
                )

            # 获取多分类预测
            multi_pred = torch.argmax(multi_output, dim=1).cpu()
            multi_preds.extend(multi_pred)
            multi_labels.extend(data['eval_label'].cpu())

        multi_preds = np.array(multi_preds)
        multi_labels = np.array(multi_labels)
        print(multi_preds)
        multi_result = "[Multi] accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f" % (
            accuracy_score(multi_labels, multi_preds),
            precision_score(multi_labels, multi_preds, average='macro'),
            recall_score(multi_labels, multi_preds, average='macro'),
            f1_score(multi_labels, multi_preds, average='macro')
        )

        self.result = multi_result

        # 每类详细评估
        target_names = ["UNIT",'FILE', 'NETFLOW','PROCESS', "DIR"]
        multi_report = classification_report(multi_labels, multi_preds, zero_division=0,target_names=target_names, labels=[0,1,2,3,4])
        self.result = multi_result + "\n\n[Per-class Multi Classification Report]\n" + multi_report

        # 保存预测
        with open(test_result_file, 'w') as f:
            for  m_pred in multi_preds:
                f.write(f"{m_pred.item()}\n")

        print(self.result)

    def save(self, file_path):
        output_path = os.path.join(file_path, "test_result.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.result)
