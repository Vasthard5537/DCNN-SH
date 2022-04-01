import math
import glob
import os
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
import torch.utils.data as loader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import torch.nn.functional as F

"""
    文件的命名规范如下：
        对于Sequence文件夹：Train_seq.csv或者Test_seq.csv
        对于Shape文件夹：Train_shape-name.csv或者Test_shape-name.csv

    一级文件夹的命名与ENCODE中的保持一致，例如：
        wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk

    各个一级文件夹中，有两个二级文件夹，即Sequence文件夹和Shape文件夹

"""


# path = "/content/drive/MyDrive"

class SampleReader:
    """
        SampleReader一次可读取一个文件夹下的一些文件，具体策略如下：
            get_seq()函数可以读取Sequence文件夹中有关的文件
            get_shape()函数可以读取Shape文件夹中有关的文件
        注：对于Train和Test，不能同时读取
    """

    def __init__(self, file_name):
        """
            file_path:
                wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk
        """
        self.seq_path = file_name + '/sequence/'
        self.shape_path = file_name + '/shape/'
        self.histone_path = file_name + '/histone/'

    def get_seq(self, Test=False):

        if Test is False:
            # row_seq = pd.read_csv(self.seq_path + 'Train_seq.csv', sep=' ', header=None)
            row_seq = pd.read_csv(self.seq_path + 'Train_seq.csv', sep=',', header=None)
        else:
            # row_seq = pd.read_csv(self.seq_path + 'Test_seq.csv', sep=' ', header=None)
            row_seq = pd.read_csv(self.seq_path + 'Test_seq.csv', sep=',', header=None)
        middle = math.ceil(len(row_seq.loc[0, 1]) / 2)
        seq_num = row_seq.shape[0]
        seq_len = len(row_seq.loc[0, 1])

        # completed_seqs = np.empty(shape=(seq_num, seq_len, 4))
        completed_seqs = np.empty(shape=(seq_num, 101, 4))
        completed_labels = np.empty(shape=(seq_num, 1))
        for i in range(seq_num):
            # completed_seqs[i] = one_hot(row_seq.loc[i, 1][middle-50:middle+50+1])
            completed_seqs[i] = one_hot(row_seq.loc[i, 1][0:101])
            completed_labels[i] = row_seq.loc[i, 2]
        completed_seqs = np.transpose(completed_seqs, [0, 2, 1])

        return completed_seqs, completed_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []

        if Test is False:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Train' + '_' + shape + '.csv', header=None))
        else:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Test' + '_' + shape + '.csv', header=None))

        """
            seq_num = shape_series[0].shape[0]
            seq_len = shape_series[0].shape[1]
        """
        shape_len = shape_series[0].shape[1]
        middle = math.ceil(shape_len / 2)
        # completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), shape_series[0].shape[1]))
        completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), 101))

        for i in range(len(shapes)):
            shape_samples = shape_series[i]
            for m in range(shape_samples.shape[1]):
                # 注意，这里把索引那一行读进去了
                completed_shape[m][i] = shape_samples.iloc[m, middle - 50:middle + 50 + 1]
        completed_shape = np.nan_to_num(completed_shape)

        return completed_shape

    # def get_histone(self, Test=False):

    #     if Test is False:
    #         histone = pd.read_csv(self.histone_path + 'Train_his' + '.csv', header=None, index_col=None,
    #                               skiprows=lambda x: x % 9 == 0)
    #     else:
    #         histone = pd.read_csv(self.histone_path + 'Test_his' + '.csv', header=None, index_col=None,
    #                               skiprows=lambda x: x % 9 == 0)

    #     histone = histone.iloc[:, 1:]
    #     num = histone.shape[0] // 8
    #     histone = np.array(np.split(histone.values, num))

    #     return histone
    def get_histone(self, Test=False):

        if Test is False:
            histone = pd.read_csv(self.histone_path + 'Train_his' + '.csv', header=None, index_col=None,
                                  skiprows=lambda x: x % 9 == 0)
            histone = histone.iloc[:, 1:]
            num = histone.shape[0] // 8
            histone = np.array(np.split(histone.values, num))
            # histone = np.zeros((37548,8,20))
            #histone = np.minimum(histone, 0)
            print("train")
            print(histone.shape)
        else:
            histone = pd.read_csv(self.histone_path + 'Test_his' + '.csv', header=None, index_col=None,
                                  skiprows=lambda x: x % 9 == 0)
            histone = histone.iloc[:, 1:]
            num = histone.shape[0] // 8
            histone = np.array(np.split(histone.values, num))
            # histone = np.zeros((9386,8,20))
            #histone = np.minimum(histone, 0)
            print("test")
            print(histone.shape)

        return histone


class SSDataset_690(Dataset):

    def __init__(self, file_name, Test=False):
        shapes = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']

        sample_reader = SampleReader(file_name=file_name)

        self.completed_seqs, self.completed_labels = sample_reader.get_seq(Test=Test)
        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)
        self.completed_histone = sample_reader.get_histone(Test=Test)

    def __getitem__(self, item):
        return self.completed_seqs[item], self.completed_histone[item], self.completed_labels[item]
        # return self.completed_seqs[item], self.completed_shape[item], self.completed_labels[item]
        # return self.completed_seqs[item], self.completed_shape[item], self.completed_histone, self.completed_labels[item]

    def __len__(self):
        return self.completed_seqs.shape[0]


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class Bottleneck(nn.Module):
    def __init__(self, nChannles, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm1d(nChannles)
        self.conv1 = nn.Conv1d(nChannles, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(interChannels)
        self.conv2 = nn.Conv1d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # print("bottle_input:")
        # print(x.shape)
        out = self.conv1(F.relu(self.bn1(x)))
        # print("bottle_1:")
        # print(out.shape)
        out = self.conv2(F.relu(self.bn2(out)))
        # print("bottle_2:")
        # print(out.shape)
        out = torch.cat((x, out), 1)
        # print("bottle_out:")
        # print(out.shape)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        out = F.avg_pool1d(out, 2)
        return out


class Block(nn.Module):
    def __init__(self, growthRate, depth, reduction):
        super(Block, self).__init__()
        nDenseBlocks = (depth - 4) // 3
        nDenseBlocks //= 2

        nChannels = int(4.5 * growthRate)
        # 头部卷积
        self.conv1 = nn.Conv1d(4, nChannels, kernel_size=1, bias=False)
        # 第一个block
        self.block1 = self._make_denseblock(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        # 第二个block
        self.block2 = self._make_denseblock(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        # 结尾池化
        self.pool = nn.MaxPool1d(25, 25)
        # 组蛋白所需处理
        nChannels = int(4.5 * growthRate)
        self.conv2 = nn.Conv1d(8, nChannels, kernel_size=1, bias=False)
        self.block3 = self._make_denseblock(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans3 = Transition(nChannels, nOutChannels)

        self.block4 = self._make_denseblock(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans4 = Transition(nChannels, nOutChannels)

        self.pool1 = nn.MaxPool1d(5, 5)
        #
        self.layernorm1 = LayerNorm(200)
        self.layernorm2 = LayerNorm(200)
        # 分类器
        self.L1 = nn.Linear(400, 100)
        self.L2 = nn.Linear(100, 50)
        self.L3 = nn.Linear(50, 1)
        self.LSM = nn.Sigmoid()

    def _make_denseblock(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x, y):
        out = F.relu(self.conv1(x))

        out = self.trans1(self.block1(out))

        out = self.trans2(self.block2(out))

        out = self.pool(out)
        out = out.view(-1, 200)

        out1 = F.relu(self.conv2(y))
        out1 = self.trans3(self.block3(out1))
        out1 = self.trans4(self.block4(out1))
        out1 = self.pool1(out1)
        out1 = out1.view(-1, 200)

        out = self.layernorm1(out)
        out1 = self.layernorm2(out1)

        out = torch.cat((out, out1), 1)

        out = F.relu(self.L1(out))
        out = F.relu(self.L2(out))
        out = self.L3(out)
        out = self.LSM(out)

        return out


def one_hot(seq):

    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]}

    code = np.empty(shape=(len(seq), 4))
    for location, base in enumerate(seq, start=0):
        code[location] = base_map[base]

    return code


class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, model_name):
        """
            定义 __call__ 函数 -> 将一个类视作一个函数
            该函数的目的 类似在class中重载()运算符
            使得这个类的实例对象可以和普通函数一样 call
            即，通过 对象名() 的形式使用
        """

        score = -val_loss

        if self.best_score is None:
            """
                初始化（第一次call EarlyStopping）
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_name)
        elif score < self.best_score + self.delta:
            """
                验证集损失没有继续下降时，计数
                当计数 大于 耐心值时，停止
                注：
                    由于模型性能没有改善，此时是不保存检查点的
            """
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            """
                验证集损失下降了，此时从头开始计数
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, model_name):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        """
            保存最优的模型Parameters
        """
        save_path = './SavedModels'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("1")
        print(model_name)
        torch.save(model.state_dict(), save_path + '/' + model_name + '.pth')
        self.val_loss_min = val_loss
        print("2")


class Constructor:
    """
        按照不同模型的接收维数，修改相关的样本维数，如：
        特征融合策略不同，卷积操作不同（1D或2D），是否融合形状特征等
    """

    def __init__(self, model, model_name='DenseBlock'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        self.loss_function = nn.BCELoss()
        self.root_path = os.path.abspath(os.curdir)

        self.batch_size = 64
        self.epochs = 15

    def save_model(self):
        print("save model")
        save_path = './SavedModels'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("2")
        torch.save(self.model.state_dict(), save_path + '/' + self.model_name + '.pth')

    def save_evaluation_indicators(self, indicators):
        # save_path = self.root_path + "/SavedIndicators"
        save_path = './SavedIndicators'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    #     写入评价指标
        file_name = save_path + "/indicators.xlsx"
        file = open(file_name, "a")
        file.write(str(indicators[0]) + " " + str(np.round(indicators[1], 4)) + " " + str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + "\n")
        file.close()

    def learn(self, TrainLoader, ValidateLoader):
        # Get Current path
        path = os.path.abspath(os.curdir)
        early_stopping = EarlyStopping(patience=5, verbose=True)

        for epoch in range(self.epochs):
            self.model.to(self.device)
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("{} Epoch {}".format(self.model_name, epoch))
                seq, histone, label = data
                #output = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
                #output = self.model(seq.to(self.device, dtype=torch.float), shape.to(self.device, dtype=torch.float))
                #output = self.model(seq.to(self.device, dtype=torch.float))
                output = self.model(seq.to(self.device, dtype=torch.float), histone.to(self.device, dtype=torch.float))
                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()

            valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_seq, valid_histone, valid_labels in ValidateLoader:
                    #valid_output = self.model(valid_seq.unsqueeze(1).to(self.device), valid_shape.unsqueeze(1).to(self.device))
                    #valid_output = self.model(valid_seq.to(self.device, dtype=torch.float),valid_shape.to(self.device, dtype=torch.float))
                    #valid_output = self.model(valid_seq.to(self.device, dtype=torch.float))
                    valid_output = self.model(valid_seq.to(self.device, dtype=torch.float), valid_histone.to(self.device, dtype=torch.float))
                    valid_labels = valid_labels.float().to(self.device)

                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)
            early_stopping(valid_loss_avg, self.model, self.root_path, self.model_name)


        print('\n---Finish Learn---\n')

    def inference(self, TestLoader):
        print("inference")

        # path = os.path.abspath(os.curdir)
        load_path = './SavedModels'
        print("3")
        self.model.load_state_dict(torch.load(load_path + '/' + self.model_name + '.pth', map_location='cpu'))
        self.model.to("cpu")

        predicted_value = []
        ground_label = []
        self.model.eval()

        for seq, histone, label in TestLoader:
            # output = self.model(seq.unsqueeze(1), shape.unsqueeze(1))
            #output = self.model(seq.float(), shape.float())
            #output = self.model(seq.float())
            output = self.model(seq.float(), histone.float())
            """ To scalar"""
            predicted_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().numpy())
            ground_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().numpy())

        print('\n---Finish Inference---\n')

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        print("measure")
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=ground_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=ground_label)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=ground_label)
        pr_auc = auc(recall, precision)

        # 写入评价指标
        indicators = [self.model_name, accuracy, roc_auc, pr_auc]
        self.save_evaluation_indicators(indicators)
        print('\n---Finish Measure---\n')

        return accuracy, roc_auc, pr_auc

    def run(self, samples_file_name, ratio=0.8):
        print("run")
        Train_Validate_Set = SSDataset_690(samples_file_name, False)

        """divide Train samples and Validate samples"""
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))
        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,batch_size=self.batch_size, shuffle=False, num_workers=0)

        Test_Set = SSDataset_690(samples_file_name, Test=True)
        TestLoader = loader.DataLoader(dataset=Test_Set,batch_size=1, shuffle=False, num_workers=0)

        self.learn(TrainLoader, ValidateLoader)
        predicted_value, ground_label = self.inference(TestLoader)

        accuracy, roc_auc, pr_auc = self.measure(predicted_value, ground_label)
        print("{}: ACC={} ROC={} PR={}".format(self.model_name, accuracy, roc_auc, pr_auc))


        print('\n---Finish Run---\n')

        return accuracy, roc_auc, pr_auc

if __name__ == "__main__":
    ROOT_PATH = "./data"
    path_cell = glob.glob(ROOT_PATH + "/*")
    print(path_cell)
    path_cell.sort()

    for index_cell, item_cell in enumerate(path_cell):
        print("itemcell:   "+item_cell)
        item_cell = item_cell.split("/")[-1]
        cell_name = item_cell.split("\\")[-1]
        print("cellname:   "+cell_name)
        path_tf = glob.glob(item_cell + "/*")
        path_tf.sort()
        for index_tf, item_tf in enumerate(path_tf):
            tf_name = item_tf.split("/")[-1]
            tf_name = tf_name.split("\\")[-1]
            print(tf_name)
            model_name = cell_name + "_" + tf_name
            print(model_name)
            # 从断点继续训练
            # point = "/content/drive/MyDrive/BC/k562/TRIM28"
            # if item_tf < point:
            #  continue
            Train = Constructor(model=Block(growthRate=16, depth=28, reduction=1), model_name=model_name)
            accuracy, roc_auc, pr_auc = Train.run(samples_file_name=item_tf)


