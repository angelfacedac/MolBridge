import os.path

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, recall_score, precision_score, \
    f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from datetime import datetime

from Params import args
from utils import Mydata, collate_fn
from mymodel import MyModel


def init_data(kth_split=None):
    if kth_split is not None:
        train_data = Mydata(root_path=os.path.join(args.root_path, str(kth_split)), kind_path=args.train_path)
        val_data = Mydata(root_path=os.path.join(args.root_path, str(kth_split)), kind_path=args.val_path)
        test_data = Mydata(root_path=os.path.join(args.root_path, str(kth_split)), kind_path=args.test_path)
    else:
        train_data = Mydata(root_path=args.root_path, kind_path=args.train_path)
        val_data = Mydata(root_path=args.root_path, kind_path=args.val_path)
        test_data = Mydata(root_path=args.root_path, kind_path=args.test_path)

    print(f"train_data_size: {len(train_data)}")
    print(f"val_data_size: {len(val_data)}")
    print(f"test_data_size: {len(test_data)}")

    train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def init_model_and_tools():
    model = MyModel()
    loss_fn = getattr(nn, args.loss_fn)()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    model.to(args.device)

    # 日志相关
    log_root_path = os.path.join("Out", f"{logdir_name}", f"{args.root_path[5] + str(k)}")
    writer = SummaryWriter(os.path.join(log_root_path, "tensorboard"))
    params_path = os.path.join(log_root_path, "params.txt")
    mymodel_path = os.path.join(log_root_path, "model.txt")
    checkpoint_path = os.path.join(log_root_path, "checkpoint")
    with open("Params.py", 'r', encoding='utf-8') as py_file:
        content = py_file.read()
    with open(params_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(content)
    with open("mymodel.py", 'r', encoding='utf-8') as py_file:
        content = py_file.read()
    with open(mymodel_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(content)

    return model, loss_fn, opt, writer, checkpoint_path


def train(model, dataloader, loss_fn, opt, epoch, writer):
    global all_batch
    # 训练模型
    model.train()
    batch = 0
    sum_loss = 0
    sum_L1 = 0
    for data in dataloader:
        batch += 1
        all_batch += 1
        embeds, adjs, masks, cnn_masks, targets = data
        P, _, L1 = model(embeds, adjs, masks, cnn_masks)
        loss = loss_fn(P, targets) + L1 * args.alpha
        sum_L1 += L1
        # print(f"第 {batch} 批训练loss:  {loss.item()}")
        writer.add_scalar(f"train_loss ( all_batch )", loss.item(), all_batch)

        sum_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"第 {epoch} 轮训练sumLoss:  {sum_loss}")
    print(f"第 {epoch} 轮训练L1loss:  {sum_L1}")
    print(f"第 {epoch} 轮训练loss:  {sum_loss - sum_L1 * args.alpha}")
    writer.add_scalar("train_loss_sum", sum_loss, epoch)
    writer.add_scalar("train_loss_model", sum_loss - sum_L1 * args.alpha, epoch)
    writer.add_scalar("train_loss_L1", sum_L1, epoch)


def save_checkpoint(checkpoint_path, model, opt, epoch):
    # 保存模型
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict()
    }, os.path.join(checkpoint_path, f"{epoch:03d}.pth"))
    print(f"第 {epoch} 轮checkpoint已保存")


def remove_checkpoint(checkpoint_path):
    os.remove(checkpoint_path)
    print(f"remove {checkpoint_path} success!")


def valida(model, dataloader, loss_fn, epoch, writer, ACC_list, F1_list, two_class=False):
    print(f"----------第 {epoch} 轮验证开始-----------")
    # 验证模型
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        y_true = []
        y_pred = []
        y_prob = []
        for data in dataloader:
            embeds, adjs, masks, cnn_masks, targets = data
            P, _, L1 = model(embeds, adjs, masks, cnn_masks)
            loss = loss_fn(P, targets) + L1 * args.alpha

            sum_loss += loss.item()
            y_true += targets.int().cpu().tolist()
            if two_class:
                y_pred += (nn.functional.sigmoid(P) > 0.5).int().cpu().tolist()
            else:
                y_pred += torch.argmax(P, dim=1).int().cpu().tolist()
            y_prob += P.cpu().tolist()

        print(f"第 {epoch} 轮验证loss:  {sum_loss}")
        writer.add_scalar("val_loss", sum_loss, epoch)

        if two_class:
            # 计算 AUC
            auc_score = roc_auc_score(y_true, y_prob)
            print("AUC:", auc_score)

            # 计算 AUPR
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            aupr_score = auc(recall, precision)
            print("AUPR:", aupr_score)

            # 计算准确率
            accuracy = accuracy_score(y_true, y_pred)
            print("Accuracy:", accuracy)

            # 计算召回率
            recall = recall_score(y_true, y_pred)
            print("Recall:", recall)

            # 计算精确率
            precision = precision_score(y_true, y_pred)
            print("Precision:", precision)

            # 计算 F1 Score
            f1 = f1_score(y_true, y_pred)
            print("F1 Score:", f1)

            # 计算混淆矩阵
            conf_matrix = confusion_matrix(y_true, y_pred)
            # print("Confusion Matrix:\n", conf_matrix)

            # 日志
            writer.add_scalar("val AUC", auc_score, epoch)
            writer.add_scalar("val AUPR", aupr_score, epoch)
            writer.add_scalar("val Accuracy", accuracy, epoch)
            writer.add_scalar("val Recall", recall, epoch)
            writer.add_scalar("val Precision", precision, epoch)
            writer.add_scalar("val F1 Score", f1, epoch)

        else:
            # 生成分类报告，转换为字典格式
            report = classification_report(y_true, y_pred, output_dict=True)

            # 访问宏平均 (macro avg) 的指标
            macro_precision = report['macro avg']['precision']
            macro_recall = report['macro avg']['recall']
            macro_f1_score = report['macro avg']['f1-score']

            # 打印宏平均指标
            # print(
            #     f"Macro Avg - Precision: {macro_precision:.2f}, Recall: {macro_recall:.2f}, F1-Score: {macro_f1_score:.2f}")

            # 访问加权平均 (weighted avg) 的指标
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1_score = report['weighted avg']['f1-score']

            # 打印加权平均指标
            # print(
            #     f"Weighted Avg - Precision: {weighted_precision:.2f}, Recall: {weighted_recall:.2f}, F1-Score: {weighted_f1_score:.2f}")

            # 使用 precision_recall_fscore_support 计算微平均
            micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                                               average='micro')

            # print(
            #     f"Micro Avg - Precision: {micro_precision:.2f}, Recall: {micro_recall:.2f}, F1-Score: {micro_f1_score:.2f}")

            # 计算准确率
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {accuracy}")

            # 日志
            writer.add_scalar("val Macro Precision", macro_precision, epoch)
            writer.add_scalar("val Macro Recall", macro_recall, epoch)
            writer.add_scalar("val Macro F1", macro_f1_score, epoch)
            writer.add_scalar("val Weighted Precision", weighted_precision, epoch)
            writer.add_scalar("val Weighted Recall", weighted_recall, epoch)
            writer.add_scalar("val Weighted F1", weighted_f1_score, epoch)
            writer.add_scalar("val Micro Precision", micro_precision, epoch)
            writer.add_scalar("val Micro Recall", micro_recall, epoch)
            writer.add_scalar("val Micro F1", micro_f1_score, epoch)
            writer.add_scalar("val Accuracy", accuracy, epoch)

            ACC_list.append((-accuracy, epoch))
            F1_list.append((-macro_f1_score, epoch))


#  checkpoint_path = os.path.join(checkpoint_path, f"{epoch:03d}.pth")
def get_model_by_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = MyModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    return model


def test(model, test_dataloader, epoch=None, writer=None, two_class=False, ACC_list=None, F1_list=None):
    print(f"---------------模型测试开始-------------------")
    # checkpoint = torch.load(checkpoint_path)
    # model = MyModel()
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(args.device)

    model.eval()
    with torch.no_grad():

        if two_class:
            y_true = []
            y_pred = []
            y_prob = []
            for data in test_dataloader:
                embeds, adjs, masks, cnn_masks, targets = data
                P, _, L1 = model(embeds, adjs, masks, cnn_masks)

                y_true += targets.int().cpu().tolist()
                y_pred += (nn.functional.sigmoid(P) > 0.5).int().cpu().tolist()
                y_prob += P.cpu().tolist()

            # 计算 AUC
            auc_score = roc_auc_score(y_true, y_prob)
            print("AUC:", auc_score)

            # 计算 AUPR
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            aupr_score = auc(recall, precision)
            print("AUPR:", aupr_score)

            # 计算准确率
            accuracy = accuracy_score(y_true, y_pred)
            print("Accuracy:", accuracy)

            # 计算召回率
            recall = recall_score(y_true, y_pred)
            print("Recall:", recall)

            # 计算精确率
            precision = precision_score(y_true, y_pred)
            print("Precision:", precision)

            # 计算 F1 Score
            f1 = f1_score(y_true, y_pred)
            print("F1 Score:", f1)

            # 计算混淆矩阵
            conf_matrix = confusion_matrix(y_true, y_pred)
            # print("Confusion Matrix:\n", conf_matrix)

            # 日志
            if writer:
                writer.add_scalar("test AUC", auc_score, epoch)
                writer.add_scalar("test AUPR", aupr_score, epoch)
                writer.add_scalar("test Accuracy", accuracy, epoch)
                writer.add_scalar("test Recall", recall, epoch)
                writer.add_scalar("test Precision", precision, epoch)
                writer.add_scalar("test F1", f1, epoch)
        else:
            y_true = []
            y_pred = []
            for data in test_dataloader:
                embeds, adjs, masks, cnn_masks, targets = data
                P, _, L1 = model(embeds, adjs, masks, cnn_masks)

                y_true += targets.int().cpu().tolist()
                y_pred += torch.argmax(P, dim=1).int().cpu().tolist()

            # 生成分类报告，转换为字典格式
            report = classification_report(y_true, y_pred, output_dict=True)

            # 访问宏平均 (macro avg) 的指标
            macro_precision = report['macro avg']['precision']
            macro_recall = report['macro avg']['recall']
            macro_f1_score = report['macro avg']['f1-score']

            # 打印宏平均指标
            # print(
            #     f"Macro Avg - Precision: {macro_precision:.2f}, Recall: {macro_recall:.2f}, F1-Score: {macro_f1_score:.2f}")

            # 访问加权平均 (weighted avg) 的指标
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1_score = report['weighted avg']['f1-score']

            # 打印加权平均指标
            # print(
            #     f"Weighted Avg - Precision: {weighted_precision:.2f}, Recall: {weighted_recall:.2f}, F1-Score: {weighted_f1_score:.2f}")

            # 使用 precision_recall_fscore_support 计算微平均
            micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                                               average='micro')

            # print(
            #     f"Micro Avg - Precision: {micro_precision:.2f}, Recall: {micro_recall:.2f}, F1-Score: {micro_f1_score:.2f}")

            # 计算准确率
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {accuracy}")

            # 日志
            if writer is not None:
                writer.add_scalar("test Macro Precision", macro_precision, epoch)
                writer.add_scalar("test Macro Recall", macro_recall, epoch)
                writer.add_scalar("test Macro F1", macro_f1_score, epoch)
                writer.add_scalar("test Weighted Precision", weighted_precision, epoch)
                writer.add_scalar("test Weighted Recall", weighted_recall, epoch)
                writer.add_scalar("test Weighted F1", weighted_f1_score, epoch)
                writer.add_scalar("test Micro Precision", micro_precision, epoch)
                writer.add_scalar("test Micro Recall", micro_recall, epoch)
                writer.add_scalar("test Micro F1", micro_f1_score, epoch)
                writer.add_scalar("test Accuracy", accuracy, epoch)

                ACC_list.append((-accuracy, epoch))
                F1_list.append((-macro_f1_score, epoch))


def save_best_result(checkpoint_path, ACC_list_val, F1_list_val, ACC_list_test, F1_list_test):
    with open(os.path.join(checkpoint_path, "result.txt"), 'w') as file:
        for vec, flag in [(ACC_list_val, "ACC_val"),
                          (F1_list_val, "F1_val"),
                          (ACC_list_test, "ACC_test"),
                          (F1_list_test, "F1_test")]:
            text = ""
            text_ = ""
            for value, ansEpoch in vec[:args.epoch//50]:
                text += f"{ansEpoch:6d}" + " "
                text_ += f"{-value:>6.4f}" + " "
            file.write(flag + "\n")
            file.write(text + "\n")
            file.write(text_ + "\n")


def main():
    global all_batch

    ACC_list_val = []
    F1_list_val = []
    ACC_list_test = []
    F1_list_test = []

    train_dataloader, val_dataloader, test_dataloader = init_data(kth_split=k)

    model, loss_fn, opt, writer, checkpoint_path = init_model_and_tools()

    all_batch = 0

    for i in range(args.epoch):
        epoch = i + 1
        print(f"----------第 {epoch} 轮训练开始-----------")
        train(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            opt=opt,
            epoch=epoch,
            writer=writer
        )
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            opt=opt,
            epoch=epoch
        )
        valida(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            epoch=epoch,
            writer=writer,
            two_class=False,
            ACC_list=ACC_list_val,
            F1_list=F1_list_val
        )
        test(
            model=model,
            test_dataloader=test_dataloader,
            epoch=epoch,
            writer=writer,
            two_class=False,
            ACC_list=ACC_list_test,
            F1_list=F1_list_test
        )

    writer.close()

    ACC_list_val.sort()
    F1_list_val.sort()
    ACC_list_test.sort()
    F1_list_test.sort()

    save_best_result(checkpoint_path, ACC_list_val, F1_list_val, ACC_list_test, F1_list_test)

    for v, epoch in F1_list_test[args.epoch//50:]:
        remove_checkpoint(os.path.join(checkpoint_path, f"{epoch:03d}.pth"))



if __name__ == '__main__':
    logdir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # logdir_name = "2024-11-16_02-37-24"

    for k in range(5):
        main()
