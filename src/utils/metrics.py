import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.load_config import CONFIG

def metrics(y_true, y_pred):

    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    accuracy = accuracy_score(y_true, y_pred)

    return macro_precision, macro_recall, macro_f1, accuracy

def convert_predictions(scores):
    """根据模式转换预测结果"""
    if CONFIG['data']['is_binary']:
        # 二分类：应用sigmoid后使用0.5阈值
        predictions = torch.sigmoid(scores) > 0.5
        return predictions.int()
    else:
        # 多分类：使用argmax
        return torch.argmax(scores, dim=1)

