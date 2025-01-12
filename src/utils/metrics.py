from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def metrics(y_true, y_pred):

    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    accuracy = accuracy_score(y_true, y_pred)

    return macro_precision, macro_recall, macro_f1, accuracy

