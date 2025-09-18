import numpy as np

from src.datasets.get_dataloader import get_dataloader
from src.load_config import CONFIG
from src.utils import set_seed_
from src.utils.manager import Manager

@set_seed_(CONFIG['train']['seed'])
def repetition(model_params_path):
    manager = Manager()
    result = {
        "acc" : [],
        "f1"  : [],
        "pre" : [],
        "rec" : []
    }

    for fold in range(5):
        dataloader = get_dataloader(fold, "test")
        acc, f1, pre, rec = manager.test(dataloader, model_params_path=model_params_path[fold])
        result["acc"].append(acc)
        result["f1"].append(f1)
        result["pre"].append(pre)
        result["rec"].append(rec)

    stats = {}
    for metric in result.keys():
        mean_val = np.mean(result[metric])
        std_val = np.std(result[metric], ddof=1)
        stats[metric] = {
            "mean" : mean_val,
            "std"  : std_val
        }
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

@set_seed_(CONFIG['train']['seed'])
def repetition_mata(model_params_path):
    manager = Manager()
    result = {
        "meta": {
            "acc": 0,
            "f1": 0,
            "pre": 0,
            "rec": 0
        },
        "metano": {
            "acc": 0,
            "f1": 0,
            "pre": 0,
            "rec": 0
        },
        "test": {
            "acc": 0,
            "f1": 0,
            "pre": 0,
            "rec": 0
        }
    }

    # 测试 meta 数据集
    dataloader = get_dataloader(4, "meta")
    acc, f1, pre, rec = manager.test(dataloader, model_params_path=model_params_path)
    result["meta"]["acc"] = acc
    result["meta"]["f1"] = f1
    result["meta"]["pre"] = pre
    result["meta"]["rec"] = rec

    # 测试 metano 数据集
    dataloader = get_dataloader(4, "metano")
    acc, f1, pre, rec = manager.test(dataloader, model_params_path=model_params_path)
    result["metano"]["acc"] = acc
    result["metano"]["f1"] = f1
    result["metano"]["pre"] = pre
    result["metano"]["rec"] = rec

    # 测试 test 数据集
    dataloader = get_dataloader(4, "test")
    acc, f1, pre, rec = manager.test(dataloader, model_params_path=model_params_path)
    result["test"]["acc"] = acc
    result["test"]["f1"] = f1
    result["test"]["pre"] = pre
    result["test"]["rec"] = rec

    # 统一输出所有结果
    print("=" * 50 + "meta" + "=" * 50)
    print(f"acc: {result['meta']['acc']:.4f} f1: {result['meta']['f1']:.4f} "
          f"pre: {result['meta']['pre']:.4f} rec: {result['meta']['rec']:.4f}")
    print("=" * 50 + "====" + "=" * 50)

    print("=" * 50 + "metano" + "=" * 50)
    print(f"acc: {result['metano']['acc']:.4f} f1: {result['metano']['f1']:.4f} "
          f"pre: {result['metano']['pre']:.4f} rec: {result['metano']['rec']:.4f}")
    print("=" * 50 + "====" + "=" * 50)

    print("=" * 50 + "test" + "=" * 50)
    print(f"acc: {result['test']['acc']:.4f} f1: {result['test']['f1']:.4f} "
          f"pre: {result['test']['pre']:.4f} rec: {result['test']['rec']:.4f}")
    print("=" * 50 + "====" + "=" * 50)

    return result

if __name__ == '__main__':

    # repetition(
    #     [
    #         "/root/autodl-tmp/MulBridge/logs/Ryu/3d2-batch1024-num4/0/292_valid.pth",
    #         "/root/autodl-tmp/MulBridge/logs/Ryu/3d2-batch1024-num4/1/352_valid.pth",
    #         "/root/autodl-tmp/MulBridge/logs/Ryu/3d2-batch1024-num4/2/424_valid.pth",
    #         "/root/autodl-tmp/MulBridge/logs/Ryu/3d2-batch1024-num4/3/362_valid.pth",
    #         "/root/autodl-tmp/MulBridge/logs/Ryu/3d2-batch1024-num4/4/329_valid.pth"
    #     ]
    # )

    repetition_mata(
        "/root/autodl-tmp/MulBridge/logs/Ryu/MulBridge/3d2-batch1024-num4/4/357_valid.pth"
        # "/root/autodl-tmp/MulBridge/logs/Ryu/MulBridge-woJoint/3d2-batch1024-num4/4/473_valid.pth"
        # "/root/autodl-tmp/MulBridge/logs/Ryu/MulBridge-woSCW/3d2-batch1024-num4/4/495_valid.pth"
        # "/root/autodl-tmp/MulBridge/logs/Ryu/GCN/3d2-batch1024-num4/4/452_valid.pth"
    )
