import os
from collections import defaultdict, deque

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from tqdm import tqdm

from src.datasets.dataloader.feature_encoding import smile_to_graph

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def draw_bar_for_num_of_atom():
    for fold in range(5):
        cnt = defaultdict(int)
        for split in ["train", "val", "test"]:
            path = "src/data/Deng/" + str(fold) + "/" + split + ".csv"
            df = pd.read_csv(path)
            for s1, s2, y in tqdm(df.values):
                num = smile_to_graph(s1)[1].shape[0]
                cnt[num] += 1
                num = smile_to_graph(s2)[1].shape[0]
                cnt[num] += 1
        keys = sorted(cnt.keys())
        values = [cnt[k] for k in keys]
        plt.figure(figsize=(10, 6))
        plt.bar(keys, values)
        plt.savefig(f"src/data/Deng/{fold}/cnt.png")


def cal_eve_shortest_path_len(num, adj):
    if num==1:
        return 0

    dp = [[1e9]*(num+1) for _ in range(num+1)]
    for i in range(1, num+1):
        for j in range(1, num+1):
            if i==j:
                dp[i][j] = 0
                continue
            if adj[i-1][j-1]==0:
                continue
            dp[i][j] = adj[i-1][j-1]

    for k in range(1, num+1):
        for i in range(1, num+1):
            for j in range(1, num+1):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])

    ans = 0
    for i in range(1, num+1):
        for j in range(1, num+1):
            if dp[i][j] == 1e9: continue
            ans += dp[i][j]
    tot = num*(num-1)
    return ans / tot


def split_by_shortest_path_len(path, save_path):
    df = pd.read_csv(path)
    ans_list = []
    for s1, s2, y in tqdm(df.values):
        _, adj = smile_to_graph(s1)
        num = adj.shape[0]
        ans1 = cal_eve_shortest_path_len(num, adj)
        _, adj = smile_to_graph(s2)
        num = adj.shape[0]
        ans2 = cal_eve_shortest_path_len(num, adj)
        ans = ans1 + ans2
        ans_list.append(ans)

    # 添加ans列并排序
    df['ans'] = ans_list
    df_sorted = df.sort_values('ans').reset_index(drop=True)

    ##############################################
    # 新增部分：绘制密度分布图
    ##############################################
    plt.figure(figsize=(10, 6))

    # 绘制密度曲线
    sns.kdeplot(df_sorted['ans'], fill=True, label='Density', color='skyblue')

    # 计算分位数位置（20%, 40%, 60%, 80%）
    quantiles = np.percentile(df_sorted['ans'], [20, 40, 60, 80])  # todo

    # 标注分界线
    for i, q in enumerate(quantiles):
        plt.axvline(q, color='red', linestyle='--', alpha=0.7,
                    label=f'Part {i + 1}-{i + 2} Boundary' if i == 0 else "")
        # plt.text(q, plt.ylim()[1] * 0.8, f'{q:.2f}\n({(i + 1) * 20}%)',
        #          rotation=90, ha='right', va='top')

    # 图表美化
    plt.title('Density Distribution of average shortest path with Partition Boundaries', fontsize=14)
    plt.xlabel('ans Value (sum of shortest path lengths)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()

    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, 'ans_distribution.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nSaved density plot to {plot_path}")
    ##############################################

    # 均分为五部分
    chunks = np.array_split(df_sorted, 5)  # todo

    # 保存每个子数据集，不包含ans列
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(save_path, f'part_{i + 1}.csv')
        chunk.drop(columns=['ans']).to_csv(chunk_path, index=False)
        print(f"Saved part {i + 1} to {chunk_path}")


def calculate_avg_shortest_paths():
    """计算平均最短路径长度并保存结果"""
    import os
    import numpy as np
    import pandas as pd
    import pickle
    from tqdm import tqdm

    result_file = "Deng/short_path.npy"
    cache_file = "Deng/molecule_cache.pkl"

    # 检查结果文件是否存在，存在则直接返回
    if os.path.exists(result_file):
        print(f"结果文件 {result_file} 已存在，直接加载...")
        return np.load(result_file)

    # 尝试加载分子缓存
    molecule_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                molecule_cache = pickle.load(f)
            print(f"已加载 {len(molecule_cache)} 个分子的缓存")
        except Exception as e:
            print(f"加载缓存失败: {e}")
            molecule_cache = {}

    ans_list = []

    for fold in range(5):
        df = pd.read_csv(f"Deng/{fold}/test.csv")
        for s1, s2, y in tqdm(df.values):
            # 检查第一个分子是否已经计算过
            if s1 not in molecule_cache:
                _, adj = smile_to_graph(s1)
                num = adj.shape[0]
                ans1 = cal_eve_shortest_path_len(num, adj)
                molecule_cache[s1] = ans1
            else:
                ans1 = molecule_cache[s1]

            # 检查第二个分子是否已经计算过
            if s2 not in molecule_cache:
                _, adj = smile_to_graph(s2)
                num = adj.shape[0]
                ans2 = cal_eve_shortest_path_len(num, adj)
                molecule_cache[s2] = ans2
            else:
                ans2 = molecule_cache[s2]

            ans = (ans1 + ans2) / 2
            ans_list.append(ans)

    # 保存分子缓存，方便下次使用
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(molecule_cache, f)
        print(f"已保存 {len(molecule_cache)} 个分子的缓存")
    except Exception as e:
        print(f"保存缓存失败: {e}")

    ans = np.array(ans_list)
    ans.sort()
    # 保存结果
    np.save(result_file, ans)
    print(f"已保存结果到 {result_file}")

    return ans


def draw_distribution(data_file="Deng/short_path.npy"):
    """绘制最短路径长度的密度分布图"""
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 加载数据
    ans = np.load(data_file)

    plt.figure(figsize=(10, 6))

    # 绘制密度曲线
    sns.kdeplot(ans, fill=True, label='Density', color='skyblue')

    # 计算分位数位置（20%, 40%, 60%, 80%）
    quantiles = np.percentile(ans, [20, 40, 60, 80])

    # 标注分界线
    for i, q in enumerate(quantiles):
        plt.axvline(q, color='red', linestyle='--', alpha=0.7,
                    label='Boundary' if i == 0 else "")
        plt.text(q, plt.ylim()[1] * 0.8, f'{q:.2f} ({(i + 1) * 20}%)',
                 rotation=90, ha='right', va='top', fontsize=15)

    # 图表美化
    plt.title('Density Distribution of average shortest path', fontsize=18)
    plt.xlabel('average of shortest path lengths', fontsize=18)
    plt.ylabel('Density', fontsize=18)

    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend()
    plt.tight_layout()

    plt.show()


def draw_dis():
    """计算并绘制最短路径长度分布"""
    # 计算或加载数据
    calculate_avg_shortest_paths()
    # 绘制分布图
    draw_distribution()

draw_dis()


