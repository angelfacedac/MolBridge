import os.path

from src.utils.vis.t_sne import draw_2d, draw_3d

draw_2d(
    data_path=os.path.join("src", "data", "Deng", "0", "test.csv"),
    model_path=os.path.join("logs/Deng/only_block_AdamW/_2048_0d005_500/0", "310_test.pth"),
    save_path=os.path.join("tmp", "t_sne.pdf"),
    device="cpu",
    begin_class=0,
    end_class=20,
    num_sample_per_class=500,
)