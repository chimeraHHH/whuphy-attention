import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_curve(csv_path, save_path):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 设置风格
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Loss (MSE)', color=color, fontsize=14, fontweight='bold')
    l1 = ax1.plot(df['epoch'], df['train_loss_mse'], color=color, linewidth=2, label='Train Loss (MSE)')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 双轴：右侧显示验证集 MAE
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Validation MAE (eV)', color=color, fontsize=14, fontweight='bold')
    l2 = ax2.plot(df['epoch'], df['val_mae'], color=color, linewidth=2, linestyle='--', label='Val MAE')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

    # 合并图例
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)

    plt.title('Training Convergence Analysis', fontsize=16, pad=20)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Training curve saved to {save_path}")

if __name__ == "__main__":
    csv_file = r"D:\Github hanjia\whuphy-attention\BJQ\training_history.csv"
    output_file = r"D:\Github hanjia\whuphy-attention\BJQ\training_curve.png"
    
    try:
        plot_curve(csv_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
