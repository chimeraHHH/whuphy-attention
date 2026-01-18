import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def parse_results(file_path):
    actuals = []
    preds = []
    with open(file_path, 'r') as f:
        for line in f:
            # 解析 "Pred: 17.5815, Actual: 17.9646"
            match = re.search(r'Pred:\s*([-\d\.]+),\s*Actual:\s*([-\d\.]+)', line)
            if match:
                preds.append(float(match.group(1)))
                actuals.append(float(match.group(2)))
    return np.array(preds), np.array(actuals)

def plot_parity(preds, actuals, save_path='parity_plot.png'):
    plt.figure(figsize=(8, 7), dpi=150)
    
    # 绘制散点
    plt.scatter(actuals, preds, alpha=0.6, c='#1f77b4', edgecolors='w', s=80, label='Test Samples')
    
    # 绘制对角线 (Perfect Prediction)
    min_val = min(min(actuals), min(preds)) - 1
    max_val = max(max(actuals), max(preds)) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    # 计算指标
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    
    # 添加文本框
    textstr = '\n'.join((
        r'$\mathrm{MAE}=%.3f$ eV' % (mae, ),
        r'$\mathrm{RMSE}=%.3f$ eV' % (rmse, ),
        r'$R^2=%.3f$' % (r2, )))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.xlabel('DFT Calculated Formation Energy (eV)', fontsize=14)
    plt.ylabel('Model Predicted Formation Energy (eV)', fontsize=14)
    plt.title('Defect Formation Energy Prediction', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    result_file = r"D:\Github hanjia\whuphy-attention\BJQ\prediction_results.txt"
    try:
        preds, actuals = parse_results(result_file)
        if len(preds) == 0:
            print("No data found in results file.")
        else:
            print("-" * 30)
            print(f"Analyzed {len(preds)} samples:")
            print(f"MAE  : {mean_absolute_error(actuals, preds):.4f} eV")
            print(f"RMSE : {np.sqrt(mean_squared_error(actuals, preds)):.4f} eV")
            print("-" * 30)
            
            plot_parity(preds, actuals, r"D:\Github hanjia\whuphy-attention\BJQ\final_result_plot.png")
    except Exception as e:
        print(f"Error: {e}")
