import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_donut_with_legend(
    labels,
    values,
    colors=None,
    dataset_name="Dermnet-ISIC",
    legend_title="Pathologies",
    outfile="mimic_macd_donut.png",
    dpi=300
):
    # 基础样式
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "figure.dpi": dpi
    })

    total = sum(values)
    n = len(labels)

    # 配色（接近你给的示例：青绿系，深浅过渡）
    if colors is None:
        colors = [
            "#216b72",  # 深青
            "#3aa07a",  # 青绿
            "#bfdde0",  # 淡青灰
            "#cfe6c8",  # 淡绿
            "#e6f3de",  # 更淡绿
            "#2f8e5b",  # 深绿
            "#5aa3a5",  # 蓝绿
        ][:n]

    # 画布与子图
    fig = plt.figure(figsize=(12, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    # 画环形饼图
    # radius 控制整体大小，width 控制环宽；edgecolor/linewidth 做出白色分隔线
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=100,           # 起始角度微调让分块位置更美观
        counterclock=False,       # 顺时针
        radius=1.15,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=3, joinstyle="round"),
        labels=None
    )

    # 中心文本
    ax.text(0, 0.18, dataset_name, ha="center", va="center",
            fontsize=22, fontweight="bold")
    ax.text(0, 0.03, f"{n} Pathologies", ha="center", va="center",
            fontsize=13, color="#445")
    ax.text(0, -0.14, f"{total}", ha="center", va="center",
            fontsize=28, fontweight="bold")
    ax.text(0, -0.28, "Total Cases", ha="center", va="center",
            fontsize=12, color="#566")

    # 构造图例句柄
    handles = [mpl.patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]

    # 右侧图例（圆角白底，接近示例样式）
    leg = ax.legend(
        handles=handles,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.12, 0.5),  # 控制在右侧居中
        frameon=True,
        facecolor="white",
        edgecolor="#e5e7eb",
        fancybox=True,
        framealpha=1.0,
        borderpad=1.0,
        labelspacing=1.2,
        fontsize=16,
        title_fontsize=20
    )

    # 让图例里的小色块更宽一点，观感更像示例
    for h in leg.legend_handles:
        h.set_width(18)
        h.set_height(8)

    # 去掉坐标轴
    ax.set_axis_off()

    # 导出
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved -> {outfile}")

if __name__ == "__main__":
    # 示例数据（和 4390 对齐）
    labels = [
        "Acne and Rosacea",
        "Connective Tissue diseases",
        "Seborrheic Keratoses",
        "Lupus",
        "Hair Loss Photos Alopecia",
        "Malignant Melanoma",
        "Nenign Melanoma",
    ]
    values = [980, 820, 510, 460, 380, 640, 600]  # 总和=4390

    draw_donut_with_legend(
        labels=labels,
        values=values,
        dataset_name="Dermnet-ISIC",
        legend_title="Pathologies",
        outfile="mimic_macd_donut.png",
        dpi=300
    )