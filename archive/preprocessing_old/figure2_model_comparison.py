import matplotlib.pyplot as plt

models = [
    "All genes\n(25,329)",
    "Selected genes\n(500)"
]

accuracy = [0.50, 0.92]

plt.figure(figsize=(6,5))

bars = plt.bar(models, accuracy)

plt.ylabel("Accuracy")
plt.title("Effect of Feature Selection on Parkinson Classification")
plt.ylim(0, 1.0)

for bar, acc in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        acc + 0.02,
        f"{acc:.2f}",
        ha="center"
    )

plt.savefig(
    "data/figure2_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure saved: data/figure2_model_comparison.png")