import matplotlib.pyplot as plt
import re


with open("out_backup_version.out", "r", encoding="utf-8") as f:
	infile = f.read()


pattern_v = r"valid_perplexity \d+\.\d+"
pattern_B = r"BLEU \d+\.\d+"

valid_p = re.findall(pattern_v, infile)
valid_b = re.findall(pattern_B, infile)

perplexity = []
bleu = []

for elem in valid_p:
	perplexity.append(float(elem.split()[1]))
for elem in valid_b:
	bleu.append(float(elem.split()[1]))

print(perplexity, bleu)

steps_perp = list(range(1, len(valid_p) + 1))
steps_bleu = list(range(1, len(valid_b) + 1))

fig, ax1 = plt.subplots()

# Perplexity curve
ax1.set_xlabel("Step")
ax1.set_ylabel("Perplexity", color="red")
ax1.plot(steps_perp, perplexity, color="red", marker="o", label="Perplexity")
ax1.tick_params(axis="y", labelcolor="red")

# BLEU curve (longer)
ax2 = ax1.twinx()
ax2.set_ylabel("BLEU score", color="blue")
ax2.plot(steps_bleu, bleu, color="blue", marker="s", label="BLEU")
ax2.tick_params(axis="y", labelcolor="blue")

plt.title("Validation & Test Metrics")

# Show both legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="lower right")

plt.show()

fig.savefig("perplexity_bleu.png")
