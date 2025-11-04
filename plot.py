import matplotlib.pyplot as plt
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="assignment1/out_assignment1.out", help="path input file (output from train.py) containing validation perplexity and BLEU score")
parser.add_argument("--output", required=True, help="output filename where you want to save the plot (with .png filetype included)")
parser.add_argument("--comparison", action="store_true", help="specifies if a comparison of model scores has to be made")
parser.add_argument("--second", default="out_assign3_task1.out", help="path to second model training output for the comparison")
args = parser.parse_args()

def plot_perpl_BLEU(input_file, output_file):
	with open(input_file, "r", encoding="utf-8") as f:
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

	fig.savefig(output_file)

def plot_BLEU_compared(input1, input2, output_file):
	with open(input1, "r", encoding="utf-8") as f1:
		infile1 = f1.read()
	with open(input2, "r", encoding="utf-8") as f2:
		infile2 = f2.read()

	pattern_B = r"BLEU \d+\.\d+"
	b1 = re.findall(pattern_B, infile1)
	b2 = re.findall(pattern_B, infile2)

	bleu1 = []
	bleu2 = []

	for elem in b1:
		bleu1.append(float(elem.split()[1]))
	for elem in b2:
		bleu2.append(float(elem.split()[1]))
	
	bleu1 = bleu1[:-2]

	x = list(range(1, len(bleu1) + 1))

	fig, ax = plt.subplots()

	ax.set_xlabel("Epochs (last 2 are on test set)")
	ax.set_ylabel("BLEU score")
	ax.plot(x, bleu1, color="blue", marker="s", label="Baseline Model")
	ax.plot(x, bleu2, color="green", marker="o", label="Joint-BPE Model")

	plt.title("Validation BLEU scores comparison")

	ax.legend()
	
	fig.savefig(output_file)


if __name__ == "__main__":
	if args.comparison and not args.second:
		raise ValueError("Comparison plot needs two input files!")

	if args.comparison:
		plot_BLEU_compared(args.input, args.second, args.output)
	else:
		plot_perpl_BLEU(args.input, args.output)
