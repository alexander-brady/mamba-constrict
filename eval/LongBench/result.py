import os, json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results")
args = parser.parse_args()

files = os.listdir(args.results_dir)
output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
compensated = False

for file in files:
    filename = os.path.join(args.results_dir, file)
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    name = '.'.join(file.split('.')[:-1])
    output.append(name+'\t'+str(round(100*(easy_acc+hard_acc)/len(pred_data), 1))+'\t'+str(round(100*easy_acc/easy, 1))+'\t'+str(round(100*hard_acc/hard, 1))+'\t'+str(round(100*short_acc/short, 1))+'\t'+str(round(100*medium_acc/medium, 1))+'\t'+str(round(100*long_acc/long, 1)))

# Write result file to the results directory
result_file = os.path.join(args.results_dir, 'result.txt')
open(result_file, 'w', encoding='utf-8').write('\n'.join(output))
print(f"Results written to {result_file}")
