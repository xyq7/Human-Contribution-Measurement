import json
import ndjson
import ast
# data path
labeled_data_file = "./human_annotation/labels.ndjson"
from collections import Counter
import matplotlib.pyplot as plt

def parse_labeled_data(file_path):
    total_entries = 0
    all_agree_count = 0
    winner_match_count = 0
    consensus_scores = []
    correct_gaps = []
    incorrect_gaps = []

    all_agree_gap = []
    in_agree_gap = []

    with open(file_path, "r") as f:
        data = ndjson.load(f)
        # Screen data
        for entry in data:
            total_entries += 1
            global_key = entry["data_row"]["global_key"]  
            data = entry['attachments']

            # Get human contribution comparison measurement results
            for item in data:
                value = item.get('value', '')
                if value.startswith('Winner:'):
                    winner = value.split(':')[1].strip()
                elif value.startswith('Gap:'):
                    gap = float(value.split(':')[1].strip())

            # Extract three human contribution comparison labeled by annotators
            label_lists = []
            c_list = []
            human_labels = entry["projects"].get("cm9j9g3qy01g407yt3a45gt30", {}).get("labels", [])
            if human_labels:
                for label in human_labels:
                    annotation = label.get("annotations", {}).get("classifications", [{}])[0].get("radio_answer", {}).get("name", "N/A")
                    label_lists.append(annotation)
                    c_list.append(label.get('performance_details')['consensus_score'])
            # import pdb; pdb.set_trace()
            consensus_scores.append(sum(c_list)/len(c_list))
            # Calculate majority vote
            label_counter = Counter(label_lists)
            if label_counter:
                majority_vote, count = label_counter.most_common(1)[0]
                if count == len(label_lists):
                    all_agree_count += 1
                    all_agree_gap.append(gap)
                else:
                    in_agree_gap.append(gap)
                if majority_vote == winner:
                    winner_match_count += 1
                    correct_gaps.append(gap)
                else:
                    incorrect_gaps.append(gap)

            # Debug prints (optional)
            print(f"Winner: {winner}, Gap: {gap}")
            print(f"Labels: {label_lists}, Majority vote: {majority_vote}")
    all_agree_ratio = all_agree_count / total_entries
    winner_match_ratio = winner_match_count / total_entries
    print(f"Label full agreement ratio: {all_agree_ratio:.4f}")
    print(f"Winner matches majority vote ratio: {winner_match_ratio:.4f}")
    # count_ones = consensus_scores.count(1)
    # print(f"consensus_scores: {count_ones}")

    # Figure
    plt.figure(figsize=(12, 6))

    plt.hist(correct_gaps, bins=30, alpha=0.5, label="Consistent", color="green")
    plt.hist(incorrect_gaps, bins=30, alpha=0.5, label="Inconsistent", color="red")

    plt.xlabel("Gap", fontsize=16)  
    plt.ylabel("Frequency", fontsize=16)  
    plt.title("Measured Human Contribution Gap Distribution", fontsize=18)  
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14)  

    plt.legend(fontsize=14)
    plt.savefig("./figures/gap_distribution.pdf")
    plt.savefig("./figures/gap_distribution.png")
if __name__ == "__main__":
    labeled_data = parse_labeled_data(labeled_data_file)