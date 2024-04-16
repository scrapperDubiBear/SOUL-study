from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, required=True, help="path of input")
    parser.add_argument("--model_size", type=str, default=None, required=True, help="llm / slm")
    #parser.add_argument("--output_fname", type=str, default=None, required=True, help="path of output file")
    return parser.parse_args()

def main():
    args = parse_args()

    input_path = args.path
    df = pd.read_csv(input_path)

    if args.model_size == 'llm':
        cand_col = 'output'
        ref_col = 'justification'
    else:
        cand_col = 'pred-justification'
        ref_col = 'justification'
    references = df[ref_col].copy()
    candidates = df[cand_col].copy()
    
    
    #Justification Evaluation
    
    #Calculate and print BLEU score
    smoother = SmoothingFunction().method4
    print(corpus_bleu(candidates, references, weights=(1, 0, 0, 0), smoothing_function=smoother)) #UNI-GRAM

    
    # Calculate ROUGE scores for each sentence pair
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(candidate, reference) for candidate, reference in zip(candidates, references)]

    # Compute average Rouge scores
    average_scores = {}
    for metric in scores[0].keys():  # Assuming all sentences have the same Rouge metrics
        metric_scores = [sentence_score[metric].fmeasure for sentence_score in scores]
        average_scores[metric] = sum(metric_scores) / len(metric_scores)

    # Print average Rouge scores
    for metric, r_score in average_scores.items():
        print(f"Average {metric} Score:", r_score)
        
    
    ref = references.tolist()
    cand = candidates.tolist()

    # Calculate BERTScores for all sentences
    P, R, F1 = score(ref, cand, lang='en')

    # Print BERTScores
    print("BERT Precision:", P.mean().item())
    print("BERT Recall:", R.mean().item())
    print("BERT F1-score:", F1.mean().item())
    
if __name__ == "__main__":
    main()