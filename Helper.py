import numpy as np


# Displays the suggested results for each query
def print_queries_results(documents, documents_processed, queries, queries_processed, scores):
    nb_queries_to_show = 3
    nb_results_to_show = 3
    for i in range(0, nb_queries_to_show):
        print(f"\n########### Query n°{i + 1} ###########")
        print(f"Query : \"{queries[i]}\" :")
        arr = np.array(scores[i])
        res = arr.argsort()[-nb_results_to_show:][::-1]
        for idx, r in enumerate(res):
            print(f"--> Result n°{idx + 1} (doc_id = [{r + 1}], score = {arr[r]}) : {documents[r].strip()}")


# Calculates and displays the results metrics for each query
def print_metrics(queries, results, scores):
    nb_results_extracted = 10
    average_precision = 0
    average_rappel = 0
    average_f_score = 0
    nb_invalid_results = 0
    for i in range(0, len(queries)):
        if i+1 not in results:
            nb_invalid_results += 1
        else:
            arr = np.array(scores[i])
            res = arr.argsort()[-nb_results_extracted:][::-1]
            nb_results_in_common = 0
            for idx, r in enumerate(res):
                if r + 1 in results[i + 1]:
                    nb_results_in_common += 1
            precision = nb_results_in_common / (
                nb_results_extracted if nb_results_extracted <= len(results[i + 1]) else len(results[i + 1]))
            rappel = nb_results_in_common / len(results[i + 1])
            f_score = 0 if (precision + rappel == 0) else 2 * (precision * rappel) / (precision + rappel)
            print(f"Query n°{i + 1} --> precision@10 : {precision:.2f}, rappel@10 : {rappel:.2f}, F-score@10 : {f_score:.2f}")
            average_precision += precision
            average_rappel += rappel
            average_f_score += f_score
    average_precision /= (len(queries) - nb_invalid_results)
    average_rappel /= (len(queries) - nb_invalid_results)
    average_f_score /= (len(queries) - nb_invalid_results)
    print(f"\nAvg precision@10 : {average_precision:.2f}")
    print(f"Avg rappel@10 : {average_rappel:.2f}")
    print(f"Avg F-score@10 : {average_f_score:.2f}")
    print(f"number of queries with unknown results : {nb_invalid_results}")

