import Parsing
import Compute
import Prepocessing
import Helper
import time

documents = Parsing.parseDocuments("./CISI.ALL")
queries = Parsing.parseQueries("./CISI.QRY")
results = Parsing.parseResults("./CISI.REL")

print("\nProcessing documents ...")
start_time = time.time()
documents_processed = Prepocessing.process_texts(documents)
print("--> processed in %.2fs seconds" % (time.time() - start_time))

print("\nComputing the vocab ...")
start_time = time.time()
vocab, reverse_vocab = Compute.get_vocab(documents_processed)
print("--> processed in %.2fs seconds" % (time.time() - start_time))
print(f"--> size of the vocabulary : {len(vocab)}")

print("\nComputing TF/IDF for each term ...")
start_time = time.time()
tf_idf_term_scores, TF_ALL, IDF_ALL, dic_TF, dic_IDF = Compute.tf_idf_for_terms(documents_processed, vocab)
print("--> processed in %.2fs seconds" % (time.time() - start_time))

print("\nProcessing queries ...")
queries_processed = Prepocessing.process_queries(queries, vocab)

print("\nComputing TF/IDF scores for queries ...")
start_time = time.time()
tf_idf_query_scores = Compute.tf_idf_for_queries(tf_idf_term_scores, reverse_vocab, queries_processed)
print("--> processed in %.2fs seconds" % (time.time() - start_time))

print("\nComputing BM25 scores ...")
start_time = time.time()
bm25_scores = Compute.bm25(dic_TF, dic_IDF, documents_processed, queries_processed)
print("--> processed in %.2fs seconds" % (time.time() - start_time))

print("\nTesting scores for TF/IDF algorithm :")
Helper.print_queries_results(documents, documents_processed, queries, queries_processed, tf_idf_query_scores)

print("\nTesting scores for BM25 algorithm :")
Helper.print_queries_results(documents, documents_processed, queries, queries_processed, bm25_scores)

print("\nComputing metrics for the results for TF/IDF algorithm :")
Helper.print_metrics(queries, results, tf_idf_query_scores)

print("\nComputing metrics for the results for BM25 algorithm :")
Helper.print_metrics(queries, results, bm25_scores)
