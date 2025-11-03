# Query Processing

## Execution Modes

* Auto/Config: To be implemented
* Manual: A CLI-menu based program

## Taat/Daat

* If IndexInfo has a value of None of Boolean for a created index then it cannot be ranked so the code will print an error and exit the operation. Note that for ranking the index should be created with IndexInfo = Wordcount or Tfidf.
* To enable Boolean queries in Taat Daat my code follows a 2 stage solution - it first normally processed the boolean query to get a list of all the relevant documents that satisfies the boolean query and then this list of documents are passed into the ranking function that uses Taat/Daat algorithm to generate scores for the documents and rank them.

By default `PLOT_ES` variable in `performance_metrics.py` file is set to False because Elasticsearch is very slow in comparison to my custom method so the plots that were being created were very skewed, so to plot meaningful plots it is set to False by default, but if you wish to see latency and thresholding plots for elasticsearch as well you can put this variable to True.
