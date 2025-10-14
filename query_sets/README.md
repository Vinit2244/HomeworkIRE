# Search System Evaluation Query Sets

This directory contains query sets designed for evaluating the performance and functional correctness of our search system. The queries are used to generate metrics for latency (p95, p99), throughput, and relevance.

## Query Design Strategy

The query sets are not random. They were curated to be diverse and to probe specific properties of the search and indexing system. Each set is divided into six categories, ensuring a comprehensive evaluation of the system's capabilities.

### Query Categories

1. **High-Frequency Queries:** Simple, common terms to measure baseline system latency and raw lookup speed.
2. **Complex Phrase Queries:** Multi-word queries that mimic natural language to test the relevance ranking and scoring algorithms.
3. **Long-Tail Queries:** Specific queries with uncommon terms to evaluate the index's retrieval efficiency for long-tail data.
4. **Typo Tolerance Queries:** Queries with deliberate misspellings to test the system's typo-correction and fuzziness capabilities.
5. **Special Character Queries:** Queries with symbols, numbers, and special characters to test the robustness of the text analyzer and tokenizer.
6. **High-Load Queries:** High-frequency, non-stopword terms designed to stress-test the system's performance when handling a very large number of results.

### Queries

* `wikipedia_queries.json`: The evaluation query set tailored for the Wikipedia dataset.

```json
"queries": [
    { "query": "History", "docs": [] },
    { "query": "Science", "docs": [] },
    { "query": "Art", "docs": [] },
    { "query": "Music", "docs": [] },
    { "query": "Politics", "docs": [] },
    { "query": "Geography", "docs": [] },
    { "query": "Mathematics", "docs": [] },
    { "query": "Philosophy", "docs": [] },
    { "query": "Literature", "docs": [] },
    { "query": "Technology", "docs": [] },
    { "query": "What were the primary causes of World War I?", "docs": [] },
    { "query": "The role of the Silk Road in cultural exchange", "docs": [] },
    { "query": "Impact of the printing press on the Renaissance", "docs": [] },
    { "query": "Theory of general relativity explained", "docs": [] },
    { "query": "Key figures in the American Civil Rights Movement", "docs": [] },
    { "query": "How do vaccines stimulate the immune system?", "docs": [] },
    { "query": "The rise and fall of the Roman Empire", "docs": [] },
    { "query": "Major themes in Shakespeare's Hamlet", "docs": [] },
    { "query": "The process of photosynthesis in plants", "docs": [] },
    { "query": "Difference between nuclear fission and fusion", "docs": [] },
    { "query": "Who was the Byzantine emperor Justinian I?", "docs": [] },
    { "query": "The discovery of the coelacanth fish", "docs": [] },
    { "query": "What is the Magnus effect in physics?", "docs": [] },
    { "query": "The architectural style of Antoni Gaudí", "docs": [] },
    { "query": "Life of the mathematician Srinivasa Ramanujan", "docs": [] },
    { "query": "The history of the programming language LISP", "docs": [] },
    { "query": "What was the significance of the Battle of Thermopylae?", "docs": [] },
    { "query": "Explain the function of mitochondria", "docs": [] },
    { "query": "The epic of Gilgamesh summary", "docs": [] },
    { "query": "Who invented the Jacquard loom?", "docs": [] },
    { "query": "Albort Einstien", "docs": [] },
    { "query": "Michelanglo Buonaroti", "docs": [] },
    { "query": "Shakespeere plays", "docs": [] },
    { "query": "The pythagoren theorm", "docs": [] },
    { "query": "Achiles Greek mythology", "docs": [] },
    { "query": "C++ programming language", "docs": [] },
    { "query": "The works of H.P. Lovecraft", "docs": [] },
    { "query": "AT&T corporate history", "docs": [] },
    { "query": "The movie Blade Runner 2049", "docs": [] },
    { "query": "Solving a Rubik's Cube", "docs": [] },
    { "query": "What is TCP/IP?", "docs": [] },
    { "query": "The chemical formula for water H2O", "docs": [] },
    { "query": "The life of Marie-Curie", "docs": [] },
    { "query": "Search for user@example.com", "docs": [] },
    { "query": "The value of pi (π)", "docs": [] },
    { "query": "Human", "docs": [] },
    { "query": "World", "docs": [] },
    { "query": "System", "docs": [] },
    { "query": "Water", "docs": [] },
    { "query": "History", "docs": [] }
]
```

* `news_queries.json`: The evaluation query set tailored for the news dataset.

```json
"queries": [
    { "query": "Politics", "docs": [] },
    { "query": "Business", "docs": [] },
    { "query": "Technology", "docs": [] },
    { "query": "Sports", "docs": [] },
    { "query": "Health", "docs": [] },
    { "query": "Election", "docs": [] },
    { "query": "Market", "docs": [] },
    { "query": "Government", "docs": [] },
    { "query": "Breaking News", "docs": [] },
    { "query": "World News", "docs": [] },
    { "query": "US-China trade war impact on the technology sector", "docs": [] },
    { "query": "Global response to the 2022 climate change report", "docs": [] },
    { "query": "Federal Reserve interest rate hike announcement", "docs": [] },
    { "query": "The role of social media in the 2020 election", "docs": [] },
    { "query": "Supply chain disruptions in the automotive industry", "docs": [] },
    { "query": "Latest developments in the Russo-Ukrainian War", "docs": [] },
    { "query": "Economic forecast for the European Union", "docs": [] },
    { "query": "Breakthroughs in cancer treatment research in 2023", "docs": [] },
    { "query": "Stock market reaction to recent inflation data", "docs": [] },
    { "query": "UK government's post-Brexit economic policies", "docs": [] },
    { "query": "Who is the CEO of Palantir Technologies?", "docs": [] },
    { "query": "The eruption of the Cumbre Vieja volcano", "docs": [] },
    { "query": "What is the CHIPS and Science Act?", "docs": [] },
    { "query": "Results of the G20 summit in Bali", "docs": [] },
    { "query": "Details of the Artemis I mission launch", "docs": [] },
    { "query": "Who won the 2023 Nobel Prize in Physics?", "docs": [] },
    { "query": "Impact of the Ever Given Suez Canal obstruction", "docs": [] },
    { "query": "The collapse of Silicon Valley Bank", "docs": [] },
    { "query": "Vertex Pharmaceuticals cystic fibrosis drug trial", "docs": [] },
    { "query": "The AUKUS security pact details", "docs": [] },
    { "query": "Volodymyr Zelenskiy speech", "docs": [] },
    { "query": "Joe Biden's administation policy", "docs": [] },
    { "query": "Jerome Powel Fed chairman", "docs": [] },
    { "query": "Barak Obama's presidency", "docs": [] },
    { "query": "Consequences of Brexsit", "docs": [] },
    { "query": "COVID-19 pandemic response", "docs": [] },
    { "query": "Apple Inc. (AAPL) quarterly earnings", "docs": [] },
    { "query": "U.S. inflation rate year-over-year", "docs": [] },
    { "query": "G-7 summit discussions", "docs": [] },
    { "query": "Russia's S-400 missile system", "docs": [] },
    { "query": "The $1.2 trillion infrastructure bill", "docs": [] },
    { "query": "Meta's stock price drop (META)", "docs": [] },
    { "query": "Boeing 737 MAX investigation", "docs": [] },
    { "query": "The U.N. Security Council resolution", "docs": [] },
    { "query": "Was the trade deficit over 10%?", "docs": [] },
    { "query": "Report", "docs": [] },
    { "query": "Official", "docs": [] },
    { "query": "Government", "docs": [] },
    { "query": "Market", "docs": [] },
    { "query": "National", "docs": [] }
]
```
