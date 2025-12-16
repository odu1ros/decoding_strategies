# LLM decoding strategies implementation

The objective is implementation of test generation algorithms and investigation of the impact of various hypermarameters on the quality and speed of LLM output.

### Project features:

* Implemented decoding strategies:
    * **Greedy Search**
    * **Sampling** (supports **Streaming** mode and **KV-Cache** optimization)
    * **Beam Search**
* Implemented following mechanisms:
    * **Temperature**
    * **Top-K** и **Top-P** filtration
    * **Repetition Penalty**
* Conducted series of experiments for generation quality comparison, as well as performance speed.

### Files

* **`decoding_strategies.py`** - the main module containing the `TextGenerator` class with implementations of `_sample_decoding`, `_beam_search_decoding`, and logit filtering functions
* **`experiments.ipynb`** - a notebook demonstrating the pipeline, iterating over parameters, and visualizing the results (time benchmarks and output comparison)

### Team

*   Nazarova
*   Astashkin
*   Semina
*   Serikova
