# multi-nlp-class-exploration
PA164 Task2

- Atributy: document-term matrix, word2vec, fasttext (původní váhy od FB, averaged), Universal sentence embeddings, and AlBERT (on the first 512 tokens since more won't fit into our available GPUs)
    - Přemýšlel jsem jak u klasifikace dlouhých dokumentů (wiki) nejlépe použít POS tagy? Word vectory jdou průměrovat se zachováním nějaké sémantické informace, ale jak je to nejlepší dělat pro POS, když je jedna predikce za celý dokument? Mohl bych dělat crop&padd, které by byly použitelné i po word vectory.
    - Informace z POS už by měla být obsažena ve word vectorech, jak ukázal pan Mikolov => náhrada za POS.
- Feature selection: žádnou / PCA / SVD
- Všechny techniky produkují (při průměrování word vectorů) konstantní matici z každého textu
- Klasifikátory na porovnání: SVM, Random forrest, MLP
- Přijdou mi dost různorodé, v prezentacích se píše o rozdílech mezi SVM, MLP a Bayes × Random forrest a rule induction

Cíle:
Rád bych všechno výše zmíněné zkusil pro klassifikaci všech úrovní labelů wiki datasetu -- chci si ověřit intuici, že atributy s nižší dimenzionalitou budou rychle ztrácet accuracy se zvýšením dimenzionality outputu. Pokud takovýto jev uvidíme, zajímá mě, jaká bude jeho míra a zda tam bude nějaký viditelný vztah (log, linear, exp?)
Když jsem tady tolik rentil o pomalosti Frameworku, bude hlavní challenge zpracovat celá wiki data (347 000 wiki dokumentů) za pár hodin i s DL učením (Framework jsem musel po ~20 dnech vypnout). Mám zkušenosti s podobně velkými datasety, mělo by to jít v 16 GB RAM i bez out-of-memory knihoven (Dask). Preprocessing už mám z Task 1, co přes gensim a pandas udělá preprocessing (steamming, etc + transformace do TF-IDF, indicator a freq) za pár sekund, tak i další kroky (kromě trénování DL metod) by měly jít udělat pod minutu/krok.

## The plan

1. generate attribute matrices, store in pyarrow
    - tf-idf (scikit)
    - average fasttext/word2vec embedding (gensim?)
    - UniversalSentenceEncoder (TF2)
2. apply dim reduction (scikit)
    - none
    - PCA (scikit)
    - Other?
3. train classificators on 9 matrices
    - SVM (scikit, slow :( single core)
    - Random forrest (scikit)
    - MLP (TF.Keras)
    
## General TODO
- Update logging to show time etc.
- Add progress, progress bar, ETA in a general manner for each of the Init/OD/CLF steps
    - Compute the ETA based on CLF type and dataset size (in cells? length?)
    - Max-time based CLF step: start with 2**10 train examples and train on 2× larger each time the training 
      stops in X seconds => at most 4× X seconds per CLF (given the algos are at most |N**2|)
