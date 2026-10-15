# Representação numérica de textos

Documento técnico que aprofunda os métodos de representação textual apresentados no notebook de aula `FIAP_Processamento_de_Textos.ipynb` (Bag of Words, binarização, TF-IDF e n-grams; material de curso não incluído neste repositório). Ele dá sequência ao [README](../README.md), aos [fundamentos linguísticos](fundamentos-linguistica.md) e ao documento sobre [operações de processamento](processamento-textos.md), que tratam das etapas anteriores (segmentação, normalização e análise).

**Convenções.** As etiquetas são as do README: **[Notebook]**, **[Complementar]** e **[Futuro]**. Todos os números deste documento foram **calculados manualmente (de forma conceitual)** sobre um corpus pequeno e podem ser conferidos à mão; **não** são saídas de execução de código. As bibliotecas são citadas apenas como referência de comportamento.

**Validação por execução.** Os valores do Bag of Words e da representação binária (seções 2 a 4) foram posteriormente reproduzidos com `scikit-learn` em [`notebooks/01_bow_binarizacao.ipynb`](../notebooks/01_bow_binarizacao.ipynb), sem divergências. Os valores de TF-IDF e de n-grams (seções 5 e 6) **continuam sem validação por execução**.

## Sumário

1. [Introdução](#1-introdução)
2. [Corpus, documentos e vocabulário](#2-corpus-documentos-e-vocabulário)
3. [Bag of Words](#3-bag-of-words)
4. [Representação binária](#4-representação-binária)
5. [TF-IDF](#5-tf-idf)
6. [N-grams](#6-n-grams)
7. [Matrizes esparsas e dimensionalidade](#7-matrizes-esparsas-e-dimensionalidade)
8. [Comparação das representações](#8-comparação-das-representações)
9. [Relação com Machine Learning](#9-relação-com-machine-learning)
10. [Evolução para embeddings](#10-evolução-para-embeddings)
11. [Referências e síntese](#11-referências-e-síntese)

---

## 1. Introdução

Algoritmos de aprendizado operam sobre números. Para usar texto em classificação, agrupamento ou busca, é preciso convertê-lo em uma estrutura numérica, como indica o próprio notebook ao abrir a seção de representação. Existem três níveis de "numerização" que costumam ser confundidos:

| Nível | O que o número representa | Exemplo para *"gato"* | Serve como atributo de modelo? |
| --- | --- | --- | --- |
| **Codificação de caracteres** | Cada símbolo, por um código (Unicode, UTF-8). | `g` = 103, `a` = 97, `t` = 116, `o` = 111 | Não: representa a grafia, não a palavra nem o significado. |
| **Índice de token** | A posição da palavra em um vocabulário. | `gato` → 7 (valor arbitrário, definido pelo vocabulário) | Não diretamente: a ordem do índice é arbitrária, e *7* não é "maior" que *3* em nenhum sentido linguístico. |
| **Vetor de características** | O documento como um todo, com uma dimensão por característica (termo, n-gram). | Uma linha de uma matriz documento-termo | Sim: é a entrada típica de algoritmos de ML clássico. |

Este documento trata do terceiro nível. Os métodos clássicos (BoW, binário, TF-IDF e n-grams) têm em comum a ideia de **uma coluna por termo**; diferem no que colocam em cada célula.

---

## 2. Corpus, documentos e vocabulário

### 2.1 Conceitos

| Conceito | Definição |
| --- | --- |
| **Corpus** | Coleção de documentos sobre a qual se constrói a representação. |
| **Documento** | Cada unidade de texto do corpus (uma resenha, uma frase, uma notícia). É uma linha da matriz. |
| **Token** | Cada unidade produzida pela tokenização; uma ocorrência. |
| **Vocabulário** | Conjunto de tokens **distintos** (tipos) do corpus, depois de aplicadas as regras de tokenização e normalização. Define as colunas da matriz. |
| **Frequência absoluta** | Número de ocorrências de um termo: em um documento, `tf(t, d)`, ou no corpus inteiro. |
| **Frequência documental** | Número de **documentos** em que o termo aparece ao menos uma vez, `df(t)`. Independe de quantas vezes aparece em cada um. |
| **Matriz documento-termo** | Matriz com uma linha por documento e uma coluna por termo do vocabulário; cada célula contém um valor associado ao par (documento, termo). |

### 2.2 Corpus recorrente

O corpus abaixo tem três documentos curtos, escritos no estilo das resenhas do notebook (o notebook usa 10 resenhas; aqui foram reduzidas para que as contas sejam conferíveis à mão):

| Documento | Texto |
| --- | --- |
| **D1** | *O filme foi excelente.* |
| **D2** | *O filme foi bom, mas o final foi fraco.* |
| **D3** | *A fotografia é linda e o filme é bom.* |

> **Convenção de tokenização adotada.** Segue o comportamento padrão do `CountVectorizer`, usado no notebook, descrito no [README](../README.md#45-bag-of-words-binarização-e-tf-idf): minúsculas, pontuação descartada, **tokens com pelo menos 2 caracteres** e **sem remoção de *stopwords***. Por isso, *o*, *a*, *e* e *é* (um caractere) deixam de ser tokens. O vocabulário é ordenado em ordem alfabética, como em `get_feature_names_out`.

Tokens por documento:

| Documento | Tokens | Quantidade |
| --- | --- | --- |
| D1 | `filme, foi, excelente` | 3 |
| D2 | `filme, foi, bom, mas, final, foi, fraco` | 7 |
| D3 | `fotografia, linda, filme, bom` | 4 |

**Vocabulário** (9 termos): `bom`, `excelente`, `filme`, `final`, `foi`, `fotografia`, `fraco`, `linda`, `mas`.

| Termo | Frequência absoluta no corpus | Frequência documental (`df`) |
| --- | --- | --- |
| bom | 2 | 2 |
| excelente | 1 | 1 |
| filme | 3 | 3 |
| final | 1 | 1 |
| foi | 3 | 2 |
| fotografia | 1 | 1 |
| fraco | 1 | 1 |
| linda | 1 | 1 |
| mas | 1 | 1 |

Verificação: a soma das frequências absolutas é 14, igual ao total de tokens (3 + 7 + 4). O termo *foi* ilustra a diferença entre os dois conceitos: ocorre 3 vezes, mas em apenas 2 documentos.

---

## 3. Bag of Words

**[Notebook]** (`CountVectorizer`, matriz `df_bow_freq`).

### 3.1 Definição

O *Bag of Words* (saco de palavras) representa cada documento por um vetor com a **contagem** de cada termo do vocabulário. A ordem das palavras é descartada: documentos com as mesmas palavras e ordem diferente geram o mesmo vetor.

### 3.2 Construção

1. **Tokenizar** (e normalizar) todos os documentos.
2. **Construir o vocabulário** com os termos distintos de todo o corpus e fixar a ordem das colunas.
3. **Contar**, para cada documento, quantas vezes cada termo ocorre.
4. **Montar a matriz** documento-termo.

No scikit-learn, as etapas 2 a 4 correspondem a `fit_transform`, e os nomes das colunas são obtidos com `get_feature_names_out`.

### 3.3 Matriz (3 documentos × 9 termos)

| Doc | bom | excelente | filme | final | foi | fotografia | fraco | linda | mas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **D1** | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **D2** | 1 | 0 | 1 | 1 | **2** | 0 | 1 | 0 | 1 |
| **D3** | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |

Conferência: as somas das linhas (3, 7 e 4) coincidem com o número de tokens de cada documento. As somas das colunas (2, 1, 3, 1, 3, 1, 1, 1, 1) coincidem com a frequência absoluta da tabela anterior. No notebook, essa soma por coluna (`df_bow_freq.sum()`) é o que gera o ranking das palavras mais frequentes.

### 3.4 Vantagens e limitações

| Vantagens | Limitações |
| --- | --- |
| Simples, rápido e fácil de interpretar. | Ignora a ordem: *"o filme não foi bom"* e *"o filme foi bom não"* geram o mesmo vetor. |
| Linha de base forte em muitas tarefas de classificação. | Não captura significado: *bom* e *excelente* são colunas independentes. |
| Cada coluna corresponde a um termo identificável. | Palavras muito comuns dominam a contagem. |
| | Documentos longos têm contagens maiores, sem que isso signifique maior relevância. |
| | Matriz esparsa e de alta dimensionalidade (seção 7). |

---

## 4. Representação binária

**[Notebook]** (`CountVectorizer(binary=True)`, matriz `df_bow_bin`).

### 4.1 Frequência x presença

A representação binária registra apenas se o termo **ocorre** (1) ou **não ocorre** (0) no documento, descartando quantas vezes. É uma variante do BoW: a contagem é substituída por um indicador.

### 4.2 Comparação com o BoW (mesmo corpus)

Matriz binária:

| Doc | bom | excelente | filme | final | foi | fotografia | fraco | linda | mas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **D1** | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **D2** | 1 | 0 | 1 | 1 | **1** | 0 | 1 | 0 | 1 |
| **D3** | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |

A única célula diferente do BoW é (D2, *foi*): **2** na contagem e **1** na versão binária. Nos demais casos, os dois métodos coincidem, porque nenhum outro termo se repete dentro de um documento.

Duas consequências:

- **Soma por coluna.** Na matriz binária, a soma de cada coluna é a **frequência documental**: 2, 1, 3, 1, 2, 1, 1, 1, 1. No BoW, é a frequência absoluta. Para *foi*, a diferença é 3 contra 2. Isso é relevante no exercício do notebook que compara as 10 palavras mais frequentes de cada método: os rankings podem divergir.
- **Aplicações.** É útil quando a **presença** importa mais que a repetição (textos curtos, busca por palavras-chave) e em modelos probabilísticos discretos para eventos binários, como o notebook observa. **[Complementar]** Um exemplo é o Naive Bayes de Bernoulli. Perde a informação de ênfase ou de repetição.

---

## 5. TF-IDF

**[Notebook]** (`TfidfVectorizer`, matriz `df_bow_tfidf`). As fórmulas e a comparação com a implementação padrão são **[Complementar]**.

### 5.1 Componentes

| Componente | Significado | Pergunta que responde |
| --- | --- | --- |
| **TF** (*Term Frequency*) | Frequência do termo `t` no documento `d`. | O quanto este termo é característico **deste** documento? |
| **DF** (*Document Frequency*) | Número de documentos que contêm `t`. | Em quantos documentos o termo aparece? |
| **IDF** (*Inverse Document Frequency*) | Medida decrescente em DF: termos presentes em muitos documentos têm peso baixo. | O quanto este termo é **raro** no corpus? |

A ideia central é que um termo é importante em um documento quando é **frequente nele e raro no restante do corpus**.

### 5.2 Fórmula conceitual e convenção adotada

```text
tf-idf(t, d) = tf(t, d) × idf(t)
```

Convenção deste documento: **logaritmo natural (ln)**, `tf(t, d)` como **contagem bruta** e `N` o número de documentos (aqui, `N = 3`). Na formulação básica, a base do logaritmo muda apenas a escala dos valores; na do scikit-learn, o "+1" fora do logaritmo faz com que a base também altere as proporções entre os termos, e a implementação usa o logaritmo natural.

### 5.3 TF-IDF "de livro" x implementação padrão do scikit-learn

| Aspecto | Formulação básica (conceitual) | `TfidfVectorizer` com parâmetros padrão |
| --- | --- | --- |
| **IDF** | `idf(t) = ln(N / df(t))` | `idf(t) = ln((1 + N) / (1 + df(t))) + 1` |
| **Termos em todos os documentos** | `idf = 0`: o termo é anulado. | `idf = 1`: o termo mantém peso mínimo. |
| **Suavização** | Nenhuma: se `df = 0`, há divisão por zero. | Soma 1 a `N` e a `df`, como se houvesse um documento extra com todos os termos. |
| **Normalização** | Nenhuma. | Cada linha é normalizada pela norma L2. |
| **TF** | Varia (contagem, frequência relativa, logarítmica). | Contagem bruta (`sublinear_tf=False`). |

Existem outras variantes na literatura; por isso, sempre que um valor de TF-IDF for citado, é preciso indicar a convenção.

**Suavização do IDF.** O termo "+1" dentro do quociente evita divisão por zero e atenua o efeito de contagens muito pequenas; o "+1" fora do logaritmo impede que termos presentes em todos os documentos sejam anulados.

**Normalização L2.** Cada vetor-documento `v` é dividido por sua norma euclidiana, `v / ‖v‖₂`, de modo que todo documento fique com comprimento 1. Isso reduz o efeito do tamanho do documento e faz com que o produto escalar entre dois vetores seja igual ao cosseno entre eles.

### 5.4 Exemplo numérico (convenção do scikit-learn, `N = 3`)

**Passo 1: IDF** (a partir das frequências documentais da seção 2.2).

| Termo | `df` | `idf = ln(4 / (1 + df)) + 1` | Valor |
| --- | --- | --- | --- |
| filme | 3 | ln(4/4) + 1 = 0 + 1 | **1,0000** |
| bom, foi | 2 | ln(4/3) + 1 = 0,2877 + 1 | **1,2877** |
| demais termos (`excelente`, `final`, `fotografia`, `fraco`, `linda`, `mas`) | 1 | ln(4/2) + 1 = 0,6931 + 1 | **1,6931** |

**Passo 2: TF × IDF** (antes da normalização).

| Doc | bom | excelente | filme | final | foi | fotografia | fraco | linda | mas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **D1** | 0 | 1,6931 | 1,0000 | 0 | 1,2877 | 0 | 0 | 0 | 0 |
| **D2** | 1,2877 | 0 | 1,0000 | 1,6931 | **2,5754** | 0 | 1,6931 | 0 | 1,6931 |
| **D3** | 1,2877 | 0 | 1,0000 | 0 | 0 | 1,6931 | 0 | 1,6931 | 0 |

**Passo 3: normalização L2.** Exemplo com D1:

```text
‖D1‖₂ = √(1,6931² + 1,0000² + 1,2877²) = √5,5249 ≈ 2,3505
excelente: 1,6931 / 2,3505 ≈ 0,7203
filme:     1,0000 / 2,3505 ≈ 0,4254
foi:       1,2877 / 2,3505 ≈ 0,5478
```

Matriz final (arredondada em 4 casas):

| Doc | bom | excelente | filme | final | foi | fotografia | fraco | linda | mas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **D1** | 0 | 0,7203 | 0,4254 | 0 | 0,5478 | 0 | 0 | 0 | 0 |
| **D2** | 0,3044 | 0 | 0,2364 | 0,4003 | 0,6089 | 0 | 0,4003 | 0 | 0,4003 |
| **D3** | 0,4445 | 0 | 0,3452 | 0 | 0 | 0,5845 | 0 | 0,5845 | 0 |

Conferência: a soma dos quadrados de cada linha é 1 (D1: 0,7203² + 0,4254² + 0,5478² ≈ 1,0000).

### 5.5 Interpretação

- **"filme"** aparece em todos os documentos e recebe o menor IDF (1,0000), embora, com a suavização, não seja anulado. Na formulação básica (`ln(N/df) = ln(3/3) = 0`), seu peso seria zero.
- Em **D2**, *foi* tem o maior valor porque é o único termo que se repete (TF = 2), mesmo com IDF moderado. O TF-IDF pondera a frequência, e não a substitui.
- Termos raros (`excelente`, `fotografia`, `linda`) têm o maior IDF e, por isso, tendem a caracterizar o documento. Nos documentos curtos de exemplo, o efeito é visível; em corpora reais, depende da distribuição do vocabulário.
- Uma consequência da normalização: o mesmo termo tem valores diferentes em documentos distintos mesmo com TF e IDF iguais, pois depende do restante do vetor (por exemplo, *bom* vale 0,3044 em D2 e 0,4445 em D3).

### 5.6 Aplicações e limitações

- **Aplicações.** Classificação de documentos, recuperação de informação (ranqueamento por similaridade de cosseno entre consulta e documentos) e identificação de termos característicos.
- **Limitações.** Continua sem ordem e sem semântica (*bom* e *excelente* seguem sem relação); o IDF depende do corpus em que foi calculado, então muda quando o corpus muda; palavras fora do vocabulário não têm peso.

---

## 6. N-grams

**[Notebook]** (`nltk.ngrams`). A vetorização por n-grams com scikit-learn é **[Complementar]**.

### 6.1 Definição

Um *n-gram* é uma sequência contígua de *n* tokens: **unigrama** (n = 1), **bigrama** (n = 2) e **trigrama** (n = 3). Para uma sequência de `L` tokens, há `L − n + 1` n-grams.

Exemplo com *"o gato dorme no sofá"* (5 tokens):

| n | N-grams | Quantidade |
| --- | --- | --- |
| 1 | `o`, `gato`, `dorme`, `no`, `sofá` | 5 |
| 2 | `(o, gato)`, `(gato, dorme)`, `(dorme, no)`, `(no, sofá)` | 4 |
| 3 | `(o, gato, dorme)`, `(gato, dorme, no)`, `(dorme, no, sofá)` | 3 |

> Observação sobre o notebook: o texto explicativo lista três bigramas para esta frase (*"O gato"*, *"gato dorme"*, *"dorme no"*) e omite *"no sofá"*. Pelo cálculo acima, são quatro.

### 6.2 Preservação parcial da ordem

Os bigramas e trigramas preservam a ordem **localmente**: *"não gostei"* passa a ser uma unidade, o que ajuda a distinguir de *"gostei"*. A ordem global do documento continua perdida, e dependências distantes entre palavras não são capturadas.

### 6.3 Crescimento do vocabulário

Cada ordem *n* adiciona novas combinações. No corpus recorrente (convenção da seção 2.2):

| Intervalo de n-grams | Tamanho do vocabulário |
| --- | --- |
| apenas unigramas | 9 |
| apenas bigramas | 10 |
| unigramas e bigramas | 19 |
| de unigramas a trigramas | 27 |

O vocabulário cresce rapidamente, e a maioria dos n-grams ocorre em poucos documentos, o que torna a matriz ainda mais esparsa. É a limitação já indicada no notebook para trigramas.

### 6.4 Aplicações

- **Classificação de textos.** Bigramas e trigramas como atributos capturam expressões e negações locais.
- **Modelos estatísticos de linguagem.** Estimam a probabilidade de uma palavra dada as `n − 1` anteriores. Para um bigrama, por máxima verossimilhança: `P(wᵢ | wᵢ₋₁) = contagem(wᵢ₋₁ wᵢ) / contagem(wᵢ₋₁)`.

Exemplo, com o texto do exercício 3 do notebook, em minúsculas (*"o gato comeu a ração. o gato dormiu no sofá. o cachorro latiu para o gato."*, 19 tokens, incluindo os pontos):

- `(o, gato)` ocorre 3 vezes; `(o, cachorro)` ocorre 1 vez; *o* ocorre 4 vezes.
- `P(gato | o) = 3 / 4 = 0,75` e `P(cachorro | o) = 1 / 4 = 0,25`.

### 6.5 Geração com NLTK x vetorização com scikit-learn

São duas coisas diferentes: uma produz **sequências de tokens**; a outra produz **colunas de uma matriz**.

| Aspecto | `nltk.ngrams` | `CountVectorizer(ngram_range=...)` |
| --- | --- | --- |
| **Natureza** | Sequências de tokens (tuplas). | Características vetoriais (colunas de contagem). |
| **Entrada** | Uma lista de tokens **já produzida**. | Textos brutos; o próprio vetorizador tokeniza. |
| **Tokenização** | A definida pelo usuário (por exemplo, `word_tokenize`). | A padrão do vetorizador: minúsculas, pontuação descartada, tokens de 2 ou mais caracteres. |
| **Saída** | Gerador de tuplas, por documento. | Matriz documento-termo esparsa para o corpus. |
| **Contagem e vocabulário** | Não contabiliza nem constrói vocabulário. | Constrói o vocabulário e conta as ocorrências. |

Consequência prática: aplicados à mesma frase, os resultados diferem. Para *"o gato dorme no sofá"*, o NLTK (com `word_tokenize`) gera `(o, gato)`, enquanto o vetorizador padrão descarta o token *o* antes de formar os bigramas e produz *"gato dorme"*, *"dorme no"* e *"no sofá"*. Um efeito colateral aparece no D3 do corpus recorrente: ao descartar tokens de um caractere, o vetorizador forma o bigrama *"linda filme"*, embora *"e o"* apareça entre as duas palavras no texto original.

---

## 7. Matrizes esparsas e dimensionalidade

### 7.1 Matriz esparsa

Uma matriz é **esparsa** quando a imensa maioria das células é zero. Em representações de texto, isso ocorre porque cada documento usa apenas uma pequena fração do vocabulário. A **densidade** é a razão entre células não nulas e o total de células.

No corpus recorrente (3 × 9 = 27 células), há 13 valores não nulos, ou seja, densidade de cerca de 48%. Um corpus tão pequeno ainda não é esparso. Em corpora reais, a densidade costuma ser baixíssima.

### 7.2 Por que o vocabulário aumenta a dimensionalidade

Cada termo distinto é uma coluna. Logo, o número de dimensões é `|V|`, o tamanho do vocabulário, que:

- cresce com o tamanho do corpus, de forma sublinear mas contínua (lei de Heaps), pois palavras novas continuam surgindo;
- cresce com a variação de forma (flexão, grafia, erros), quando não há normalização;
- cresce muito com n-grams (seção 6.3).

### 7.3 Representação esparsa x densa

O termo "denso" tem dois sentidos distintos neste contexto:

| Sentido | O que significa | Onde aparece |
| --- | --- | --- |
| **Formato de armazenamento** | Guardar **todas** as células da matriz, inclusive as nulas (representação densa), ou apenas os valores não nulos e suas posições (representação esparsa). | `CountVectorizer` e `TfidfVectorizer` devolvem matrizes esparsas (formato do SciPy); `toarray()` as converte em densas. |
| **Tipo de vetor** | Vetores de baixa dimensão (dezenas a centenas) com a maioria dos valores diferentes de zero. | *Embeddings* (seção 10). Contrastam com vetores esparsos de alta dimensão como os do BoW. |

**[Notebook]** No notebook, `bow_freq.toarray()` converte a matriz esparsa em densa para criar o `DataFrame`. Isso é adequado para 10 resenhas, mas inviável em corpora grandes.

### 7.4 Consequências para armazenamento e processamento

Exemplo hipotético, com números ilustrativos: 10.000 documentos, 50.000 termos e, em média, 50 termos distintos por documento.

| Formato | Células ou valores armazenados | Memória aproximada |
| --- | --- | --- |
| Densa | 10.000 × 50.000 = 500 milhões | ≈ 4 GB (8 bytes por valor) |
| Esparsa (CSR) | ≈ 500 mil não nulos, mais índices | ≈ 6 MB (valores de 8 bytes e índices de 4 bytes) |

Além do armazenamento, há outras consequências:

- **Processamento.** Algoritmos que exploram a esparsidade ignoram os zeros e ficam muito mais rápidos.
- **Risco de sobreajuste.** Com muito mais colunas do que documentos, modelos tendem a memorizar o treino.
- **Medidas de distância** perdem poder de discriminação em dimensões muito altas.
- **Mitigação.** Normalização (minúsculas, lematização), descarte de termos muito raros ou muito comuns, limite de vocabulário e técnicas de redução de dimensionalidade. Todas são opções de projeto a serem avaliadas por tarefa.

---

## 8. Comparação das representações

| Aspecto | Bag of Words | Representação binária | TF-IDF | N-grams |
| --- | --- | --- | --- | --- |
| **Funcionamento** | Contagem de cada termo por documento. | Indicador de presença (1) ou ausência (0). | Contagem ponderada pela raridade do termo no corpus (com normalização). | Termos formados por *n* tokens consecutivos, que podem ser contados, binarizados ou ponderados. |
| **Informação preservada** | Frequência do termo no documento. | Presença do termo. | Frequência relativa à raridade no corpus. | Ordem local entre palavras. |
| **Informação perdida** | Ordem, contexto, significado. | Frequência, além de ordem e significado. | Ordem e significado; o valor depende do corpus. | Ordem global, semântica; dependências distantes. |
| **Limitações** | Alta dimensionalidade; palavras comuns dominam. | Insensível à ênfase e à repetição. | Mais difícil de interpretar; sensível ao corpus de ajuste. | Vocabulário cresce muito; matriz mais esparsa. |
| **Aplicações** | Linha de base de classificação; análise de vocabulário. | Modelos de eventos binários; textos curtos; presença de palavras-chave. | Recuperação de informação; classificação; termos característicos. | Detecção de expressões e negação local; modelos de linguagem. |
| **No notebook** | `CountVectorizer` | `CountVectorizer(binary=True)` | `TfidfVectorizer` | `nltk.ngrams` |

Observação: n-grams **não** são uma alternativa aos três primeiros métodos, e sim uma escolha ortogonal sobre **o que é uma coluna**. É possível ter BoW, binário ou TF-IDF de unigramas, de bigramas ou de ambos.

---

## 9. Relação com Machine Learning

**[Complementar]** O notebook produz as matrizes, mas não treina modelos. Os modelos citados abaixo são mencionados apenas em nível conceitual. **[Futuro]**

### 9.1 Como as matrizes alimentam os modelos

- **Classificação.** Cada linha é um exemplo, cada coluna um atributo e há um rótulo por documento (sentimento, tema, *spam*). Os exercícios do notebook com resenhas IMDb apontam nessa direção. A escolha entre contagem, presença e TF-IDF influencia a adequação ao algoritmo (por exemplo, contagens para o Naive Bayes multinomial, presença para o de Bernoulli).
- **Agrupamento.** Sem rótulos, documentos com vetores próximos são agrupados por tema.
- **Recuperação de informação.** A consulta é convertida pelo **mesmo** vetorizador e comparada com os documentos por similaridade (por exemplo, cosseno). Com vetores normalizados em L2, isso equivale ao produto escalar. Nas matrizes do exemplo, a similaridade de cosseno entre D1 e D2 é cerca de 0,43, e entre D1 e D3, cerca de 0,15 (calculadas manualmente, sobre o TF-IDF da seção 5.4).

### 9.2 Treino e teste

Para avaliar um modelo de forma honesta, os dados são separados em **treino** (usado para ajustar o modelo) e **teste** (usado apenas para medir o desempenho em dados não vistos).

### 9.3 Ajuste do vetorizador apenas no treino

O vetorizador também **aprende** com os dados: o vocabulário e, no TF-IDF, os valores de IDF. Esse aprendizado deve ocorrer **somente no conjunto de treino**.

| Etapa | Operação | Conceito |
| --- | --- | --- |
| Treino | `fit_transform` no treino | Define vocabulário e IDF, e transforma os documentos de treino. |
| Teste | `transform` no teste | Aplica o vetorizador **já ajustado**; termos novos são ignorados. |

### 9.4 Vazamento de dados

O **vazamento de dados** (*data leakage*) ocorre quando informação do conjunto de teste influencia o treinamento. Em texto, o caso típico é ajustar o vetorizador no corpus inteiro antes da separação: o vocabulário e o IDF passam a conter informação do teste, e o desempenho medido fica otimista. Pontos de atenção:

- Ajustar o vetorizador **depois** de separar os dados.
- Em validação cruzada, ajustá-lo dentro de cada partição (por exemplo, com um *pipeline*).
- Aplicar ao teste exatamente o mesmo pré-processamento aplicado ao treino.

---

## 10. Evolução para embeddings

**[Futuro]** Introdução conceitual; não faz parte do notebook, exceto pelo uso dos vetores do spaCy em `similarity()`. Ver o [mapa de evolução do README](../README.md#6-mapa-de-evolução-dos-conhecimentos).

- **Limitação de partida.** Nas representações clássicas, *bom* e *excelente* são colunas independentes: o vetor não registra que são próximos em significado.
- **Word embeddings.** Cada palavra é associada a um vetor aprendido a partir de grandes volumes de texto, de modo que palavras usadas em contextos semelhantes ficam próximas no espaço (hipótese distribucional, citada nos [fundamentos](fundamentos-linguistica.md#26-semântica)).
- **Vetores densos.** Têm poucas centenas de dimensões, quase todas diferentes de zero, em contraste com os vetores esparsos de dezenas de milhares de dimensões do BoW.
- **Representações contextuais.** Em modelos como os baseados em Transformers, o vetor de uma palavra depende da frase, e *banco* em *"banco da praça"* pode ser diferente de *banco* em *"banco central"*. Isso trata uma limitação dos vetores estáticos.
- **Busca semântica.** Consulta e documentos são convertidos em vetores e comparados por similaridade, recuperando textos por **significado**, e não apenas por palavras compartilhadas. É a evolução natural do uso de TF-IDF e cosseno da seção 9.

Estes temas estão planejados para documentos específicos.

---

## 11. Referências e síntese

### 11.1 Síntese

| Método | Em uma frase |
| --- | --- |
| **Bag of Words** | Conta quantas vezes cada termo aparece em cada documento, descartando a ordem. |
| **Binária** | Registra só a presença do termo, descartando a frequência. |
| **TF-IDF** | Pondera a contagem pela raridade do termo no corpus e normaliza o vetor; valores dependem da convenção adotada. |
| **N-grams** | Amplia o que é uma coluna, de termos isolados para sequências de tokens, preservando ordem local ao custo de muito mais dimensões. |

Pontos de atenção que atravessam os métodos: a representação depende das decisões de tokenização e normalização; as matrizes são esparsas e de alta dimensão; o vetorizador deve ser ajustado apenas no treino; e nenhuma dessas representações captura, por si, o significado.

### 11.2 Referências

- SPÄRCK JONES, K. A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, v. 28, n. 1, 1972.
- SALTON, G.; BUCKLEY, C. Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, v. 24, n. 5, 1988.
- MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. *Introduction to Information Retrieval*. Cambridge University Press, 2008. Capítulos sobre ponderação de termos e modelo de espaço vetorial, e sobre compressão de índices (leis de Zipf e de Heaps): <https://nlp.stanford.edu/IR-book/>
- JURAFSKY, D.; MARTIN, J. H. *Speech and Language Processing*. Capítulos sobre modelos de linguagem n-gram e semântica vetorial: <https://web.stanford.edu/~jurafsky/slp3/>
- PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, v. 12, 2011.
- Documentação do scikit-learn: [extração de *features* de texto](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) e [armadilhas comuns, incluindo vazamento de dados](https://scikit-learn.org/stable/common_pitfalls.html).
- Documentação de [matrizes esparsas do SciPy](https://docs.scipy.org/doc/scipy/reference/sparse.html) e do [NLTK](https://www.nltk.org/).
- MIKOLOV, T. et al. *Efficient Estimation of Word Representations in Vector Space*, 2013: <https://arxiv.org/abs/1301.3781>
- DEVLIN, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018: <https://arxiv.org/abs/1810.04805>
- REIMERS, N.; GUREVYCH, I. Sentence-BERT: sentence embeddings using Siamese BERT-networks. In: *Proceedings of EMNLP-IJCNLP*, 2019. <https://arxiv.org/abs/1908.10084>
