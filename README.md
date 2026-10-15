# Processamento de Textos: da Linguística ao PLN

Repositório educacional de estudos e experimentos sobre Ciência de Dados e Inteligência Artificial aplicadas ao processamento de textos. O ponto de partida é o notebook de aula `FIAP_Processamento_de_Textos.ipynb` (*Técnicas para Processamento de Textos*, Prof. Orlando Junior, FIAP), que percorre o caminho que vai da separação do texto em unidades até a sua representação numérica para modelos de Machine Learning.

O notebook de aula é material de curso e **não é redistribuído** neste repositório (ver [créditos](#créditos-e-direitos)). Aqui estão reunidos: notas de estudo e documentação técnica em `docs/` e um experimento autoral reproduzível em `notebooks/` (Bag of Words e binarização, com resultados validados; ver a [seção 4.7](#47-experimento-autoral-bag-of-words-e-binarização)). Para executá-lo, ver [Ambiente e execução](#ambiente-e-execução).

## Sumário

1. [Apresentação do projeto](#1-apresentação-do-projeto)
2. [Fundamentos da Linguística](#2-fundamentos-da-linguística)
3. [Processamento de Linguagem Natural](#3-processamento-de-linguagem-natural)
4. [Conteúdos estudados no notebook](#4-conteúdos-estudados-no-notebook)
5. [Bibliotecas e ferramentas](#5-bibliotecas-e-ferramentas)
6. [Mapa de evolução dos conhecimentos](#6-mapa-de-evolução-dos-conhecimentos)
7. [Organização e evolução do repositório](#7-organização-e-evolução-do-repositório)
8. [Referências e próximos estudos](#8-referências-e-próximos-estudos)

---

## 1. Apresentação do projeto

### Objetivo

- **Educacional:** consolidar os fundamentos de Linguística necessários para entender *por que* as técnicas de PLN existem e *o que* cada uma resolve.
- **Técnico:** documentar e, nas próximas etapas, implementar de forma organizada as técnicas clássicas de processamento de textos, servindo de base para tópicos mais avançados.
- **Profissional:** manter um repositório incremental, com documentação clara e código Python simples, adequado para portfólio.

### Linguística, PLN e Inteligência Artificial

A **Linguística** descreve como a linguagem humana funciona. O **Processamento de Linguagem Natural (PLN)** transforma essa descrição em procedimentos computacionais. A **Inteligência Artificial** (em especial o Machine Learning) fornece os métodos que aprendem esses padrões a partir de dados, em vez de depender apenas de regras escritas à mão.

```text
Linguística  ──►  PLN  ──►  Machine Learning / IA
(o que é a        (como tratar     (como aprender padrões
 linguagem)        texto no          a partir de grandes
                   computador)       volumes de texto)
```

### Como ler este documento

Cada conteúdo é identificado pela sua origem:

| Etiqueta | Significado |
| --- | --- |
| **[Notebook]** | Está presente no notebook de aula da FIAP (não incluído neste repositório), em texto e/ou código. |
| **[Complementar]** | Fundamento teórico adicionado aqui para dar contexto; não aparece no notebook. |
| **[Futuro]** | Tópico previsto para aprofundamento. Não foi implementado nem estudado ainda. |

**Origem x estado.** As etiquetas acima indicam *de onde vem* ou *como se enquadra* o assunto. O **estado** indica o quanto ele avançou no repositório, e é independente da origem:

| Estado | Significado |
| --- | --- |
| **Estudado** | O conceito foi apresentado e compreendido, em aula ou neste documento. |
| **Implementado** | Existe código funcional que o aplica. |
| **Validado** | A implementação teve sua execução e seus resultados verificados e registrados. |
| **Planejado** | Previsto para as próximas etapas; ainda não iniciado. |

Situação atual:

- Os assuntos do notebook de aula estão **estudados** e têm código de exemplo no próprio notebook, mas esse código **não foi validado** aqui, pois o notebook não guarda saídas de execução.
- **Bag of Words** e **binarização** estão **implementados** e **validados** no experimento autoral `notebooks/01_bow_binarizacao.ipynb` (ver [seção 4.7](#47-experimento-autoral-bag-of-words-e-binarização)).
- TF-IDF, n-grams, os exercícios e todos os tópicos `[Futuro]` estão **planejados**; TF-IDF e n-grams têm apenas cálculos conceituais em `docs/`.

> Os exercícios do notebook de aula estão enunciados, mas as células de resposta estão vazias. Portanto, este README **não** apresenta resultados de exercícios.

---

## 2. Fundamentos da Linguística

**[Complementar]**, com os pontos de contato com o notebook indicados. Os seis níveis abaixo seguem o quadro de Linguística do curso (Fonética, Fonologia, Morfologia, Sintática, Semântica e Pragmática), que também destaca a dimensão **léxica**.

| Nível | O que estuda | Exemplo simples | Relação com o processamento de textos | No notebook |
| --- | --- | --- | --- | --- |
| **Fonética** | Os sons da fala como fenômeno físico e articulatório. | A diferença entre o "t" de *tia* e o de *tio* em certos dialetos. | Base do reconhecimento e da síntese de fala. Em texto escrito, só aparece indiretamente (por exemplo, em correção ortográfica fonética). | Não |
| **Fonologia** | Como os sons se organizam em um sistema que distingue significados. | *pata* x *bata*: trocar /p/ por /b/ muda a palavra. | Modelagem de fala; em texto, relaciona-se a rimas e a grafias que refletem pronúncia. | Não |
| **Morfologia** | A estrutura interna das palavras (radical, prefixos, sufixos, flexões). | *cantávamos* = radical *cant-* + marcas de tempo e pessoa. | Fundamenta *stemming* e lematização, que reduzem variações de uma palavra à sua forma-base. | Não (a tokenização é anterior a ela) |
| **Léxico** (dimensão lexical) | O vocabulário de uma língua e as unidades que o compõem. | *gato*, *correr*, *mesa* como itens do vocabulário. | Origina o vocabulário de um corpus e a identificação de unidades (tokens). | Sim: análise léxica e tokenização |
| **Sintaxe** | Como as palavras se combinam em sintagmas e frases. | *O gato viu o rato* tem um sintagma nominal sujeito e um verbal com seu objeto. | Fundamenta *parsing*, *chunking* e a identificação de sujeito, verbo e objeto. | Sim: parsing e chunking |
| **Semântica** | O significado de palavras, frases e textos. | *cão* e *cachorro* são sinônimos. | Fundamenta similaridade textual, busca por significado e desambiguação. | Sim: semântica lexical, composicional e similaridade |
| **Pragmática** | O uso da linguagem em contexto: intenção, ironia, implicaturas. | "Está frio aqui" pode ser, na prática, um pedido para fechar a janela. | Análise de sentimentos com ironia, diálogo, detecção de intenção. É uma das áreas mais difíceis do PLN. | Não |

### Conceito linguístico x operação computacional

Um erro comum é tratar o nome de uma operação como se fosse o nível linguístico correspondente. A tabela separa os dois:

| Conceito linguístico | Operação computacional associada | Observação |
| --- | --- | --- |
| Léxico, palavra | Tokenização | Tokenizar é segmentar o texto em unidades; não interpreta nem classifica. |
| Morfologia | *Stemming*, lematização | Operações aproximadas; o *stemming* pode cortar o radical sem respeitar a morfologia real. |
| Classe gramatical (substantivo, verbo...) | *POS tagging* | Atribui um rótulo a cada token; ver o parágrafo abaixo. |
| Sintaxe | *Parsing*, *chunking* | Produzem estruturas (árvores ou blocos), não significado. |
| Semântica | Similaridade, vetores de palavras | Aproximam significado por geometria; não é "compreensão" no sentido humano. |

**Classes gramaticais, morfologia e sintaxe.** As classes gramaticais (substantivo, verbo, adjetivo...) ficam na fronteira entre morfologia e sintaxe: são definidas em parte por propriedades morfológicas (por exemplo, verbos flexionam em tempo e pessoa) e em parte pela função sintática e pela distribuição na frase. O *POS tagging* atribui uma classe a cada token usando pistas de ambos os níveis, pois a mesma forma pode ter classes distintas conforme o contexto (*"canto"* como verbo ou substantivo). Por isso o *POS tagging* costuma ser a ponte entre a tokenização e a análise sintática: o *chunking* e o *parsing* consomem essas etiquetas. No notebook, as *tags* são fornecidas manualmente, e não geradas por um *tagger* (ver as observações de revisão na [seção 4](#pontos-do-notebook-que-merecem-revisão), incluindo a mistura de conjuntos de *tags*).

Nota terminológica: no notebook, *análise léxica* é tratada como sinônimo de tokenização. Em linguística, o nível léxico trata do vocabulário; o uso computacional (herdado de compiladores) refere-se a segmentar a entrada em unidades.

---

## 3. Processamento de Linguagem Natural

**[Complementar]**

**O que é.** PLN é a área que permite a sistemas computacionais analisar, interpretar e gerar linguagem humana, escrita ou falada.

**Como o computador processa linguagem.** Computadores operam sobre números, não sobre palavras. Uma forma **didática** de organizar o processamento, típica do PLN clássico, é o fluxo abaixo:

1. **Segmentar** o texto em unidades (tokens).
2. **Anotar ou estruturar** essas unidades (classes gramaticais, sintagmas, árvores).
3. **Representar** o texto de forma numérica (vetores).
4. **Aplicar um modelo** (regras, estatística ou Machine Learning) para uma tarefa: classificar, buscar, resumir, traduzir.

Esse fluxo representa uma abordagem possível, não uma sequência obrigatória. Nem todo sistema executa explicitamente tokenização por palavras, *POS tagging*, *parsing* e vetorização como etapas independentes: abordagens recentes costumam usar tokenização por subpalavras e aprender a representação e a tarefa de forma integrada, dispensando etapas de anotação linguística explícita.

O notebook percorre exatamente essa ordem (segmentar, estruturar, extrair significado, representar numericamente).

**Principais desafios.**

- **Ambiguidade:** "manga" (fruta ou parte da roupa); "Vi o homem com o binóculo" (quem tem o binóculo?).
- **Dependência de contexto:** o significado muda conforme a situação (pragmática).
- **Variação linguística:** gírias, regionalismos, erros de grafia, abreviações.
- **Idioma:** boa parte das ferramentas e dos recursos é mais madura para o inglês; o português exige modelos e dados próprios.
- **Linguagem figurada:** ironia, metáfora, humor.

**Relação com Ciência de Dados e Machine Learning.** O texto é um dado não estruturado. O trabalho do cientista de dados é transformá-lo em uma representação tabular ou vetorial (como as matrizes BoW e TF-IDF do notebook), e então aplicar técnicas de aprendizado de máquina, como classificação ou agrupamento, e avaliar os resultados.

---

## 4. Conteúdos estudados no notebook

A ordem abaixo preserva a sequência pedagógica da aula. Os itens 4.1 a 4.6 são **[Notebook]**; os trechos marcados como **[Complementar]** acrescentam contexto. O item 4.7 descreve o experimento autoral deste repositório. Para o aprofundamento de cada tema, ver os documentos em [`docs/`](docs/).

### 4.1 Análise léxica e tokenização

- **Definição.** Etapa preliminar que quebra o texto bruto em unidades menores, os *tokens* (palavras, subpalavras ou pontuações). Apenas identifica quais são os elementos do texto.
- **Funcionamento.** O notebook compara duas abordagens:
  - *Tokenização ingênua:* `str.split()` separa apenas por espaços; a pontuação fica colada à palavra.
  - *Tokenização por regras linguísticas:* `nltk.word_tokenize(..., language="portuguese")` separa pontuação e trata contrações e apóstrofos.
- **Exemplo conceitual.** Para `"Olá, mundo! A tokenização começou."`, a abordagem ingênua mantém `"Olá,"` e `"mundo!"` como tokens únicos; o NLTK separa `"Olá"` e `","`, `"mundo"` e `"!"`.
- **Aplicações.** Primeira etapa de praticamente qualquer pipeline de PLN: contagem de palavras, busca, classificação.
- **Limitações.**
  - A tokenização ingênua não lida com pontuação, hifens ou contrações.
  - Mesmo com regras, o resultado depende do idioma e das convenções do tokenizador.
  - **[Complementar]** Tokens não são necessariamente palavras: modelos modernos usam subpalavras.
- **Exercícios propostos** (sem solução no notebook): `split()`, normalização para minúsculas, filtro de pontuação e frequência de tokens com NLTK.

### 4.2 Análise sintática, parsing e chunking

- **Definição.**
  - *Parsing:* analisa a sequência de palavras segundo uma gramática formal e produz uma **árvore de derivação** hierárquica completa.
  - *Chunking* (*shallow parsing*): identifica apenas blocos de nível inferior (como NP e VP), planos e sem sobreposição, sem construir a árvore completa.
- **Funcionamento.**
  - Parsing: uma Gramática Livre de Contexto (CFG) definida com `nltk.CFG.fromstring` e analisada por `nltk.ChartParser`.
  - Chunking: uma regra por expressão regular sobre *tags* de classe gramatical, com `nltk.RegexpParser` (padrão `NP: {<DET>?<ADJ>*<NOUN>+}`), aplicada a uma frase **já anotada manualmente**.
- **Exemplo conceitual.** Com a gramática do notebook, `"o gato viu o rato"` resulta em `(S (NP (Det o) (N gato)) (VP (V viu) (NP (Det o) (N rato))))`. No chunking, os blocos nominais `o gato` e `o rato` são isolados e o verbo fica fora de qualquer bloco.
- **Aplicações.** Extração de entidades e de sintagmas nominais, análise gramatical, apoio a tarefas semânticas.
- **Limitações.**
  - O parsing tem custo computacional maior e, com gramática escrita à mão, só cobre o vocabulário e as construções previstas nela (o `ChartParser` falha para palavras fora da gramática). **[Complementar]** Frases ambíguas geram mais de uma árvore.
  - O chunking é mais rápido e tolerante a ruído, mas não captura relações hierárquicas.
  - O padrão de chunking do notebook segue a ordem adjetivo + substantivo do inglês: em `"O gato preto"`, o adjetivo posposto fica fora do bloco `NP`.
- **Exercícios propostos** (sem solução no notebook): POS tagging, padrões NP, VP e PP, e extração de sintagmas, em frases em inglês.

### 4.3 Análise semântica e similaridade

- **Definição.** Etapa de extração do significado de palavras, frases e textos. O notebook distingue:
  - **Semântica lexical** (palavras): sinônimos, antônimos, hiperônimos e hipônimos.
  - **Semântica composicional** (frases): o significado de uma expressão complexa é determinado pelo significado de suas partes e pelas regras sintáticas que as combinam (Princípio da Composicionalidade de Frege).
- **Funcionamento.** A similaridade semântica quantifica a proximidade de significado entre duas unidades textuais. No notebook, usa-se o modelo `pt_core_news_md` do spaCy e `doc.similarity()`. **[Complementar]** Por padrão, a função calcula a similaridade de cosseno entre os vetores dos documentos (no spaCy, a média dos vetores das palavras). Matematicamente, o cosseno varia de -1 a 1; na prática, o valor obtido depende dos vetores disponíveis e do modelo utilizado, e não deve ser interpretado como uma escala absoluta de "grau de significado".
- **Exemplo conceitual.** *"O cão latiu no quintal"* e *"O cachorro fez barulho no jardim"* devem ter alta similaridade, apesar de quase não compartilharem palavras. Uma frase sobre programação deve ter similaridade baixa com ambas.
- **Aplicações.** Busca por significado, recuperação de informação, agrupamento e recomendação de documentos (o exercício propõe ranquear manchetes de notícias por relevância em relação a uma consulta).
- **Limitações.**
  - **[Complementar]** A similaridade do spaCy se baseia em vetores de palavras estáticos, e o resultado depende do modelo escolhido e do idioma.
  - **[Complementar]** Um vetor único por palavra não distingue sentidos diferentes de uma mesma palavra.
  - O notebook não guarda as saídas numéricas; os rótulos "alta" e "baixa" são comentários esperados, não resultados registrados.

### 4.4 Métodos de representação textual

- **Definição.** Conversão do texto em representação numérica que possa ser manipulada matematicamente, requisito para qualquer algoritmo de aprendizado.
- **Funcionamento.** O notebook cita, em nível conceitual, abordagens históricas: codificação de caracteres (ASCII, Unicode), indexação de palavras em um vocabulário, e vetores de contagem ou ponderados (BoW e TF-IDF). Elas se diferenciam pelo que o número representa:
  - **Codificação de caracteres:** cada símbolo recebe um código numérico (ASCII, Unicode). Representa a *grafia*, não o significado nem a palavra.
  - **Contagem de termos (BoW):** cada documento vira um vetor com a frequência de cada termo do vocabulário (seção 4.5).
  - **Ponderação por TF-IDF:** a contagem é ajustada pela raridade do termo no corpus, valorizando termos mais discriminativos (seção 4.5).
  - **Representações semânticas vetoriais** (*embeddings*): vetores densos aprendidos, em que proximidade indica similaridade de uso e significado. **[Futuro]** Não estudadas no notebook; ver [seção 6](#6-mapa-de-evolução-dos-conhecimentos).
- **Exemplo conceitual.** O documento *"O filme foi excelente"* vira uma linha de números, uma posição por termo do vocabulário.
- **Aplicações.** Classificação, tradução, sumarização e análise de sentimentos.
- **Limitações.** Perda de contexto e de semântica, alta dimensionalidade e incapacidade de capturar relações entre palavras. Essas limitações motivam os tópicos da [seção 6](#6-mapa-de-evolução-dos-conhecimentos).

### 4.5 Bag of Words, binarização e TF-IDF

Os três métodos são aplicados a um corpus de 10 resenhas curtas de filmes em português, usando `scikit-learn` e `pandas`.

#### Bag of Words (contagem)

- **Definição.** Representa cada documento por um vetor com a contagem de cada termo do vocabulário, ignorando a ordem.
- **Funcionamento.** `CountVectorizer().fit_transform(corpus)` gera uma matriz documento x termo, convertida em `DataFrame`; em seguida, somam-se as colunas para listar as 10 palavras mais frequentes.
- **Exemplo conceitual.** Se "filme" aparece 2 vezes em uma resenha, o valor da coluna "filme" nessa linha é 2.
- **Aplicações.** Classificação de documentos, análise exploratória de vocabulário.
- **Limitações.** Ignora ordem e contexto; palavras muito comuns dominam a contagem; a matriz é esparsa e cresce com o vocabulário.

#### Binarização

- **Definição.** Variante do BoW em que cada célula indica apenas a presença (1) ou ausência (0) do termo.
- **Funcionamento.** `CountVectorizer(binary=True)`.
- **Exemplo conceitual.** "filme" aparecendo 2 vezes ou 5 vezes resulta no mesmo valor 1.
- **Aplicações.** Modelos probabilísticos discretos que tratam eventos binários, segundo o próprio notebook.
- **Limitações.** Descarta a frequência, que pode ser informativa.

#### TF-IDF

- **Definição.** *Term Frequency-Inverse Document Frequency*: pondera a frequência do termo no documento pela raridade desse termo no corpus, refletindo sua importância relativa.
- **Funcionamento.** `TfidfVectorizer().fit_transform(corpus)`. **[Complementar]** Conceitualmente, `tf-idf(t, d) = tf(t, d) × idf(t)`; no scikit-learn, a implementação padrão aplica suavização no IDF e normalização L2 por documento.
- **Exemplo conceitual.** "filme" aparece em várias resenhas do corpus e tende a receber peso baixo; um termo que aparece em uma única resenha, como "caricatos", recebe peso alto nela.
- **Aplicações.** Classificação de documentos, recuperação de informação e modelagem de tópicos.
- **Limitações.** Continua sem capturar ordem ou semântica, e o vocabulário é fixado no corpus de treino.

> **[Complementar]** Por padrão, o `CountVectorizer` e o `TfidfVectorizer` convertem o texto para minúsculas e só consideram tokens com pelo menos dois caracteres. Isso significa que palavras de uma letra como "a", "o" e "é" são descartadas no corpus em português, sem que o notebook mencione isso.

### 4.6 N-grams

- **Definição.** Sequências contíguas de *n* tokens: unigramas (1), bigramas (2) e trigramas (3).
- **Funcionamento.** `nltk.word_tokenize` seguido de `nltk.ngrams(tokens, n)`.
- **Exemplo conceitual.** Em *"O gato dorme no sofá"*, os bigramas são `(O, gato)`, `(gato, dorme)`, `(dorme, no)` e `(no, sofá)`.
- **Aplicações.** Previsão da próxima palavra, análise de sentimentos e classificação de textos, conforme o notebook.
- **Limitações.** Capturam apenas contexto local; a dimensionalidade cresce rapidamente com *n*.
- **Exercícios propostos** (sem solução no notebook): bigramas, trigramas e distribuição de frequência de bigramas com a classe de frequência do NLTK.

### 4.7 Experimento autoral: Bag of Words e binarização

Origem: autoral deste repositório. Estado: **implementado** e **validado**.

O notebook [`notebooks/01_bow_binarizacao.ipynb`](notebooks/01_bow_binarizacao.ipynb) reproduz com `scikit-learn` (`CountVectorizer`) as matrizes que foram calculadas à mão em [`docs/representacao-textual.md`](docs/representacao-textual.md), sobre um corpus didático de três documentos:

| Documento | Texto |
| --- | --- |
| D1 | *O filme foi excelente.* |
| D2 | *O filme foi bom, mas o final foi fraco.* |
| D3 | *A fotografia é linda e o filme é bom.* |

**Resultados validados** (execução real, com os asserts do notebook aprovados):

- **Vocabulário:** 9 termos, em ordem alfabética: `bom, excelente, filme, final, foi, fotografia, fraco, linda, mas`.
- **Dimensão das matrizes:** 3 × 9, tanto no BoW quanto na versão binária.
- **Tokens por documento:** 3, 7 e 4, total de 14 ocorrências.
- **Células não nulas:** 13 em cada matriz.
- **Diferença entre BoW e binarização:** uma única célula, (D2, *foi*), com valor 2 no BoW e 1 na versão binária.
- **Frequência documental:** *filme* = 3; *bom* e *foi* = 2; os demais termos = 1.

Matriz BoW obtida:

| Doc | bom | excelente | filme | final | foi | fotografia | fraco | linda | mas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **D1** | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **D2** | 1 | 0 | 1 | 1 | 2 | 0 | 1 | 0 | 1 |
| **D3** | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |

Ambiente da validação: Python 3.12.10, pandas 2.3.3 e scikit-learn 1.9.0. "Validado" significa aqui que os valores calculados manualmente coincidem com os da biblioteca para este corpus; não se refere ao desempenho em nenhuma tarefa de Machine Learning.

### Pontos do notebook que merecem revisão

Observações feitas durante a análise, úteis para estudo e para uma eventual correção do material:

- **Células de n-grams:** o laço `for` apenas atribui `tokens` a cada documento, e os bigramas e trigramas são calculados depois, fora do laço. Como está escrito, só o **último** documento do corpus é efetivamente processado.
- **Comentário das *tags* de POS:** o código usa *tags* universais (`DET`, `NOUN`, `ADJ`, `VERB`), mas a lista de comentários mistura *tags* do Penn Treebank. Nela, `NN` aparece descrito como "pronome possessivo", quando na verdade designa substantivo comum no singular.
- **Exercício 3 de chunking:** a sentença é em inglês, mas o resultado esperado cita `estudou` e `estudou bastante`, em português.
- **Exercícios de representação:** pedem o conjunto de dados IMDb, cujo carregamento não é mostrado no notebook.

---

## 5. Bibliotecas e ferramentas

Identificadas por inspeção dos *imports* e comandos de instalação do notebook de aula. A coluna "Uso no notebook" refere-se a esse notebook; o experimento autoral usa apenas pandas e scikit-learn (ver [Ambiente e execução](#ambiente-e-execução)).

| Biblioteca | Finalidade geral | Uso no notebook |
| --- | --- | --- |
| **NLTK** | Biblioteca clássica de PLN, com tokenizadores, *taggers*, parsers e gramáticas. | `word_tokenize`, `CFG`, `ChartParser`, `RegexpParser`, `ngrams`. Requer os recursos `punkt` e `punkt_tab`. |
| **spaCy** | Biblioteca de PLN com modelos pré-treinados por idioma, voltada ao uso em produção. | Modelo `pt_core_news_md` e `doc.similarity()` para similaridade semântica. |
| **scikit-learn** | Biblioteca de Machine Learning em Python. | `CountVectorizer` (BoW e binarização) e `TfidfVectorizer`. |
| **pandas** | Manipulação de dados tabulares. | Estrutura matrizes BoW e TF-IDF em `DataFrame` para inspeção e ordenação. |

Nenhuma outra biblioteca de Machine Learning ou de *deep learning* é utilizada no notebook de aula.

### Ambiente e execução

O experimento autoral [`notebooks/01_bow_binarizacao.ipynb`](notebooks/01_bow_binarizacao.ipynb) depende apenas de `pandas` e `scikit-learn`. As dependências estão em [`requirements.txt`](requirements.txt), com as versões usadas na validação (Python 3.12.10). NLTK e spaCy **não** são necessários para ele.

Em um terminal, na raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/01_bow_binarizacao.ipynb
```

Para executar todas as células sem abrir a interface, gravando o resultado em um novo arquivo:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_bow_binarizacao.ipynb --output-dir saida_local
```

A execução termina sem erros quando todos os `assert` da seção 5 do notebook são aprovados e a mensagem "Todas as verificações passaram." é exibida. O diretório `saida_local/` não é versionado e pode ser removido.

---

## 6. Mapa de evolução dos conhecimentos

**[Futuro]**, com estado **Planejado**. Nenhum dos tópicos abaixo foi estudado a fundo ou implementado neste repositório. A coluna da direita indica de onde cada um se origina.

| Tópico | O que é | Origem a partir do conteúdo atual |
| --- | --- | --- |
| **Pré-processamento textual** | Normalização, remoção de *stopwords*, *stemming*, lematização. | Complementa a tokenização e melhora as matrizes BoW e TF-IDF. |
| **Machine Learning aplicado a textos** | Classificação, agrupamento e avaliação com métricas. | Usa BoW e TF-IDF como entrada; os exercícios com IMDb apontam nessa direção. |
| **Word Embeddings** | Vetores densos que codificam relações semânticas (Word2Vec, GloVe, fastText). | Supera limitações de BoW e TF-IDF. O notebook já usa vetores do spaCy em `similarity()`, mas sem estudá-los. |
| **Redes neurais para PLN** | Modelos recorrentes e convolucionais para sequências de texto. | Extensão natural dos n-grams, que modelam contexto local. |
| **Transformers** | Arquitetura baseada em atenção, que modela contexto em toda a sequência. | Resolve a limitação de contexto e de ambiguidade de palavra. |
| **Large Language Models (LLMs)** | Modelos de linguagem em larga escala, em geral baseados em Transformers. | Generalização da ideia de prever a próxima palavra, já presente nos n-grams. |
| **Busca semântica** | Recuperação por similaridade de significado, usando vetores. | Evolução da similaridade semântica e do TF-IDF. |
| **Retrieval-Augmented Generation (RAG)** | Combina busca de documentos com geração de texto por um LLM. | Depende de busca semântica e de LLMs. |

Os níveis linguísticos não cobertos pelo notebook (fonética, fonologia, morfologia e pragmática) também são candidatos a estudo futuro.

### Limitações atuais

- Há **um único experimento** executável (BoW e binarização); TF-IDF, n-grams, tokenização, parsing e similaridade semântica têm apenas documentação.
- O corpus do experimento tem **três documentos** e foi escolhido para permitir conferência manual; ele não representa o comportamento em corpora reais.
- Nenhum modelo de Machine Learning foi treinado ou avaliado.
- O notebook de aula não é distribuído aqui, e seus exercícios não estão resolvidos.
- As referências bibliográficas foram selecionadas de obras e documentações reconhecidas, mas os links externos devem ser verificados antes de citações formais.

---

## 7. Organização e evolução do repositório

### Estrutura atual

```text
.
├── README.md                          # documento central (este arquivo)
├── requirements.txt                   # dependências do experimento
├── .gitignore
├── docs/
│   ├── fundamentos-linguistica.md     # níveis linguísticos e conexão com o PLN
│   ├── processamento-textos.md        # tokenização, POS tagging, parsing, chunking, semântica
│   └── representacao-textual.md       # BoW, binária, TF-IDF, n-grams, esparsidade
└── notebooks/
    └── 01_bow_binarizacao.ipynb       # experimento: BoW e binarização (validado)
```

| Arquivo | Conteúdo |
| --- | --- |
| [`docs/fundamentos-linguistica.md`](docs/fundamentos-linguistica.md) | Níveis de análise linguística e relação com operações computacionais. |
| [`docs/processamento-textos.md`](docs/processamento-textos.md) | Funcionamento e limitações das técnicas de processamento. |
| [`docs/representacao-textual.md`](docs/representacao-textual.md) | Representações numéricas clássicas, com exemplos calculados à mão. |
| [`notebooks/01_bow_binarizacao.ipynb`](notebooks/01_bow_binarizacao.ipynb) | Reprodução do BoW e da binarização com scikit-learn. |

### Estrutura futura sugerida

Diretórios **sugeridos** para as próximas etapas; nenhum deles existe ainda.

```text
.
├── notebooks/                 # novos experimentos (TF-IDF, n-grams...), numerados em sequência
├── data/
│   ├── raw/                   # dados originais, sem alteração
│   └── processed/             # dados tratados
├── src/                       # código Python reutilizável (tokenização, vetorização...)
└── tests/                     # testes dos módulos em src/
```

Princípios de evolução:

- Cada etapa deve ser independente e documentada.
- Dados brutos nunca são sobrescritos.
- Código que se repete em notebooks migra para `src/`.
- Conteúdo teórico e planejado pode ser incluído no README, desde que esteja claramente identificado pela etiqueta de origem e pelo estado (ver [seção 1](#como-ler-este-documento)). Resultados e funcionalidades só devem ser descritos como existentes quando tiverem sido implementados e validados.

---

## 8. Referências e próximos estudos

### Documentação das ferramentas

- [NLTK](https://www.nltk.org/) e o livro *Natural Language Processing with Python* (Bird, Klein e Loper): <https://www.nltk.org/book/>
- [spaCy](https://spacy.io/)
- [scikit-learn: extração de *features* de texto](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

### Livros de referência

- JURAFSKY, D.; MARTIN, J. H. *Speech and Language Processing*. Edição em rascunho disponível em: <https://web.stanford.edu/~jurafsky/slp3/>
- MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. *Introduction to Information Retrieval*. Disponível em: <https://nlp.stanford.edu/IR-book/>

### Artigos para os tópicos de evolução

- MIKOLOV, T. et al. *Efficient Estimation of Word Representations in Vector Space*, 2013: <https://arxiv.org/abs/1301.3781>
- VASWANI, A. et al. *Attention Is All You Need*, 2017: <https://arxiv.org/abs/1706.03762>
- DEVLIN, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018: <https://arxiv.org/abs/1810.04805>
- LEWIS, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020: <https://arxiv.org/abs/2005.11401>

### Créditos e direitos

- **Material de aula:** o notebook `FIAP_Processamento_de_Textos.ipynb` (*Técnicas para Processamento de Textos*) é de autoria do Prof. Orlando Junior, FIAP. Ele é a base dos tópicos marcados como **[Notebook]**, mas **não é redistribuído** neste repositório. Os exercícios e o texto original pertencem aos seus autores.
- **Conteúdo autoral:** a documentação em `docs/` e o notebook `notebooks/01_bow_binarizacao.ipynb` são escritos para este repositório, com corpus e exemplos próprios, exceto onde indicado como referência ao material de aula.
- **Bibliotecas:** NLTK, spaCy, scikit-learn e pandas pertencem aos respectivos projetos e licenças.
- **Licença do repositório:** ainda não definida.

### Temas que merecem aprofundamento

- Completar e resolver os exercícios do notebook, registrando as saídas.
- Morfologia computacional: *stemming* e lematização para o português.
- POS tagging e parsing para o português com modelos estatísticos, em vez de gramáticas escritas à mão.
- Avaliação de modelos de texto (métricas, separação treino e teste).
- Limitações éticas e de viés em modelos treinados com texto.
