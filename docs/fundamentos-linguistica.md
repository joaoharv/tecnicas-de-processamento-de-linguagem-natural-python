# Fundamentos da Linguística aplicados ao PLN

Documento de referência que aprofunda a [seção 2 do README](../README.md#2-fundamentos-da-linguística). Enquanto o README apresenta os níveis linguísticos de forma resumida, aqui cada nível é detalhado, com exemplos em português, aplicações computacionais e limitações, e conectado às operações de processamento de textos.

**Convenções.** São usadas as mesmas etiquetas do README: **[Notebook]** (presente no notebook de aula `FIAP_Processamento_de_Textos.ipynb`, material de curso não incluído neste repositório), **[Complementar]** (fundamento teórico adicional) e **[Futuro]** (previsto para aprofundamento). Os exemplos deste documento são conceituais: não há código executado nem resultados experimentais.

## Sumário

1. [Introdução](#1-introdução)
2. [Níveis de análise linguística](#2-níveis-de-análise-linguística)
3. [Relações entre os níveis](#3-relações-entre-os-níveis)
4. [Linguística e operações computacionais](#4-linguística-e-operações-computacionais)
5. [Linguística no PLN clássico e moderno](#5-linguística-no-pln-clássico-e-moderno)
6. [Exemplos práticos conceituais](#6-exemplos-práticos-conceituais)
7. [Síntese e conexões](#7-síntese-e-conexões)
8. [Referências](#8-referências)

---

## 1. Introdução

Todo sistema de PLN, mesmo o mais recente, assume respostas para perguntas que a Linguística estuda há décadas: o que é uma "palavra", o que torna uma frase gramatical, quando duas expressões significam o mesmo, e como o contexto altera o sentido. Conhecer essas respostas é útil por quatro razões:

- **Definir as unidades de trabalho.** Tokens, lemas e sintagmas são decisões de modelagem que têm origem em conceitos linguísticos.
- **Interpretar erros.** Saber se uma falha vem de ambiguidade lexical, de estrutura sintática ou de contexto pragmático orienta a correção.
- **Construir e avaliar dados.** Corpora anotados, métricas e guias de anotação são produtos de análises linguísticas.
- **Escolher a técnica adequada.** Tarefas que dependem de estrutura gramatical pedem ferramentas diferentes das que dependem de significado ou de intenção.

Para a Inteligência Artificial, a Linguística funciona menos como um conjunto de etapas obrigatórias e mais como o mapa do problema: indica quais fenômenos existem e, portanto, onde um modelo pode falhar.

---

## 2. Níveis de análise linguística

Cada nível abaixo é uma **perspectiva de análise** sobre o mesmo material linguístico, e não uma etapa fixa de um programa. Para a visão resumida e a tabela comparativa, ver o [README](../README.md#2-fundamentos-da-linguística).

### 2.1 Fonética

**[Complementar]** Não aparece no notebook.

- **Definição.** Estuda a produção (fonética articulatória), a propagação (acústica) e a percepção (auditiva) dos sons da fala. Os sons são descritos em notação própria, como o Alfabeto Fonético Internacional (IPA).
- **Exemplo em português.** A letra *x* representa sons diferentes: /ʃ/ em *xícara*, /ks/ em *táxi* e /z/ em *exame*. Em sentido inverso, o som /s/ pode ser escrito de formas distintas, como em *sapo*, *cesta* e *caça*.
- **Aplicação computacional.** Reconhecimento automático de fala (ASR), síntese de fala (TTS) e análise de sotaques.
- **Limitações.** O sinal acústico varia entre falantes, ruído e velocidade de fala. A grafia não codifica todos os detalhes da pronúncia; por isso, trabalhar apenas com texto escrito esconde esse nível.

### 2.2 Fonologia

**[Complementar]** Não aparece no notebook.

- **Definição.** Estuda como os sons se organizam em um sistema de **fonemas**, as unidades que distinguem significados, e quais regras governam suas combinações.
- **Exemplo em português.** *faca* e *vaca* formam um par mínimo: trocar /f/ por /v/ muda a palavra, o que mostra que são fonemas distintos. Já o *d* de *dia*, pronunciado [dʒ] em muitas variedades do português brasileiro, e o *d* de *data* são realizações (alofones) do mesmo fonema.
- **Aplicação computacional.** Conversão de grafema para fonema (G2P) em síntese de fala, modelos acústicos de reconhecimento de fala e correção ortográfica baseada em pronúncia.
- **Limitações.** As regras variam por dialeto. Em textos informais, grafias como *"naum"* e *"vc"* refletem a pronúncia, mas de forma não padronizada.

### 2.3 Morfologia

**[Complementar]** O notebook não trata de morfologia explicitamente; a tokenização é uma etapa anterior a ela.

- **Definição.** Estuda a estrutura interna das palavras, formada por **morfemas**: radical, afixos (prefixos e sufixos), vogais temáticas e desinências. Distingue-se:
  - **Flexão:** variações de uma mesma palavra, como gênero, número, tempo e pessoa (*gato, gata, gatos*).
  - **Derivação:** criação de novas palavras por afixação (*feliz → infeliz → infelizmente*).
  - **Composição:** junção de palavras (*guarda-chuva*).
- **Exemplo em português.** *cantávamos* decompõe-se em *cant-* (radical), *-a-* (vogal temática), *-va-* (desinência modo-temporal, pretérito imperfeito) e *-mos* (desinência número-pessoal, primeira pessoa do plural).
- **Aplicação computacional.** *Stemming* e lematização; análise morfológica para reduzir a variação do vocabulário; modelos de subpalavras que aproveitam regularidades de prefixos e sufixos.
- **Limitações.** O português tem flexão verbal rica, com formas irregulares (*fui* pode ser de *ser* ou de *ir*) e ambiguidade entre classes (*canto* como verbo ou substantivo). Procedimentos aproximados podem unir palavras de sentidos distintos ou deixar separadas formas relacionadas.

### 2.4 Léxico e análise lexical

**[Notebook]** (análise léxica e tokenização). O notebook trata a análise lexical como a identificação das unidades do texto, e não do significado delas.

- **Definição.** O léxico é o conjunto de itens lexicais de uma língua (os **lexemas**), com suas propriedades. Duas distinções ajudam a trabalhar com texto:
  - **Lexema e forma:** *cantar* é o lexema; *canto, cantas, cantávamos* são formas.
  - **Tipo (*type*) e ocorrência (*token*):** em *"o gato viu o rato"*, há 5 tokens e 4 tipos, já que *o* se repete.
- **Exemplo em português.** Nem toda unidade lexical é uma palavra gráfica: *pé de moleque* é uma expressão com significado próprio; *guarda-chuva* é uma palavra com hífen; *do* equivale a *de + o*.
- **Aplicação computacional.** Tokenização, normalização (caixa e acentuação), construção de vocabulário, identificação de expressões com várias palavras e contagem de frequências, como no BoW do notebook.
- **Limitações.** Definir "palavra" não é trivial: clíticos (*dá-me*, *vê-lo*), contrações, hifenização e abreviações geram decisões que afetam todo o restante do processamento. Itens que ocorrem uma única vez no corpus são frequentes e difíceis de modelar.

### 2.5 Sintaxe

**[Notebook]** (parsing e chunking).

- **Definição.** Estuda como as palavras se combinam em **sintagmas** e **orações**, e as relações entre eles. Duas representações são comuns:
  - **Constituência:** a frase é dividida em blocos aninhados (sintagma nominal, verbal, preposicional). É a base do *parsing* com gramáticas livres de contexto.
  - **Dependência:** a frase é descrita por relações entre palavras, como sujeito, objeto e modificador.
- **Exemplo em português.** Em *"O gato preto viu o rato"*:
  - Constituência: `[SN O gato preto] [SV viu [SN o rato]]`.
  - Dependência: *gato* é sujeito de *viu*; *rato* é objeto; *preto* modifica *gato*; *o* é determinante.
  - As abreviações SN, SV e SP (sintagma nominal, verbal e preposicional) correspondem às siglas NP, VP e PP usadas no notebook.
- **Aplicação computacional.** *Parsing* de constituintes ou de dependências, *chunking*, extração de relações do tipo sujeito-verbo-objeto, verificação gramatical.
- **Limitações.** A ambiguidade estrutural é comum (ver [seção 6](#6-exemplos-práticos-conceituais)). Gramáticas escritas à mão, como a do notebook, cobrem poucas construções. Textos informais, com erros ou fragmentados, fogem às regras formais.

### 2.6 Semântica

**[Notebook]** (semântica lexical, composicional e similaridade).

- **Definição.** Estuda o significado das expressões. Divide-se em:
  - **Semântica lexical:** relações entre palavras, como sinonímia (*cão*, *cachorro*), antonímia (*alto*, *baixo*), hiperonímia e hiponímia (*animal* e *cachorro*), polissemia (um item com sentidos relacionados) e homonímia (itens distintos com a mesma forma).
  - **Semântica composicional:** o significado da frase resulta do significado das partes e de sua estrutura sintática.
  - **Semântica distribucional** **[Complementar]**: palavras que ocorrem em contextos semelhantes tendem a ter significados semelhantes. É a base dos vetores de palavras usados em similaridade.
- **Exemplo em português.** *banco* (instituição financeira ou assento) é homonímia, e a escolha do sentido depende do contexto. *"Chutar o balde"* ilustra o limite da composicionalidade: o sentido da expressão não decorre das palavras isoladas.
- **Aplicação computacional.** Similaridade semântica, busca por significado, desambiguação de sentidos de palavras, agrupamento e classificação de documentos.
- **Limitações.** Vetores estáticos associam uma única representação a cada palavra, mesmo com vários sentidos. Modelos que ignoram a ordem das palavras, como BoW, podem aproximar frases de sentidos opostos, como em *"o filme foi bom"* e *"o filme não foi bom"*. Similaridade numérica não equivale a compreensão.

### 2.7 Pragmática

**[Complementar]** Não aparece no notebook.

- **Definição.** Estuda o significado em uso, dependente do contexto, do falante, do ouvinte e da situação. Inclui:
  - **Atos de fala:** dizer algo é também realizar uma ação (pedir, prometer, ordenar).
  - **Implicaturas:** o que é comunicado além do que é dito literalmente.
  - **Dêixis:** expressões cuja referência depende da situação (*eu*, *aqui*, *amanhã*).
  - **Ironia e polidez.**
- **Exemplo em português.** *"Você pode fechar a janela?"* tem forma de pergunta sobre capacidade, mas funciona como pedido. *"Que ótimo, choveu no dia do casamento"* expressa, em geral, o oposto do que as palavras literais dizem.
- **Aplicação computacional.** Análise de sentimentos com ironia, detecção de intenção em assistentes virtuais, sistemas de diálogo, moderação de conteúdo.
- **Limitações.** O contexto necessário muitas vezes não está no texto: conhecimento de mundo, relação entre os interlocutores, tom de voz. A concordância entre anotadores humanos também tende a ser menor nesses fenômenos, o que dificulta a criação de dados e a avaliação.

---

## 3. Relações entre os níveis

Os níveis não operam isoladamente: a interpretação de uma frase resulta de informação de vários níveis, e a falta de decisão em um deles é muitas vezes resolvida por outro. Considere a frase:

> **"Maria entregou o relatório ao diretor, que adorou o resultado."**

| Nível | O que se identifica na frase | Contribuição para a interpretação |
| --- | --- | --- |
| **Léxico** | *relatório*, *diretor*, *resultado* e as formas verbais *entregou* e *adorou* (lexemas *entregar* e *adorar*). | Define o vocabulário e as unidades da frase. |
| **Morfologia** | *entregou* = *entreg-* + *-ou* (terceira pessoa do singular, pretérito perfeito); *relatório* e *diretor* são substantivos masculinos no singular. | Informa tempo, pessoa e número, e, com isso, que a ação é passada e tem como agente *Maria*. |
| **Sintaxe** | `[SN Maria] [SV entregou [SN o relatório] [SP ao diretor], [oração relativa que adorou o resultado]]`. | Estabelece quem faz o quê e que o pronome relativo *que* retoma um termo anterior. |
| **Semântica** | *adorar* exige um sujeito capaz de sentir. | Entre *relatório* e *diretor*, apenas *diretor* é candidato a *adorar*; logo, *que* se refere ao diretor. |
| **Pragmática** | Se o falante sabe que o diretor nunca aprova nada e diz a frase com ênfase em *"é claro, adorou"*, a leitura pode ser irônica. | O mesmo conteúdo literal pode comunicar o oposto, conforme o contexto. |

Pontos a observar:

- A **sintaxe** isolada admite mais de uma ligação para *que*; foi a **semântica** que decidiu entre elas.
- A **pragmática** não altera a estrutura nem o significado literal, mas muda o que o falante quer comunicar.
- Se a frase fosse falada, **fonética e fonologia** (entonação, ênfase) forneceriam a pista de ironia, que no texto escrito se perde ou depende de pontuação.

Para o PLN, a consequência é que erros se propagam: uma segmentação lexical ruim prejudica a morfologia e a sintaxe, e uma estrutura sintática errada compromete a interpretação semântica.

---

## 4. Linguística e operações computacionais

Um **conceito linguístico** descreve um fenômeno da língua e existe independentemente de qualquer programa. Uma **operação computacional** é um procedimento que aproxima esse fenômeno em dados, com escolhas, regras ou modelos próprios, e pode errar. A mesma operação pode ser implementada por algoritmos diferentes.

| Operação | Conceito linguístico subjacente | O que a operação faz | O que ela não faz | No notebook |
| --- | --- | --- | --- | --- |
| **Tokenização** | Palavra e unidade lexical | Segmenta o texto em tokens | Não classifica nem interpreta; "token" não é sinônimo de "palavra" | **[Notebook]** |
| ***POS tagging*** | Classe gramatical (morfologia e sintaxe) | Atribui uma etiqueta a cada token, usando o contexto | Não constrói estrutura; depende do conjunto de etiquetas adotado | Tags fornecidas manualmente, usadas no chunking |
| ***Stemming*** | Radical, flexão e derivação | Corta afixos por regras, produzindo um radical aproximado | Não garante uma palavra real nem respeita a morfologia | Não (**[Complementar]**) |
| **Lematização** | Lexema (forma de dicionário) | Associa a forma flexionada ao lema, com dicionário e/ou análise morfológica | Não resolve sempre a ambiguidade (*fui*, *canto*) | Não (**[Complementar]**) |
| ***Parsing*** | Estrutura sintática | Gera uma árvore de constituintes ou de dependências, segundo uma gramática ou modelo | Não interpreta o significado; só cobre o que a gramática ou o modelo prevê | **[Notebook]** (CFG e `ChartParser`) |
| ***Chunking*** | Sintagma (nível superficial) | Delimita blocos planos, como NP e VP | Não captura hierarquia nem relações entre blocos | **[Notebook]** (`RegexpParser`) |
| **Similaridade semântica** | Relações de significado (semântica lexical e distribucional) | Quantifica a proximidade entre vetores | Não verifica verdade nem entende negação ou intenção | **[Notebook]** (spaCy) |

Alguns esclarecimentos:

- **Stemming x lematização.** O *stemming* é um procedimento de truncamento por regras (por exemplo, o algoritmo RSLP para o português); a lematização busca a forma de dicionário e, por isso, costuma exigir mais recursos linguísticos. A saída do *stemming* pode não ser uma palavra válida; a da lematização, em geral, é.
- **Parsing de constituintes x de dependências.** São formalismos distintos para descrever sintaxe. O notebook usa constituintes com uma gramática livre de contexto.
- **Chunking e IOB.** Em sistemas treináveis, o *chunking* costuma ser tratado como rotulagem de sequência, com marcações de início, interior e fora do bloco (formato IOB). No notebook, usa-se uma regra por expressão regular sobre as etiquetas.
- **Similaridade.** A medida (por exemplo, o cosseno) é uma operação matemática sobre vetores; "significado" é o fenômeno linguístico que se tenta aproximar com esses vetores.

---

## 5. Linguística no PLN clássico e moderno

A forma de usar informação linguística mudou ao longo do tempo, sem que a relevância dos fenômenos tenha desaparecido.

| Abordagem | Como a informação linguística é usada | Vantagens | Limitações |
| --- | --- | --- | --- |
| **Baseada em regras** | Conhecimento codificado de forma explícita por especialistas: gramáticas, léxicos, padrões. A CFG e o padrão de chunking do notebook são exemplos em pequena escala. | Transparente, controlável e útil com poucos dados. | Cobertura limitada, custo de manutenção e dificuldade com variação e ambiguidade. |
| **Estatística e Machine Learning clássico** | Atributos derivados de análises linguísticas (etiquetas, lemas, n-grams) e corpora anotados para treinar modelos. BoW, TF-IDF e n-grams do notebook se enquadram aqui. | Aprende regularidades a partir de dados; funciona bem em tarefas bem delimitadas. | Depende de engenharia de atributos e de dados anotados; ignora, em geral, contexto amplo e semântica. |
| **Redes neurais** | Representações vetoriais aprendidas (*embeddings*) e modelos de sequência ou de atenção, que aprendem padrões diretamente do texto. | Reduz a necessidade de atributos manuais; capta contexto e semelhança de uso. | Menos interpretável; exige muitos dados e capacidade computacional; pode refletir vieses dos dados. |

Quatro observações para evitar interpretações equivocadas:

- Modelos neurais modernos, incluindo Transformers e LLMs, **não executam obrigatoriamente** tokenização por palavras, *POS tagging*, *parsing* e análise semântica como etapas explícitas e separadas. Em geral, aprendem uma representação integrada a partir de grandes volumes de texto.
- Isso não significa ausência de decisões linguísticas: a **tokenização** (frequentemente por subpalavras) é uma etapa explícita; os dados, as tarefas e as métricas de avaliação são definidos com base em categorias linguísticas.
- Pesquisas de interpretabilidade investigam em que medida as representações aprendidas codificam informações como classe gramatical ou estrutura sintática. Os resultados dependem do modelo e da tarefa, e não autorizam concluir que o modelo "faz" uma análise linguística equivalente à humana.
- Abordagens se combinam: um sistema pode usar regras para validar a saída de um modelo estatístico, ou *embeddings* como atributos de um classificador clássico.

**[Futuro]** *Word embeddings*, Transformers e LLMs aparecem no [mapa de evolução do README](../README.md#6-mapa-de-evolução-dos-conhecimentos) e ainda não foram estudados ou implementados neste repositório.

---

## 6. Exemplos práticos conceituais

Exemplos textuais curtos, sem código e sem resultados de execução.

**Exemplo 1: do texto às unidades lexicais.**
*"Os gatos pretos dormiam no sofá."*

- Tokens: *Os, gatos, pretos, dormiam, no, sofá, .*
- Lemas possíveis: *o, gato, preto, dormir, em + o (contração), sofá*.
- Observação: *no* é uma contração de *em + o*. Se o objetivo for análise gramatical, pode ser interessante desfazê-la; para contagem de palavras, pode ser mantida.

**Exemplo 2: ambiguidade lexical.**

- *"Sentei no banco da praça."*
- *"Saquei dinheiro no banco."*

A mesma forma *banco* tem sentidos distintos; o contexto próximo (*sentei*, *praça*, *saquei*, *dinheiro*) é o que permite escolher. Um modelo que trate *banco* como um único item perde essa distinção.

**Exemplo 3: ambiguidade sintática.**
*"O policial prendeu o ladrão com a arma."*

- Leitura 1: o policial usou a arma para prender o ladrão (`[SP com a arma]` modifica o verbo).
- Leitura 2: o ladrão que tinha a arma foi preso (`[SP com a arma]` modifica *o ladrão*).

A mesma sequência de palavras admite duas árvores. Escolher entre elas exige conhecimento semântico ou de contexto.

**Exemplo 4: o limite de BoW e similaridade.**

- *"O filme foi bom."*
- *"O filme não foi bom."*

As duas frases compartilham quase todos os termos e diferem por um único (*não*). Uma representação por contagem trata a diferença como pequena, embora o sentido seja oposto. O exemplo mostra por que ordem, negação e composição importam, e motiva as representações contextuais do [mapa de evolução](../README.md#6-mapa-de-evolução-dos-conhecimentos).

**Exemplo 5: do literal à intenção.**
*"Você sabe que horas são?"*

- Em um ponto de ônibus, é um pedido de informação.
- Em uma reunião, dita a quem chegou atrasado, pode ser uma repreensão.

O conteúdo literal é o mesmo; a intenção comunicativa (pragmática) muda com a situação.

---

## 7. Síntese e conexões

| Nível | Problema abordado | Técnica computacional | Aplicações possíveis | Estado no repositório |
| --- | --- | --- | --- | --- |
| **Fonética** | Representar e reconhecer os sons da fala | Modelos acústicos, ASR, TTS | Transcrição, assistentes de voz, acessibilidade | Não abordado |
| **Fonologia** | Relacionar sons a unidades distintivas e à escrita | G2P, modelos fonológicos | Síntese de fala, correção por pronúncia | Não abordado |
| **Morfologia** | Reduzir e analisar a variação das formas de uma palavra | *Stemming*, lematização, análise morfológica | Busca, redução de vocabulário, análise de textos | **[Complementar]** (conceito); **[Futuro]** (prática) |
| **Léxico** | Definir e identificar as unidades do texto | Tokenização, normalização, vocabulário | Contagem de termos, indexação, pré-processamento | **[Notebook]** |
| **Sintaxe** | Descrever a estrutura das frases | *Parsing*, *chunking*, *POS tagging* | Extração de informação, verificação gramatical | **[Notebook]** |
| **Semântica** | Aproximar o significado de palavras e textos | Similaridade, vetores de palavras, BoW e TF-IDF como base | Busca por significado, agrupamento, classificação | **[Notebook]** (similaridade e representação); **[Futuro]** (embeddings, busca semântica) |
| **Pragmática** | Interpretar intenção e contexto | Modelagem de diálogo, detecção de ironia e de intenção | Assistentes virtuais, análise de sentimentos, moderação | **[Futuro]** |

As relações entre BoW, TF-IDF e n-grams e os níveis linguísticos são indiretas: essas técnicas operam sobre os **tokens** (nível lexical) e produzem representações numéricas usadas em tarefas semânticas, sem realizar, por si, análise sintática ou pragmática.

---

## 8. Referências

### Presentes no notebook **[Notebook]**

- Princípio da Composicionalidade (Frege), indicado no próprio notebook: <https://en.wikipedia.org/wiki/Principle_of_compositionality>
- Documentação das bibliotecas usadas: [NLTK](https://www.nltk.org/), [spaCy](https://spacy.io/) e [scikit-learn](https://scikit-learn.org/stable/).

### Complementares **[Complementar]**

**Obras gerais de PLN e Linguística Computacional**

- JURAFSKY, D.; MARTIN, J. H. *Speech and Language Processing*. Edição em rascunho: <https://web.stanford.edu/~jurafsky/slp3/>
- MANNING, C. D.; SCHÜTZE, H. *Foundations of Statistical Natural Language Processing*. MIT Press, 1999.
- BIRD, S.; KLEIN, E.; LOPER, E. *Natural Language Processing with Python*. O'Reilly, 2009. Disponível em: <https://www.nltk.org/book/>

**Fundamentos de pragmática e semântica**

- AUSTIN, J. L. *How to Do Things with Words*. Oxford University Press, 1962.
- GRICE, H. P. Logic and Conversation. In: COLE, P.; MORGAN, J. L. (Eds.). *Syntax and Semantics 3: Speech Acts*. Academic Press, 1975.
- HARRIS, Z. S. Distributional structure. *Word*, v. 10, n. 2-3, 1954.
- FIRTH, J. R. A synopsis of linguistic theory, 1930-1955. In: *Studies in Linguistic Analysis*. Blackwell, 1957.

**Recursos e algoritmos citados**

- MARCUS, M. P.; SANTORINI, B.; MARCINKIEWICZ, M. A. Building a large annotated corpus of English: the Penn Treebank. *Computational Linguistics*, v. 19, n. 2, 1993.
- *Universal Dependencies*, projeto de anotação multilíngue de dependências: <https://universaldependencies.org/>
- ORENGO, V. M.; HUYCK, C. A stemming algorithm for the Portuguese language. In: *Proceedings of the 8th International Symposium on String Processing and Information Retrieval (SPIRE)*, 2001.

**Pesquisa sobre representações em modelos neurais**

- TENNEY, I.; DAS, D.; PAVLICK, E. BERT rediscovers the classical NLP pipeline. In: *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2019. <https://arxiv.org/abs/1905.05950>
