# Processamento de Textos: operações e técnicas

Documento técnico que aprofunda as operações de processamento de textos apresentadas no notebook de aula `FIAP_Processamento_de_Textos.ipynb` (material de curso não incluído neste repositório). Ele complementa o [README](../README.md) (visão geral) e o documento de [fundamentos linguísticos](fundamentos-linguistica.md) (conceitos), explicando **como cada técnica funciona**, **que problema linguístico resolve** e **onde falha**.

**Convenções.** As etiquetas são as do README: **[Notebook]** (presente no notebook), **[Complementar]** (conteúdo adicional) e **[Futuro]** (previsto, não estudado). Os exemplos são conceituais, obtidos por raciocínio sobre as técnicas, e **não** correspondem a execuções de código. Representações vetoriais (BoW, TF-IDF e *embeddings*) serão tratadas em documentos específicos e aparecem aqui apenas como contexto.

## Sumário

1. [Introdução ao processamento computacional de textos](#1-introdução-ao-processamento-computacional-de-textos)
2. [Tokenização](#2-tokenização)
3. [Normalização e pré-processamento](#3-normalização-e-pré-processamento)
4. [POS tagging](#4-pos-tagging)
5. [Parsing](#5-parsing)
6. [Chunking](#6-chunking)
7. [Análise semântica](#7-análise-semântica)
8. [Comparação das técnicas](#8-comparação-das-técnicas)
9. [Conexão com Machine Learning e IA](#9-conexão-com-machine-learning-e-ia)
10. [Referências e próximos estudos](#10-referências-e-próximos-estudos)

---

## 1. Introdução ao processamento computacional de textos

Para um computador, uma frase passa por camadas de abstração sucessivas. Cada camada é uma **decisão de modelagem**, e erros em uma se propagam para as seguintes.

| Camada | O que é | Exemplo |
| --- | --- | --- |
| **Linguagem humana** | Sistema de comunicação com som, estrutura, significado e uso em contexto. | O que uma pessoa quer dizer ao falar ou escrever. |
| **Texto bruto** | Sequência de caracteres codificados (por exemplo, em Unicode), sem estrutura explícita. | `"Os alunos não gostaram das provas difíceis."` |
| **Tokens** | Unidades em que o texto é segmentado. | `Os`, `alunos`, `não`, `gostaram`, `das`, `provas`, `difíceis`, `.` |
| **Anotações linguísticas** | Informação acrescentada aos tokens ou à frase: lemas, classes gramaticais, sintagmas, árvores. | `gostaram` → verbo, lema *gostar* |
| **Representações computacionais** | Estruturas numéricas que algoritmos de aprendizado podem processar. | Vetores, matrizes (tema de documentos futuros). |

Este documento usa a frase **"Os alunos não gostaram das provas difíceis."** como exemplo recorrente. A sequência geral de etapas é didática e **não obrigatória**, como indicado na [seção 3 do README](../README.md#3-processamento-de-linguagem-natural).

---

## 2. Tokenização

Relação linguística: **léxico** (ver [fundamentos, seção 2.4](fundamentos-linguistica.md#24-léxico-e-análise-lexical)). A tokenização é uma operação; "palavra" é um conceito linguístico, e os dois não coincidem exatamente.

### 2.1 Tokenização por espaços **[Notebook]**

Divide o texto nos espaços em branco (em Python, `str.split()`).

- **Funcionamento.** Não há regras linguísticas: tudo entre espaços é um token.
- **Exemplo.** `"Os alunos não gostaram das provas difíceis."` produz `Os`, `alunos`, `não`, `gostaram`, `das`, `provas` e `difíceis.`, com o ponto colado à última palavra.
- **Limitações.** Pontuação, aspas e parênteses ficam colados às palavras; *"difíceis."* e *"difíceis"* passam a ser tokens distintos. Não serve para idiomas escritos sem espaços entre palavras.

### 2.2 Tokenização baseada em regras **[Notebook]**

Aplica regras (muitas vezes expressões regulares) que separam pontuação, tratam contrações, abreviações e outros padrões. No notebook, usa-se `nltk.word_tokenize`.

- **Funcionamento.** Separa o ponto de *difíceis*, a vírgula de *"Olá,"* etc., gerando tokens de pontuação.
- **Exemplo.** `"Olá, mundo!"` resulta em `Olá`, `,`, `mundo`, `!`.
- **Limitações.** As regras refletem convenções do tokenizador e do idioma. **[Complementar]** No NLTK, o parâmetro `language` seleciona o modelo de segmentação de **sentenças**, enquanto as regras de separação de palavras seguem convenções originalmente voltadas ao inglês (tratamento de apóstrofos e contrações como *don't*). Em português, casos como *dá-me*, *d'água* e *pé de moleque* exigem atenção e podem não ser tratados como se espera.

### 2.3 Palavras e pontuação

A decisão de **manter ou descartar** pontuação depende da tarefa. Em análise sintática, a pontuação ajuda a delimitar orações; em contagem de termos, costuma ser descartada; em análise de sentimentos, sinais como `!` e `?` podem ser informativos. Tokenizar é separar; filtrar é uma operação posterior (seção 3).

### 2.4 Tokenização clássica e por subpalavras **[Complementar]**

| Aspecto | Clássica (palavras) | Por subpalavras |
| --- | --- | --- |
| **Unidade** | Palavra ou sinal de pontuação. | Fragmentos de palavra, aprendidos a partir de um corpus. |
| **Vocabulário** | Cresce com o corpus; palavras novas ficam fora do vocabulário. | Tamanho fixo e controlado; palavras novas são decompostas em fragmentos conhecidos. |
| **Exemplo ilustrativo** | `infelizmente` | `in` + `feliz` + `mente` (a divisão real depende do vocabulário aprendido). |
| **Uso típico** | PLN clássico, BoW, TF-IDF. | Modelos neurais modernos, como Transformers. |

Algoritmos conhecidos incluem BPE, WordPiece e Unigram (este último implementado, por exemplo, na biblioteca SentencePiece). Fragmentos de subpalavras **não** correspondem necessariamente a morfemas: a divisão é estatística, e pode coincidir ou não com a morfologia linguística.

---

## 3. Normalização e pré-processamento

**[Complementar]** O notebook aplica minúsculas apenas em um exercício e, no uso de `CountVectorizer` e `TfidfVectorizer`, de forma implícita; as demais operações não são tratadas. Também são previstas no [mapa de evolução do README](../README.md#6-mapa-de-evolução-dos-conhecimentos).

### 3.1 Operações

| Operação | O que faz | Exemplo | Risco |
| --- | --- | --- | --- |
| **Conversão para minúsculas** | Unifica variantes de caixa. | `Banco` e `banco` viram o mesmo item. | Perde a distinção entre nomes próprios e comuns (*Marta* / *marta*), siglas e ênfase. |
| **Tratamento de pontuação** | Remove ou separa sinais. | `difíceis.` → `difíceis` | Pontuação pode carregar informação (ênfase, perguntas, limites de oração). |
| **Remoção de acentos** | Converte `é` em `e`. | `país` → `pais` | Pode fundir palavras distintas (*país* e *pais*) e gerar ambiguidade. |
| ***Stopwords*** | Remove palavras muito frequentes e de pouco conteúdo lexical (artigos, preposições). | `os`, `das` | Listas comuns incluem *não*; removê-la inverte o sentido de *"não gostaram"*. |
| ***Stemming*** | Reduz a forma a um radical por regras de truncamento. | `gostaram` → radical aproximado *gost-* | O radical pode não ser uma palavra; pode agrupar termos de significados distintos. |
| **Lematização** | Reduz a forma flexionada ao lema (forma de dicionário). | `gostaram` → *gostar*; `difíceis` → *difícil*; `das` → *de* + *a(s)* | Depende de recursos linguísticos; ambiguidades como *fui* (*ser* ou *ir*) exigem contexto. |

### 3.2 *Stemming* x lematização

- O ***stemming*** é rápido e simples, mas impreciso; opera sobre a **forma**. Exemplos de implementações para o português: RSLP e Snowball.
- A **lematização** opera sobre o **lexema**, é mais custosa e mais interpretável; usa dicionários e/ou análise morfológica, e em muitos casos o contexto (a classe gramatical).
- Ambos tentam reduzir a variação flexional, que no português é grande nos verbos, e se relacionam à **morfologia** (ver [fundamentos, seção 2.3](fundamentos-linguistica.md#23-morfologia)).

### 3.3 Operações opcionais x etapas obrigatórias

Não existe uma lista universal de etapas obrigatórias. O que é indispensável depende da técnica:

- **Quase sempre necessária:** alguma forma de segmentação do texto em unidades (tokenização, no sentido amplo), pois sem ela não há o que processar.
- **Opcionais e dependentes da tarefa:** minúsculas, remoção de pontuação, de acentos e de *stopwords*, *stemming* e lematização.
- **Normalização não é neutra:** cada operação descarta informação. Em busca e contagem de termos, pode reduzir ruído; em análise de sentimentos, análise sintática ou modelos neurais com subpalavras, pode prejudicar o resultado.
- **Consistência:** o que for aplicado aos dados de treinamento deve ser aplicado, da mesma forma, aos dados novos.

---

## 4. POS tagging

Relação linguística: **classes gramaticais**, na fronteira entre morfologia e sintaxe (ver [README, seção 2](../README.md#2-fundamentos-da-linguística)).

### 4.1 O que é

*Part-of-Speech tagging* é a atribuição automática de uma **classe gramatical** (substantivo, verbo, adjetivo etc.) a cada token, considerando o contexto. É uma tarefa de **classificação de sequências**: a mesma forma pode receber etiquetas distintas conforme a frase.

**Exemplo ambíguo.** A forma *a* é artigo em *"a casa"*, pronome em *"eu a vi"* e preposição em *"vou a Roma"*. Só o contexto distingue os três usos.

**Exemplo da frase recorrente** (etiquetas universais, de forma ilustrativa):

| Token | Os | alunos | não | gostaram | de + as (*das*) | provas | difíceis | . |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Etiqueta | DET | NOUN | ADV | VERB | ADP + DET | NOUN | ADJ | PUNCT |

### 4.2 Etiquetas

Um **conjunto de etiquetas** (*tagset*) é uma convenção; resultados de conjuntos diferentes não são diretamente comparáveis.

- **Universal POS (UPOS):** conjunto reduzido e multilíngue do projeto *Universal Dependencies* (`NOUN`, `VERB`, `ADJ`, `DET`...).
- **Penn Treebank:** conjunto detalhado para o inglês (`NN`, `NNS`, `VBZ`, `JJ`, `DT`...).

### 4.3 Como é feito

- **Baseado em regras:** léxico e regras contextuais.
- **Estatístico:** modelos probabilísticos de sequência treinados em corpora anotados.
- **Neural:** modelos treinados em corpora anotados, que usam o contexto da frase.

### 4.4 Tags manuais do notebook x modelos automáticos

**[Notebook]** No chunking, a frase `O gato preto viu o rato` é fornecida **já etiquetada à mão**, com `DET`, `NOUN`, `ADJ` e `VERB`. Não há um *tagger* em ação: o notebook ensina a regra de chunking, não a etiquetagem.

Um modelo automático, por sua vez, recebe apenas o texto e **infere** as etiquetas, podendo errar, sobretudo com ambiguidade, palavras desconhecidas e variação informal. Os exercícios de parsing e chunking pedem POS tagging de frases em inglês; **[Complementar]** nesse caso, o *tagger* padrão do NLTK é voltado ao inglês e produz etiquetas do Penn Treebank, o que explica a presença de etiquetas desse conjunto nos comentários do notebook. Os modelos do spaCy para o português usam etiquetas universais.

> Observação já registrada no [README](../README.md#pontos-do-notebook-que-merecem-revisão): o código do notebook usa etiquetas universais, mas a lista de comentários mistura conjuntos. Por isso, ao reutilizar a regra de chunking com um *tagger*, é preciso garantir que as etiquetas da regra sejam as mesmas que o *tagger* produz.

---

## 5. Parsing

Relação linguística: **sintaxe**, estrutura de constituintes (ver [fundamentos, seção 2.5](fundamentos-linguistica.md#25-sintaxe)). **[Notebook]**

### 5.1 Gramática Livre de Contexto (CFG)

Uma CFG é composta por:

- **Símbolos não terminais** (categorias, como `S`, `NP`, `VP`);
- **Símbolos terminais** (as palavras);
- **Regras de produção**, que dizem como um símbolo se expande em outros (`S -> NP VP`);
- **Símbolo inicial** (`S`).

A gramática do notebook é pequena e escrita à mão: `S -> NP VP`, `NP -> Det N | Det N PP`, `VP -> V NP | V NP PP`, `PP -> P NP`, com um léxico de poucas palavras (*o, um, gato, rato, tapete, viu, correu, no, com*).

### 5.2 Árvore sintática

A **árvore de derivação** (*parse tree*) mostra como a frase é construída a partir das regras. Para `"o gato viu o rato"`:

```text
(S
  (NP (Det o) (N gato))
  (VP (V viu) (NP (Det o) (N rato))))
```

Cada nó interno é uma categoria sintática; as folhas são as palavras. A árvore representa a **estrutura**, não o significado.

### 5.3 ChartParser

O `nltk.ChartParser` é um analisador que usa uma **tabela (*chart*)** para armazenar resultados parciais: sub-análises já realizadas são reaproveitadas em vez de recalculadas. Isso é uma forma de programação dinâmica, e torna a análise mais eficiente do que testar todas as combinações de regras. O analisador devolve **todas** as árvores que a gramática permite para a frase.

Duas consequências práticas:

- Se alguma palavra **não** estiver na gramática, o analisador não consegue processá-la (o NLTK sinaliza que a gramática não cobre a entrada).
- Se a frase for compatível com a gramática por mais de um caminho, serão devolvidas várias árvores.

### 5.4 Ambiguidade sintática

Uma mesma sequência de palavras pode admitir mais de uma estrutura. A gramática do notebook, inclusive, permite isso para frases como `"o gato viu o rato no tapete"`, pois a regra `NP -> Det N PP` e a regra `VP -> V NP PP` são ambas aplicáveis:

```text
Leitura A (o PP modifica o NP "o rato"):
(S (NP (Det o) (N gato))
   (VP (V viu) (NP (Det o) (N rato) (PP (P no) (NP (Det o) (N tapete))))))

Leitura B (o PP modifica o verbo):
(S (NP (Det o) (N gato))
   (VP (V viu) (NP (Det o) (N rato)) (PP (P no) (NP (Det o) (N tapete)))))
```

Na leitura A, o rato está no tapete; na B, o gato viu o rato enquanto estava no tapete. Essas leituras foram obtidas por análise da gramática, não por execução. O exemplo ilustra que a sintaxe, sozinha, não escolhe a interpretação: isso depende de semântica e contexto (ver [fundamentos, seção 6](fundamentos-linguistica.md#6-exemplos-práticos-conceituais)).

### 5.5 Limitações e abordagens modernas

- **Cobertura:** gramáticas manuais são pequenas; idioma natural exige milhares de regras e léxicos extensos.
- **Ambiguidade:** o número de árvores pode crescer muito com frases longas.
- **Concordância e restrições:** a CFG do notebook não verifica gênero, número ou regência.
- **[Complementar]** Analisadores estatísticos e neurais, treinados em *treebanks*, escolhem a árvore mais provável e tratam texto real; muitos produzem **dependências** em vez de constituintes. O spaCy, por exemplo, fornece análise de dependências, mas esse recurso não é usado no notebook.

---

## 6. Chunking

Relação linguística: **sintagmas**, em nível superficial. **[Notebook]**

### 6.1 *Shallow parsing*

O *chunking* identifica **blocos planos e sem sobreposição** (sintagmas nominais, verbais, preposicionais), sem construir a hierarquia completa. Em troca da profundidade, ganha velocidade e robustez a ruído.

| Característica | Parsing | Chunking |
| --- | --- | --- |
| Saída | Árvore hierárquica completa | Blocos planos |
| Relações entre blocos | Sim | Não |
| Custo | Maior | Menor |
| Tolerância a ruído | Menor | Maior |

### 6.2 RegexpParser

O `nltk.RegexpParser` aplica **expressões regulares sobre uma sequência de etiquetas**, não sobre o texto. A regra do notebook é:

```text
NP: {<DET>?<ADJ>*<NOUN>+}
```

Leitura: um determinante opcional, zero ou mais adjetivos e um ou mais substantivos formam um sintagma nominal (`NP`).

### 6.3 Identificação de sintagmas

Para a frase recorrente com etiquetas `DET NOUN ADV VERB ADP DET NOUN ADJ`, regras de chunking em cascata poderiam, conceitualmente, identificar:

```text
[NP Os alunos]  não  [VP gostaram]  [PP das [NP provas difíceis]]
```

Observações:

- Regras posteriores podem usar blocos produzidos por regras anteriores (por exemplo, um `PP` formado por preposição e `NP`).
- O resultado depende **inteiramente** da qualidade das etiquetas fornecidas.
- Os exercícios do notebook pedem padrões para `NP`, `PP` e `VP`, em frases em inglês.

### 6.4 Limitações de regras dependentes da ordem

A regra do notebook assume a ordem **adjetivo + substantivo**, típica do inglês (*the black cat*). No português, o adjetivo costuma vir **depois** do substantivo (*o gato preto*):

| Frase | Etiquetas | Resultado da regra `{<DET>?<ADJ>*<NOUN>+}` |
| --- | --- | --- |
| *the black cat* | `DET ADJ NOUN` | `NP` completo. |
| *o gato preto* | `DET NOUN ADJ` | `NP` apenas com *o gato*; *preto* fica fora do bloco. |

Uma regra adaptada, como `NP: {<DET>?<ADJ>*<NOUN>+<ADJ>*}`, cobre adjetivos nas duas posições. Isso mostra que **regras manuais são específicas de idioma, de *tagset* e de estilo**: coordenações (*"a casa grande e bonita"*), apostos e estruturas aninhadas exigem novas regras, e cada regra pode interagir com as demais.

---

## 7. Análise semântica

Relação linguística: **significado** (ver [fundamentos, seção 2.6](fundamentos-linguistica.md#26-semântica)). **[Notebook]**

### 7.1 Semântica lexical

Trata do significado das palavras e das relações entre elas: sinonímia (*cão*, *cachorro*), antonímia, hiperonímia (*animal* ⊃ *cachorro*), polissemia e homonímia.

- **Uso computacional.** Recursos lexicais organizados em rede, como WordNets, e vetores de palavras.
- **Limitação.** O sentido depende do contexto: *banco* tem mais de um sentido, e uma representação com um único registro por palavra não os distingue.

### 7.2 Semântica composicional

O significado de uma expressão complexa é determinado pelo significado das partes e pela estrutura sintática (Princípio da Composicionalidade, de Frege, citado no notebook). Por isso a semântica **depende** da sintaxe: *"o gato viu o rato"* e *"o rato viu o gato"* têm as mesmas palavras e significados diferentes, por causa da estrutura.

- **Limitação.** Expressões idiomáticas (*"chutar o balde"*) e usos figurados não são composicionais.

### 7.3 Similaridade textual

Quantifica o quanto duas unidades textuais (palavras, frases, documentos) se aproximam. Há dois tipos:

- **Superficial (léxica):** baseada em palavras compartilhadas.
- **Semântica:** baseada em representações que aproximam o significado, mesmo sem palavras em comum.

No notebook, `doc.similarity()` do spaCy (modelo `pt_core_news_md`) calcula a similaridade de cosseno entre vetores dos documentos. O notebook compara *"O cão latiu no quintal"* com *"O cachorro fez barulho no jardim"* (esperada alta) e com uma frase sobre programação (esperada baixa). O notebook não guarda as saídas.

### 7.4 Limitações da similaridade vetorial

- **Perda de estrutura.** Vetores de documento por média de vetores de palavras ignoram ordem e composição: *"o gato viu o rato"* e *"o rato viu o gato"* contêm as mesmas palavras e, por isso, tendem a receber vetores idênticos, embora tenham sentidos diferentes.
- **Negação e antonímia.** *"bom"* e *"ruim"* ocorrem em contextos semelhantes, e podem ter vetores próximos, mesmo com sentidos opostos.
- **Ambiguidade.** Um vetor único por palavra mistura sentidos.
- **Vocabulário.** Palavras fora do vocabulário do modelo não têm vetor útil.
- **Dependência do modelo.** O valor numérico não é uma escala absoluta; depende do modelo, do idioma e do domínio, e deve ser interpretado de forma comparativa.
- **Similaridade não é equivalência.** Alta similaridade não implica que as frases tenham o mesmo sentido ou o mesmo valor de verdade.

---

## 8. Comparação das técnicas

| Técnica | Entrada | Saída | Objetivo | Limitações | Biblioteca relacionada |
| --- | --- | --- | --- | --- | --- |
| **Tokenização por espaços** **[Notebook]** | Texto bruto | Lista de tokens | Segmentação simples | Pontuação colada; sem tratamento de contrações | Python (`str.split`) |
| **Tokenização por regras** **[Notebook]** | Texto bruto | Lista de tokens e pontuação | Segmentação em palavras e sinais | Regras específicas de idioma e de convenção | NLTK (`word_tokenize`) |
| **Normalização e *stopwords*** **[Complementar]** | Tokens | Tokens normalizados ou filtrados | Reduzir variação e ruído | Descarta informação; depende da tarefa | NLTK, Python padrão |
| ***Stemming*** **[Complementar]** | Tokens | Radicais aproximados | Reduzir variações flexionais | Imprecisão; pode agrupar termos distintos | NLTK (RSLP, Snowball) |
| **Lematização** **[Complementar]** | Tokens (e contexto) | Lemas | Reduzir ao lexema | Depende de recursos e de ambiguidade | spaCy |
| ***POS tagging*** **[Complementar]**; tags manuais no **[Notebook]** | Tokens | Token + etiqueta | Classificar tokens gramaticalmente | Ambiguidade; dependência do *tagset* e do idioma | spaCy, NLTK |
| ***Parsing*** **[Notebook]** | Tokens e gramática | Árvore sintática | Estrutura hierárquica completa | Cobertura da gramática; ambiguidade; custo | NLTK (`CFG`, `ChartParser`) |
| ***Chunking*** **[Notebook]** | Tokens com etiquetas | Blocos planos | Identificar sintagmas | Regras dependentes de ordem e de *tagset*; sem hierarquia | NLTK (`RegexpParser`) |
| **Similaridade semântica** **[Notebook]** | Dois textos e vetores de palavras | Valor numérico | Quantificar proximidade de significado | Perde estrutura; antonímia e negação; dependente do modelo | spaCy (`similarity`) |

---

## 9. Conexão com Machine Learning e IA

As técnicas deste documento produzem **unidades** e **anotações** que alimentam sistemas de aprendizado. Os três usos abaixo são apresentados em alto nível; as representações vetoriais serão detalhadas em documentos próprios. **[Futuro]**

### 9.1 Classificação de textos

Tarefas como análise de sentimentos, detecção de *spam* e classificação por tema transformam um texto em um conjunto de atributos e aplicam um classificador. As etapas deste documento definem **quais atributos existem**: tokens, lemas, classes gramaticais, sintagmas. Escolhas de pré-processamento, como remover *stopwords* ou manter a negação, influenciam diretamente o desempenho.

### 9.2 Recuperação de informação

Em sistemas de busca, o texto é analisado para criar um índice, e a consulta do usuário precisa passar pelo **mesmo** tratamento (tokenização, normalização, eventual lematização) para que termos equivalentes se encontrem. A similaridade semântica da seção 7 aponta a evolução para a busca por significado, e não apenas por correspondência de palavras.

### 9.3 Representações vetoriais

Os algoritmos de aprendizado operam sobre números. Por isso, depois de segmentar e anotar, o texto precisa ser convertido em vetores. Existem representações baseadas em contagem e ponderação e representações semânticas aprendidas, cada uma com vantagens e limitações próprias, e elas serão tratadas separadamente.

### 9.4 Clássico e moderno

Nas abordagens clássicas, as etapas acima são, em geral, explícitas e controladas pelo desenvolvedor. Em modelos neurais modernos, grande parte da análise é aprendida de forma integrada, e etapas como POS tagging e parsing não são necessariamente executadas separadamente; a tokenização, geralmente por subpalavras, continua sendo uma etapa explícita. Conhecer as técnicas clássicas segue útil para **interpretar erros, construir dados e escolher abordagens**.

---

## 10. Referências e próximos estudos

### Referências

- BIRD, S.; KLEIN, E.; LOPER, E. *Natural Language Processing with Python*. O'Reilly, 2009. Capítulos sobre processamento de texto bruto, etiquetagem, extração de informação (*chunking*) e análise de estrutura sintática: <https://www.nltk.org/book/>
- JURAFSKY, D.; MARTIN, J. H. *Speech and Language Processing*. Edição em rascunho: <https://web.stanford.edu/~jurafsky/slp3/>
- MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. *Introduction to Information Retrieval*. Cambridge University Press, 2008, em especial o capítulo sobre vocabulário de termos: <https://nlp.stanford.edu/IR-book/>
- PETROV, S.; DAS, D.; MCDONALD, R. A universal part-of-speech tagset. In: *Proceedings of LREC*, 2012. <https://arxiv.org/abs/1104.2086>
- *Universal Dependencies*: <https://universaldependencies.org/>
- ORENGO, V. M.; HUYCK, C. A stemming algorithm for the Portuguese language. In: *Proceedings of SPIRE*, 2001.
- SENNRICH, R.; HADDOW, B.; BIRCH, A. Neural machine translation of rare words with subword units. In: *Proceedings of ACL*, 2016. <https://arxiv.org/abs/1508.07909>
- KUDO, T.; RICHARDSON, J. SentencePiece: a simple and language independent subword tokenizer and detokenizer for neural text processing. In: *Proceedings of EMNLP (System Demonstrations)*, 2018. <https://arxiv.org/abs/1808.06226>
- Documentação: [NLTK](https://www.nltk.org/), [spaCy](https://spacy.io/usage/linguistic-features) e [scikit-learn](https://scikit-learn.org/stable/).

### Próximos estudos

- Implementação em Python das operações descritas, com exemplos resolvidos e saídas registradas (etapas seguintes do repositório).
- Comparação prática entre *stemming* e lematização para o português.
- *POS tagging* e análise de dependências com modelos estatísticos para o português.
- Avaliação de *taggers* e *parsers*: métricas e conjuntos de teste.
- Documentos específicos sobre BoW, TF-IDF e *embeddings*.
- Tokenização por subpalavras e sua relação com modelos Transformers.
