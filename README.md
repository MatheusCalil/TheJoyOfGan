# TheJoyOfGan

## Introdução

### Estrutura do projeto

### Motivação

### Prova de conceito: CIFAR-10

Foi utilizada a base [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (somente carros) para uma abordagem inicial. Essa base foi escolhida por ser conceituada no mundo de ciência de dados, possuir um conjunto suficientemente de exemplos de treino e vasta literatura.

![alt text](https://github.com/MatheusCalil/TheJoyOfGan/blob/master/Cifar/Unknown.jpg)

Figura - Imagens sintéticas e Imagens de carros CIFAR-10. Imagens de 0 a 9 são as sintetizadas pelo modelo.


## Generative Adversarial Network - GAN

Uma GAN é composta por duas redes neurais com objetivos opostos: Uma delas tenta entender o que é real e o que é sintético, enquanto a outra tenta sintetizar imagens de forma que tente ser real o suficiente para enganar a primera.

![alt text](https://miro.medium.com/max/2724/1*nAVqFluPijpBWR2tI4gCxg.png)

Figura - Exemplo de GAN para o classico dataset MNIST Fonte: https://medium.com/@naokishibuya/understanding-generative-adversarial-networks-4dafc963f2ef

Nas primeiras iterações, o generator normalmente irá gerar imagens de ruídos com uma certa distribição (normalmente uma distribuição normal) atráves de um processo de desconvolução. Conforme o generator aprende, essas imagens deixam de se tornar ruídos e se tornam mais "humanamente reconhecíveis".

Porém, conforme o generator cria imagens melhores, o discriminator se confunde mais e deixa de entender o que é uma imagem real de uma imagem sintética.

O treinamento de uma GAN consiste em balancear esses dois modelos. Para isso, foi proposto o seguinte pipeline:
- Separar um batch do dataset de imagens reais (label 1)
- Gerar um batch de imagens sintéticas (label 0)
- Treinar o gerador com as imagens reais e sintéticas por batchs
- Calcular a loss e atualizar os gradientes
- Repetir o processo até o fim da epoch
- Ao final da epoch, observamos a loss de cada um dos modelos. Caso fiquem inalterados por muitas epochs, o modelo pode ter deixado de aprender e uma modificação na arquitetura deve ser proposta.
- Também exibimos uma imagem criada pelo gerador ao final de cada epoca para observar alguma melhora qualitativa.

## Arquitetura da GAN

EM DESENVOLVIMENTO

## Limitações

Segundo [Theis et al., 2015](https://arxiv.org/abs/1511.01844), por ser desconhecida a distribuição das imagens geradas, não existe uma unica métrica quantitativa capaz de avaliar o modelo generator. Entendo que, por causa disso, a avalição da qualidade do generator é majoritariamente qualitativa e subjetiva.

A ausencia de uma GPU gera limitações na arquitura proposta, resultando também em uma queda de performance das imagens geradas.

Por ser um dataset gerado manualmente, temos uma quantidade limitada de dados, o que pode comprometer a performance.
