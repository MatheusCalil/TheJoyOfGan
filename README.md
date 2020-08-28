# TheJoyOfGan

# Introdução

### Sobre GANs e suas aplicações.

GANs foram inicialmente propostas por Ian Goodfellow em seu [artigo de 2014 entitulado Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Desde então, diversas arquiteturas e [aplicações](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/) foram propostas. Entre suas aplicações, a mais famosa talvez seja seu uso em [filtros em redes sociais](https://dimensionless.in/trending-story-faceapp-gans/).

Para Data Science, GANs podem ser utilizadas na prática para gerar dados sintéticos, remoção de ruídos em imagens e treinamento semi-supervisionado.

### Motivação

Decidi montar esse projeto com o unico proposito de aprendizado e diversão. O conjunto de dados utilizado são as pinturas do [lendário artista Bob Ross](https://www.youtube.com/user/BobRossInc), cujo programa The Joy of Painting inspirou o nome do projeto. Todos os dados foram obtidos no site com [todas as pinturas feitas por Bob Ross](https://www.twoinchbrush.com/all-paintings?page=14).
Bob Ross acreditava que qualquer um podia pintar como ele, então decidi descobrir se uma rede neural artificial também poderia.

### Estrutura do projeto
```md
Cifar -> Arquivos referentes aos testes realizados na base cifar-10
  Cifar10GAN.py - Código Python utilizado para prova de conceito de GANs utilizando 
  Figuras - Imagens geradas pelo modelo e exemplo de imagens do dataset
Dataset -> Contém o conjunto de dados, compactado, utilizado para treinamento da GAN
GAN Images V1 -> Resultado da primeira abordagem utilizando as pinturas de Bob Ross
Final
  Model - > contem os arquivos do generator utilizado para gerar as imagens em Samples
  Samples -> Amostras de imagens geradas pelo modelo (0 a 9) e imagens do dataset (10 a 19)
  TheJoyOfGAN.py -> Arquivo Python com o código de treinamento e vizualização
```

### Prova de conceito: CIFAR-10

Foi utilizada a base [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (somente carros) para uma abordagem inicial de teste de arquiteturas de GANs. Essa base foi escolhida por ser conceituada no mundo de ciência de dados, possuir um conjunto suficientemente grande de exemplos de treino e vasta literatura.

<p align="center">
  <img src="https://github.com/MatheusCalil/TheJoyOfGan/blob/master/Cifar/Unknown.jpg" />
</p>

<p align="center">
Figura - Imagens sintéticas e Imagens de carros CIFAR-10. Imagens de 0 a 9 são as sintetizadas pelo modelo.
</p>

[Esse medium](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9) foi utilizado como referência para a construção do modelo para CIFAR-10.

# Generative Adversarial Network - GAN

Uma GAN é composta por duas redes neurais com objetivos opostos: Uma delas tenta entender o que é real e o que é sintético, enquanto a outra tenta sintetizar imagens de forma que tente ser real o suficiente para enganar a primera.

![alt text](https://miro.medium.com/max/2724/1*nAVqFluPijpBWR2tI4gCxg.png)

<p align="center">
Figura - Exemplo de GAN para o classico dataset MNIST Fonte: https://medium.com/@naokishibuya/understanding-generative-adversarial-networks-4dafc963f2ef
</p>

Nas primeiras iterações, o generator normalmente irá gerar imagens de ruídos com uma certa distribição (normalmente uma distribuição normal) atráves de um processo de desconvolução. Conforme o generator aprende, essas imagens deixam de se tornar ruídos e se tornam mais "humanamente reconhecíveis".

Porém, conforme o generator cria imagens melhores, o discriminator se confunde mais e deixa de entender o que é uma imagem real de uma imagem sintética.

O treinamento de uma GAN consiste em balancear esses dois modelos. Para isso, foi proposto o seguinte pipeline:
- Separar um batch do dataset de imagens reais (label 1)
- Gerar um batch de imagens sintéticas (label 0)
- Treinar o discriminator com as imagens reais e sintéticas por batchs
- Calcular a loss e atualizar os gradientes. Para discriminator, atualizamos com a loss de imagens reais e sintéticas. Para generator, atualizamos com a loss de imagens sintéticas.
- Repetir o processo até o fim da epoch.
- Ao final da epoch, observamos a loss de cada um dos modelos. Caso fiquem inalterados por muitas epochs, o modelo pode ter deixado de aprender e uma modificação na arquitetura deve ser proposta.
- Também exibimos uma imagem criada pelo gerador ao final de cada epoca para observar alguma melhora qualitativa.

### Arquitetura Proposta de GAN

A arquitetura proposta foi inspirada na Spectral Normalization GAN (SN-GAN) desenvolvida por [Miyato et al., 2018](https://arxiv.org/abs/1802.05957)

<p align="center">
  <img src="https://github.com/MatheusCalil/TheJoyOfGan/blob/master/Final/SN-GAN.jpg" />
</p>
<p align="center">
Figura - Arquitetura de GAN proposta por Miyato et al.
</p>

### Alterações
<b>Discriminator:</b>
- Foram adicionador camadas de Dropout(0.25) ao final de cada bloco.
- Foram adicionados [BatchNormalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) dentro de cada convolução.
- Uma camada Densa de 1024 neurônios foi adicionada antes da camada Densa proposta na figura.

<b>Generator:</b>
- Sem alterações

### Anotações sobre arquitetura e treinamento
- Toda implementação foi realizada com [TensorFlow 2.0](https://www.tensorflow.org) e algumas etapas, como o cálculo da loss, foram retiradas da documentação.
- Não tive sucesso utilizando camadas Conv2DTranspose para desconvolução. Em vez disso, utilizei UpSampling e Conv2D.
- A adição da última camada densa representa a diferença entre as imagens da pasta "/GAN Images V1" e "/Final/Samples".
- Acompanhe a Loss do modelo, mas não espere estabilidade. As duas redes estão brigando entre si para se enganarem. Notei que no caso de sucesso, a loss da generator subiu consideravelmente após algumas epocas enquanto a do discriminator, caiu. [Mufti et al., 2019](https://arxiv.org/pdf/1903.06259.pdf) também reporta esse comportamento.

# Resultados

O treinamento ocorreu durante 200 epochs, utilizando batchs de 15 imagens. Os resultados principais estão representados nas próximas figuras.

<p align="center">
  <img src="https://github.com/MatheusCalil/TheJoyOfGan/blob/master/Final/training_process.jpg" />
</p>
<p align="center">
Figura - Diferentes imagens geradas por Epoch. Superior Esquerdo: Ruído gerado na epoch 0. Superior Direito: 50 epoch. Inferior Esquerdo: 150 epoch. Inferior Direito: 200 Epoch. Podemos observar o incremento nas imagens geradas conforme o modelo é treinado.
</p>

<p align="center">
  <img src="https://github.com/MatheusCalil/TheJoyOfGan/blob/master/Final/Generated_Samples.jpg" />
</p>
<p align="center">
Figura - De 0 a 9 imagens geradas pelo modelo, de 10 a 19 imagens do dataset.
</p>

<p align="center">
  <img src="https://github.com/MatheusCalil/TheJoyOfGan/blob/master/Final/Casos_similares.png" />
</p>
<p align="center">
Figura - Casos visualmente semelhantes Dataset versos Modelo.
</p>



### Trabalhos relacionados:

Utilizei os resultados de [Mufti et al., 2019](https://arxiv.org/pdf/1903.06259.pdf) como estado da arte.

<p align="center">
  <img src="https://github.com/MatheusCalil/TheJoyOfGan/blob/master/Final/Mufti.jpg" />
</p>
<p align="center">
Figura - Imagens geradas por Mufti et al., 2019
</p>

# Limitações

Segundo [Theis et al., 2015](https://arxiv.org/abs/1511.01844), por ser desconhecida a distribuição das imagens geradas, não existe uma unica métrica quantitativa capaz de avaliar o modelo generator. Entendo que, por causa disso, a avalição da qualidade do generator é majoritariamente qualitativa e subjetiva.

A ausencia de uma GPU gera limitações na arquitura proposta, resultando também em uma queda de performance das imagens geradas. O tempo de treino, por epoch, foi de aproximadamente 2 minutos. Cada teste de arquitetura me custava em torno de 50 epochs (1 hora e meia) e cada execução mais longa, de 200 epochs, demorava 6 horas para executar. Não foi observado uma melhora em imagens geradas por muito mais epochs depois de 200, observando um limite de 400 epochs.

Por ser um dataset gerado manualmente, temos uma quantidade limitada de dados, o que pode comprometer a performance.
