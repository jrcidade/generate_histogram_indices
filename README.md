Histograma serve para visualizar a distribuição dos valores de um raster (imagem ou camada contínua). Ele mostra, em forma de gráfico, quantos pixels existem em cada intervalo de valores.

Para que ele serve na prática:

Exploração dos dados: você entende rapidamente se os valores estão concentrados em uma faixa (por exemplo, NDVI entre 0.3 e 0.7) ou espalhados.

Definição de simbologia: ao criar estilos (ex.: mapa NDVI), o histograma ajuda a escolher os intervalos de classes e cores mais representativos.

Identificação de outliers: valores muito baixos ou altos aparecem como “pontas” isoladas no gráfico, indicando possíveis erros ou áreas específicas.

Ajuste de contraste: em imagens de satélite ou drones, o histograma mostra se a imagem está “lavada” ou muito escura; com isso, você pode aplicar estiramento de contraste.

Suporte para análises estatísticas: o histograma já fornece estatísticas básicas (mínimo, máximo, média, desvio-padrão) que ajudam na interpretação dos dados.

#### Dependências
pip install rasterio numpy matplotlib pandas
