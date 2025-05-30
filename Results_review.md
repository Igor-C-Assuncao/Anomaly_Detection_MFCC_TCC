## Revisão dos Resultados

- **Anomalias detectadas:** 70 em 610  
  O modelo classificou 70 ciclos como anômalos dentre 610 ciclos de teste.

- **Aviso: Ajustando tamanho dos labels (509) e dos resultados (610) para 509**  
  Isso indica que havia 509 rótulos (labels) e 610 resultados (scores). O código ajustou ambos para 509, descartando 101 ciclos do final dos resultados. Isso pode indicar:
  - Alguma inconsistência na geração dos ciclos ou labels.
  - Possível perda de ciclos ou erro na concatenação dos labels.
  - **Atenção:** O ideal é investigar e corrigir a origem dessa diferença para garantir que cada ciclo tenha seu label correspondente.

- **ROC-AUC: 0.7391**  
  O modelo tem uma boa capacidade de separar normais de anômalos (quanto mais próximo de 1, melhor).

- **Acurácia: 0.3320**  
  Baixa acurácia. Isso pode ocorrer se a maioria dos ciclos for normal e o modelo estiver classificando poucos como anômalos.

- **Precisão: 1.0000**  
  Todas as predições de anomalia realmente eram anomalias (sem falsos positivos).

- **Recall: 0.1646**  
  O modelo detectou apenas cerca de 16% das anomalias reais (muitos falsos negativos).

- **F1-score: 0.2827**  
  Baixo, pois o recall é baixo.

### Interpretação

- O modelo está **muito conservador**: só classifica como anomalia quando tem muita certeza, por isso a precisão é 1.0, mas o recall é baixo.
- O limiar pode estar alto demais, ou o modelo pode estar subajustado.
- O ajuste automático do tamanho dos labels pode estar mascarando problemas de alinhamento entre labels e ciclos.

### Recomendações

1. **Verifique a geração dos labels e ciclos:**  
   Garanta que ambos tenham o mesmo tamanho e ordem desde o preprocessamento.

2. **Analise a distribuição dos erros de reconstrução:**  
   Veja se o limiar está adequado ou se precisa ser ajustado.

3. **Considere balancear melhor recall e precisão:**  
   Ajuste o limiar ou use outras estratégias para melhorar o recall sem perder muita precisão.

4. **Faça prints de debug:**  
   Imprima alguns exemplos de ciclos, labels e scores para garantir que estão alinhados.

---

## Possíveis causas para diferença de tamanho entre ciclos de teste/treino e resultados após treinamento

Se você tem, por exemplo:
- **Entrada de teste:** 244 ciclos (shape: (244, 20, 313))
- **Saída do modelo (test_losses):** 345 valores

Isso indica que o modelo está processando mais ciclos do que deveria. As causas mais comuns são:

### 1. **Flatten ou reshape errado**
No pipeline de avaliação, pode estar ocorrendo um flatten ou reshape que transforma cada ciclo em múltas amostras, ou está processando sub-janelas dentro de cada ciclo.

### 2. **Loop sobre sub-ciclos**
Se o código de avaliação faz um loop sobre cada ciclo e, dentro dele, processa subpartes (por exemplo, janelas menores), o número de saídas será maior que o número de entradas.

### 3. **Conversão de listas para arrays**
Se, ao transformar a lista de ciclos em tensores, você faz um reshape ou concatenação errada, pode acabar duplicando ou fragmentando os dados.

### 4. **Função de geração de ciclos**
Se a função que gera os ciclos para o teste está diferente da usada no preprocessamento, pode gerar mais (ou menos) ciclos do que o esperado.

### 5. **Uso de batch/janela no modelo**
Se o modelo espera batches ou janelas e você está passando os ciclos de forma diferente entre treino e teste, pode haver descompasso.

---

## Como diagnosticar e corrigir

1. **Imprima o shape dos dados em cada etapa**  
   Antes e depois de cada transformação (carregamento, conversão para tensor, entrada no modelo), faça:
   ```python
   print("Shape dos ciclos de teste:", np.array(test_cycles).shape)
   print("Shape dos tensores de teste:", test_tensors.shape)  # se aplicável
   print("Tamanho do resultado:", len(test_losses))
   ```

2. **Garanta que cada ciclo gera um único erro de reconstrução**  
   O loop de avaliação deve gerar um erro por ciclo, não por subjanela.

3. **Revise a função predict**  
   Certifique-se de que ela está processando cada ciclo individualmente e retornando um erro por ciclo.

4. **Verifique a função de geração de ciclos**  
   Confirme que a função usada no preprocessamento é a mesma usada para todos os folds e para treino/teste.

5. **Padronize o pipeline**  
   O número de ciclos de entrada deve ser igual ao número de saídas do modelo para que as métricas façam sentido.

---

**Resumo:**  
O problema é quase sempre causado por processamento extra (sub-janelas, flatten, reshape) ou inconsistência na geração dos ciclos. Imprima os shapes em cada etapa e ajuste o pipeline para garantir 1 ciclo de entrada = 1 erro de saída.
