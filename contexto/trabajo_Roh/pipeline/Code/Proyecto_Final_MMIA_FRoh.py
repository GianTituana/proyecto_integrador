# Cell 4
output_dir = '/home/froh/FinalProject'

!pip install lightning
!pip install nlpaug
!pip install -U sentence-transformers

# Para instalar EasyOCR
!pip install easyocr

# Importaciones de Procesamiento y cálculo
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from matplotlib.cm import viridis
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_rel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve, RepeatedKFold, RepeatedStratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import csv
import itertools
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import scipy.stats as stats
import seaborn as sns
import os

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification


import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from torchmetrics import AUROC, Accuracy


# Cell 6
# ruta correcta a archivo Excel
path = '/home/froh/dataset.xlsx'
df = pd.read_excel(path)

df.head()

# Cell 7
# Función para ver el tamaño y forma del dataset
def get_df_size(df, header='Dataset dimensions'):
  print(header,
        '\n# Attributes: ', df.shape[1],
        '\n# Entries: ', df.shape[0],'\n')

# Cell 8
print("Número de registros con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(df['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((df['final_pregunta_isAcusatoria'] == 1).sum()/df.shape[0]*100,2 ), "%")
print("Número de registros sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(df['final_pregunta_isAcusatoria'] == 0).sum(), " - ",  round((df['final_pregunta_isAcusatoria'] == 0).sum()/df.shape[0]*100,2 ), "%")

# Cell 9
# Contar las clases en la columna 'final_pregunta_isAcusatoria'
class_counts = df['final_pregunta_isAcusatoria'].value_counts()

# Crear un DataFrame para usar hue y aplicar la paleta de colores
class_counts_df = class_counts.reset_index()
class_counts_df.columns = ['class', 'count']

# Graficar un gráfico de columnas con etiquetas que cuente las clases
plt.figure(figsize=(4, 4))
ax = sns.barplot(x='class', y='count', data=class_counts_df, palette='viridis', hue='class', dodge=False, legend=False)

# Ajustar el rango del eje y para que llegue hasta 5500
ax.set_ylim(0, 5500)

# Añadir etiquetas a las barras
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 40, str(v) + " = " + str(round(v / 5005 * 100, 2)) + "%", ha='center', fontsize=12)

plt.title('Classes count')
plt.xlabel('Classes')
plt.ylabel('Count')

# Guardar el gráfico como SVG
plt.savefig('/home/froh/class_counts.svg', format='svg', bbox_inches='tight')

# Mostrar el gráfico
plt.show()

# Cell 11
# Dividir el DataFrame en conjuntos de entrenamiento y prueba de manera estratificada
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['final_pregunta_isAcusatoria'], random_state=72)

# Verificar las proporciones en los conjuntos de entrenamiento y prueba
print("Distribución en el conjunto de entrenamiento:")
print(train_df['final_pregunta_isAcusatoria'].value_counts(normalize=False),train_df['final_pregunta_isAcusatoria'].value_counts(normalize=True))
print("\nDistribución en el conjunto de prueba:")
print(test_df['final_pregunta_isAcusatoria'].value_counts(normalize=False),test_df['final_pregunta_isAcusatoria'].value_counts(normalize=True))

#Verifica directorio
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Guardar los conjuntos de entrenamiento y prueba en archivos CSV
train_df.to_csv(f'{output_dir}/train_dataset.csv', index=False)
test_df.to_csv(f'{output_dir}/test_dataset.csv', index=False)

# Cell 12
print("Número de registros totales con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(df['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((df['final_pregunta_isAcusatoria'] == 1).sum()/df.shape[0]*100,2 ), "%")
print("Número de registros totales sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(df['final_pregunta_isAcusatoria'] == 0).sum(), " - ",  round((df['final_pregunta_isAcusatoria'] == 0).sum()/df.shape[0]*100,2 ), "%")
print("Número de registros Train con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(train_df['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((train_df['final_pregunta_isAcusatoria'] == 1).sum()/train_df.shape[0]*100,2 ), "%")
print("Número de registros Train sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(train_df['final_pregunta_isAcusatoria'] == 0).sum(), " - ",  round((train_df['final_pregunta_isAcusatoria'] == 0).sum()/train_df.shape[0]*100,2 ), "%")
print("Número de registros Test con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(test_df['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((test_df['final_pregunta_isAcusatoria'] == 1).sum()/test_df.shape[0]*100,2 ), "%")
print("Número de registros Test sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(test_df['final_pregunta_isAcusatoria'] == 0).sum(), " - ",  round((test_df['final_pregunta_isAcusatoria'] == 0).sum()/test_df.shape[0]*100,2 ), "%")

# Cell 13
# Eliminar las columnas 'contract_id', 'pregunta_id' y 'sum_pregunta_isAcusatoria'
train_df = train_df.drop(columns=['contract_id', 'pregunta_id', 'sum_pregunta_isAcusatoria'])
test_df = test_df.drop(columns=['contract_id', 'pregunta_id', 'sum_pregunta_isAcusatoria'])

df_libre = train_df[train_df['final_pregunta_isAcusatoria'] == 0].copy()
df_acusa = train_df[train_df['final_pregunta_isAcusatoria'] == 1].copy()
df_acusa.head()

# Cell 16
import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Cargar los datos
df = df_acusa

# Definir los aumentadores
# Aumentador de sinónimos usando WordNet
synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='spa')
# Aumentadores de BERT en español para inserción y sustitución de palabras
contextual_word_embs_aug_insert = naw.ContextualWordEmbsAug(model_path='dccuchile/bert-base-spanish-wwm-cased', action="insert")
contextual_word_embs_aug_substitute = naw.ContextualWordEmbsAug(model_path='dccuchile/bert-base-spanish-wwm-cased', action="substitute")


# Aplicar aumento de datos
aumento_sinonimos = []
aumento_insert_BERT = []
aumento_substitute_BERT = []

for index, row in df.iterrows():
    original_comment = row['pregunta']
    for _ in range(33):  # Generar 33 aumentaciones por cada comentario
        # Aumentación usando sinónimos
        augmented_comment = synonym_aug.augment(original_comment)
        aumento_sinonimos.append(augmented_comment)
        # Aumentación insertando palabras con BERT
        augmented_comment = contextual_word_embs_aug_insert.augment(original_comment)
        aumento_insert_BERT.append(augmented_comment)
        # Aumentación sustituyendo palabras con BERT
        augmented_comment = contextual_word_embs_aug_substitute.augment(original_comment)
        aumento_substitute_BERT.append(augmented_comment)

# Crear un DataFrame con los comentarios aumentados
aumento_sinonimos_df = pd.DataFrame(aumento_sinonimos, columns=['pregunta'])
aumento_insert_BERT_df = pd.DataFrame(aumento_insert_BERT, columns=['pregunta'])
aumento_substitute_BERT_df = pd.DataFrame(aumento_substitute_BERT, columns=['pregunta'])
aumento_sinonimos_df['final_pregunta_isAcusatoria']=1
aumento_insert_BERT_df['final_pregunta_isAcusatoria']=1
aumento_substitute_BERT_df['final_pregunta_isAcusatoria']=1

# Convertir explícitamente a enteros
aumento_sinonimos_df['final_pregunta_isAcusatoria'] = aumento_sinonimos_df['final_pregunta_isAcusatoria'].astype(int)
aumento_insert_BERT_df['final_pregunta_isAcusatoria'] = aumento_insert_BERT_df['final_pregunta_isAcusatoria'].astype(int)
aumento_substitute_BERT_df['final_pregunta_isAcusatoria'] = aumento_substitute_BERT_df['final_pregunta_isAcusatoria'].astype(int)

# Guardar los comentarios aumentados en un archivo CSV
aumento_sinonimos_df.to_csv(f'{output_dir}/a_aumento_sinonimos.csv', index=False)
aumento_insert_BERT_df.to_csv(f'{output_dir}/b_aumento_insert_BERT.csv', index=False)
aumento_substitute_BERT_df.to_csv(f'{output_dir}/c_aumento_substitute_BERT.csv', index=False)

print("Aumento de datos completado y guardado en archivos")

# Cell 18
import pandas as pd
import nlpaug.augmenter.word as naw
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Comprobar si hay una GPU disponible y configurarla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar los datos
df = df_acusa

# Cargar modelos y tokenizers de GPT-2 en español y moverlos a la GPU
gpt2_model_name = 'datificate/gpt2-small-spanish'
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)

# Definir la función para generar texto con GPT-2
def generate_text(prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.8, top_p=0.95):
    inputs = gpt2_tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = gpt2_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_p=top_p,
        temperature=temperature,
        do_sample=True,
        pad_token_id=gpt2_tokenizer.pad_token_id
    )
    return [gpt2_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Aplicar aumento de datos
aumento_sentence_GPT2 = []

for index, row in df.iterrows():
    original_comment = row['pregunta']
    for _ in range(33):  # Generar 33 aumentaciones por cada comentario
        # Aumentación generando nuevas oraciones con GPT-2
        augmented_comment = generate_text(original_comment, max_new_tokens=200, num_return_sequences=1)[0]
        aumento_sentence_GPT2.append(augmented_comment)

# Crear un DataFrame con los comentarios aumentados
aumento_sentence_GPT2_df = pd.DataFrame(aumento_sentence_GPT2, columns=['pregunta'])
aumento_sentence_GPT2_df['final_pregunta_isAcusatoria']=1

# Convertir explícitamente a enteros
aumento_sentence_GPT2_df['final_pregunta_isAcusatoria'] = aumento_sentence_GPT2_df['final_pregunta_isAcusatoria'].astype(int)

# Guardar los comentarios aumentados en un archivo CSV
aumento_sentence_GPT2_df.to_csv(f'{output_dir}/d_aumento_sentence_GPT2.csv', index=False)

print("Aumento de datos completado y guardado en archivos")

# Cell 20
import pandas as pd
import nlpaug.augmenter.word as naw
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Comprobar si hay una GPU disponible y configurarla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar los datos
df = df_acusa

# Cargar modelos y tokenizers de GPT-2 en español y moverlos a la GPU
gpt2_model_name = 'datificate/gpt2-small-spanish'
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)

# Definir la función para generar texto con GPT-2
def generate_text(prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.8, top_p=0.95):
    inputs = gpt2_tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = gpt2_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_p=top_p,
        temperature=temperature,
        do_sample=True,
        pad_token_id=gpt2_tokenizer.pad_token_id
    )
    return [gpt2_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Aplicar aumento de datos
aumento_sentence_prompt_GPT2 = []

for index, row in df.iterrows():
    original_comment = row['pregunta']
    for _ in range(33):  # Generar 33 aumentaciones por cada comentario
        # Aumentación generando nuevas oraciones con GPT-2
        prompt = (
            f"Genera una versión diferente del siguiente comentario manteniendo el significado y contexto: \"{original_comment}\". "
            f"Asegúrate de que el comentario generado sea coherente y fluido. El comentario generado empieza después de 'Comentario: '.\n"
            "Comentario: "
        )
        augmented_comment = generate_text(prompt, max_new_tokens=200, num_return_sequences=1)[0]
        # Extraer el comentario generado después del marcador
        generated_text = augmented_comment.split("Comentario: ")[-1].strip()
        aumento_sentence_prompt_GPT2.append(generated_text)

# Crear un DataFrame con los comentarios aumentados
aumento_sentence_prompt_GPT2_df = pd.DataFrame(aumento_sentence_prompt_GPT2, columns=['pregunta'])
aumento_sentence_prompt_GPT2_df['final_pregunta_isAcusatoria']=1

# Convertir explícitamente a enteros
aumento_sentence_prompt_GPT2_df['final_pregunta_isAcusatoria'] = aumento_sentence_prompt_GPT2_df['final_pregunta_isAcusatoria'].astype(int)

# Guardar los comentarios aumentados en un archivo CSV
aumento_sentence_prompt_GPT2_df.to_csv(f'{output_dir}/e_aumento_sentence_prompt_GPT2.csv', index=False)

print("Aumento de datos completado y guardado en archivos")

# Cell 22
!pip install openai

# Cell 23
from openai import OpenAI
import os

key = 'sk-None-[AQUI COLOCAR KEY]'
os.environ["OPENAI_API_KEY"] = key

## Set the API key and model name
MODEL="gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))

# Función para generar texto usando la API de OpenAI
def generate_text(prompt, max_tokens=150, temperature=0.8, top_p=0.95, n=1):
    completion = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": "Eres un experto en inteligencia artificial especializado en la generación de textos en español y clasificación de comentarios. Debes proporcionar respuestas coherentes y contextualmente relevantes para los siguientes comentarios con el objetivo de realizar data augmentation con tus respuestas a modelos de clasificación binaria de si tiene o no indicios de corrupción. La clase proporcionada tiene solo estos pocos datos y es la clase etiquetada que presenta indicios de corrupción"}, # <-- This is the system message that provides context to the model
        {"role": "user", "content": prompt}],  # <-- This is the user message for which the model will generate a response
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=None
        )
    return [choice.message.content.strip() for choice in completion.choices]

# Cell 24
# Cargar los datos
df = df_acusa

# Aplicar aumento de datos
aumento_sentence_prompt_GPT4o_mini = []

for index, row in df.iterrows():
    original_comment = row['pregunta']
    for _ in range(33):  # Generar 33 aumentaciones por cada comentario
        prompt = (
            f"Genera una versión diferente del siguiente comentario manteniendo el significado y contexto: \"{original_comment}\".\n"
            "Asegúrate de que el comentario generado sea coherente y fluido."
        )
        augmented_comment = generate_text(prompt, max_tokens=200, temperature=0.8, top_p=0.95, n=1)
        aumento_sentence_prompt_GPT4o_mini.append(augmented_comment)

# Crear un DataFrame con los comentarios aumentados
aumento_sentence_prompt_GPT4o_mini_df = pd.DataFrame(aumento_sentence_prompt_GPT4o_mini, columns=['pregunta'])
aumento_sentence_prompt_GPT4o_mini_df['final_pregunta_isAcusatoria']=1

# Convertir explícitamente a enteros
aumento_sentence_prompt_GPT4o_mini_df['final_pregunta_isAcusatoria'] = aumento_sentence_prompt_GPT4o_mini_df['final_pregunta_isAcusatoria'].astype(int)

# Guardar los comentarios aumentados en un archivo CSV
aumento_sentence_prompt_GPT4o_mini_df.to_csv(f'{output_dir}/f_aumento_sentence_prompt_GPT4o_mini.csv', index=False)

print("Aumento de datos completado y guardado en archivos")

# Cell 26
import pandas as pd
import os

# Cargar los datasets de aumento
aumento_sinonimos = pd.read_csv(f'{output_dir}/a_aumento_sinonimos.csv')
aumento_insert_BERT = pd.read_csv(f'{output_dir}/b_aumento_insert_BERT.csv')
aumento_substitute_BERT = pd.read_csv(f'{output_dir}/c_aumento_substitute_BERT.csv')
aumento_sentence_GPT2 = pd.read_csv(f'{output_dir}/d_aumento_sentence_GPT2.csv')
aumento_sentence_prompt_GPT2 = pd.read_csv(f'{output_dir}/e_aumento_sentence_prompt_GPT2.csv')
aumento_sentence_prompt_GPT4o_mini = pd.read_csv(f'{output_dir}/f_aumento_sentence_prompt_GPT4o_mini.csv')

# Función para crear datasets balanceados con nombres específicos según el tipo de aumento
def crear_datasets(df_libre, df_acusa, aumentos, dataset_name, clase_0, clase_1):
    datasets = {}
    num_originals = len(df_acusa)
    registros_necesarios = clase_1 - num_originals

    for aumento_name, aumento_df in aumentos.items():
        # Inicializar el DataFrame de la clase 1 con los datos originales
        df_aumento_clase_1 = df_acusa.copy()

        # Seleccionar el número adecuado de registros aumentados equitativamente por cada registro original
        indices = []
        original_index = 0
        incremento = 0
        while len(indices) < registros_necesarios:
            indices.append(aumento_df.index[original_index * 33 + incremento % 33])
            original_index += 1
            if original_index >= num_originals:
                original_index = 0
                incremento += 1

        df_aumento_clase_1 = pd.concat([df_aumento_clase_1, aumento_df.loc[indices]])

        # Seleccionar registros para la clase 0
        df_aumento_clase_0 = df_libre.sample(clase_0, random_state=42)

        # Crear el dataset balanceado
        datasets[f"{dataset_name}_{aumento_name}"] = pd.concat([df_aumento_clase_0, df_aumento_clase_1]).drop_duplicates()

    return datasets

# Definir los aumentadores
aumentos = {
    'sinonimos': aumento_sinonimos,
    'insert_BERT': aumento_insert_BERT,
    'substitute_BERT': aumento_substitute_BERT,
    'sentence_GPT2': aumento_sentence_GPT2,
    'sentence_prompt_GPT2': aumento_sentence_prompt_GPT2,
    'sentence_prompt_GPT4o_mini': aumento_sentence_prompt_GPT4o_mini
}

# Crear los datasets especificados
datasets = {}
datasets['original_3886_118'] = pd.concat([df_libre, df_acusa]).drop_duplicates()
datasets['undersampled_118_118'] = pd.concat([df_libre.sample(117, random_state=42), df_acusa]).drop_duplicates()
datasets['unbalanced_1000_118'] = pd.concat([df_libre.sample(1000, random_state=42), df_acusa]).drop_duplicates()

# Crear los datasets balanceados
datasets.update(crear_datasets(df_libre, df_acusa, aumentos, 'balanced_500', 500, 500))
datasets.update(crear_datasets(df_libre, df_acusa, aumentos, 'balanced_1000', 1000, 1000))
datasets.update(crear_datasets(df_libre, df_acusa, aumentos, 'balanced_2500', 2500, 2500))
datasets.update(crear_datasets(df_libre, df_acusa, aumentos, 'balanced_total', 3886, 3880))

# Crear los datasets desbalanceados con clase 1 aumentada a 2500
datasets.update(crear_datasets(df_libre, df_acusa, aumentos, 'unbalanced_3886_2500', 3886, 2500))

# Crear un dataset que contenga a todos los demás, eliminando duplicados
todos_los_datasets = pd.concat(datasets.values(), ignore_index=True).drop_duplicates()
datasets['total_augmented_data'] = todos_los_datasets

# Crear el directorio si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar los datasets
for name, dataset in datasets.items():
    dataset.to_csv(f'{output_dir}/{name}.csv', index=False)

print("Creación de datasets completada y guardada en archivos CSV.")

# Cell 27
print(len(datasets.keys()))
datasets.keys()

# Cell 28
for datasetkey in datasets.keys():
    dataset = datasets[datasetkey]
    print(datasetkey)
    print("Número de registros totales con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(dataset['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((dataset['final_pregunta_isAcusatoria'] == 1).sum()/dataset.shape[0]*100,2 ), "%")
    print("Número de registros totales sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(dataset['final_pregunta_isAcusatoria'] == 0).sum(), " - ", round((dataset['final_pregunta_isAcusatoria'] == 0).sum()/dataset.shape[0]*100,2 ), "%")

# Cell 29
# Lista para almacenar los resultados
results = []

# Recorrer cada dataset y calcular los valores requeridos
for dataset_key in datasets.keys():
    dataset = datasets[dataset_key]
    num_risk = (dataset['final_pregunta_isAcusatoria'] == 1).sum()
    perc_risk = round(num_risk / dataset.shape[0] * 100, 2)
    num_no_risk = (dataset['final_pregunta_isAcusatoria'] == 0).sum()
    perc_no_risk = round(num_no_risk / dataset.shape[0] * 100, 2)

    # Almacenar los resultados en un diccionario
    result = {
        'Dataset': dataset_key,
        'Num_Registros_Riesgo': num_risk,
        'Perc_Registros_Riesgo': perc_risk,
        'Num_Registros_No_Riesgo': num_no_risk,
        'Perc_Registros_No_Riesgo': perc_no_risk
    }

    # Agregar el diccionario a la lista de resultados
    results.append(result)

# Convertir la lista de resultados en un DataFrame
results_df = pd.DataFrame(results)

# Mostrar el DataFrame
print(results_df)
results_df.to_csv(f'{output_dir}/data.csv', index=False)

# Cell 31
# ruta correcta a archivo Excel
path = '/home/froh/train_datasetAUG.csv'
df = pd.read_csv(path)

df.head()

# Cell 32
print("Número de registros con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(df['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((df['final_pregunta_isAcusatoria'] == 1).sum()/df.shape[0]*100,2 ), "%")
print("Número de registros sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(df['final_pregunta_isAcusatoria'] == 0).sum(), " - ",  round((df['final_pregunta_isAcusatoria'] == 0).sum()/df.shape[0]*100,2 ), "%")


# Eliminar las columnas 'contract_id', 'pregunta_id' y 'sum_pregunta_isAcusatoria'
df = df.drop(columns=['contract_id', 'pregunta_id', 'sum_pregunta_isAcusatoria'])


# Crear el dataset especificado
datasets['balanced_AUG_GPT4o_mini'] = df

# Cell 34
datasets['total_augmented_data']

# Cell 35
print("Número de registros con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(datasets['total_augmented_data']['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((datasets['total_augmented_data']['final_pregunta_isAcusatoria'] == 1).sum()/datasets['total_augmented_data'].shape[0]*100,2 ), "%")
print("Número de registros sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(datasets['total_augmented_data']['final_pregunta_isAcusatoria'] == 0).sum(), " - ",  round((datasets['total_augmented_data']['final_pregunta_isAcusatoria'] == 0).sum()/datasets['total_augmented_data'].shape[0]*100,2 ), "%")

# Cell 36
# Crear el dataset especificado
datasets['balanced_AUG_GPT4o_mini_total_augmented'] = pd.concat([datasets['balanced_AUG_GPT4o_mini'], datasets['total_augmented_data']])
# Eliminar duplicados de la concatenación
datasets['balanced_AUG_GPT4o_mini_total_augmented'] = datasets['balanced_AUG_GPT4o_mini_total_augmented'].drop_duplicates()

print("Número de registros con riesgo de corrupción (final_pregunta_isAcusatoria=1): ",(datasets['balanced_AUG_GPT4o_mini_total_augmented']['final_pregunta_isAcusatoria'] == 1).sum(), " - ", round((datasets['balanced_AUG_GPT4o_mini_total_augmented']['final_pregunta_isAcusatoria'] == 1).sum()/datasets['balanced_AUG_GPT4o_mini_total_augmented'].shape[0]*100,2 ), "%")
print("Número de registros sin riesgo de corrupción (final_pregunta_isAcusatoria=0): ",(datasets['balanced_AUG_GPT4o_mini_total_augmented']['final_pregunta_isAcusatoria'] == 0).sum(), " - ",  round((datasets['balanced_AUG_GPT4o_mini_total_augmented']['final_pregunta_isAcusatoria'] == 0).sum()/datasets['balanced_AUG_GPT4o_mini_total_augmented'].shape[0]*100,2 ), "%")

# Cell 37
print(len(datasets.keys()))
datasets.keys()

# Cell 40
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from openai import OpenAI
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Importaciones de Texto
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from gensim.models import KeyedVectors
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

train_data = train_df.copy()
test_data = test_df.copy()

# Configuración de la carpeta de salida
output_dir = '/home/froh/FinalProject_cl'
#Verifica directorios
if not os.path.exists(output_dir): os.makedirs(output_dir)

output_dir_emb = '/home/froh/FinalProject_cl/embeddings'
if not os.path.exists(output_dir_emb): os.makedirs(output_dir_emb)

# Configuración del cliente OpenAI para embeddings
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Cargar los vectores de palabras desde el archivo de texto
path = '/home/froh/a_cc.es.300.vec'
model_ff = KeyedVectors.load_word2vec_format(path, binary=False)


# Cargar el modelo en español
nlp = spacy.load('es_core_news_sm')



# Cell 42
from sentence_transformers import SentenceTransformer

# Función para convertir texto en un vector promedio de word embeddings
def text_to_vector(words, model):
    total_words = len(words)
    found_words = 0
    word_vectors = []
    not_found_list =[]  # Lista para almacenar palabras no encontradas

    #words = text.split()  # Asumiendo que text es una cadena de lemas separados por espacios
    for word in words:
        if word in model.key_to_index:
            word_vectors.append(model[word])
            found_words += 1
        else:
            not_found_list.append(word)  # Añadir la palabra no encontrada a la lista

    if not word_vectors:
        average_vector = np.zeros(model.vector_size)
    else:
        average_vector = np.mean(np.array(word_vectors), axis=0)

    # Cantidad de palabras no encontradas
    not_found_words = total_words - found_words

    return average_vector, total_words, found_words, not_found_words, not_found_list

def preprocesar(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Remover caracteres no alfabéticos
    texto = re.sub(r'[^a-zñáéíóú]+', ' ', texto, flags=re.IGNORECASE)
    # Procesamiento con spaCy
    doc = nlp(texto)
    # Lemmatización y remoción de stopwords
    lemmas = [token.lemma_ for token in doc if token.text not in STOP_WORDS and len(token.text) > 1]
    return lemmas, ' '.join(lemmas)

def create_sbert_embeddings(df, test_data, model_name='all-mpnet-base-v2', preprocesamiento = "False"):
    # Cargar el modelo SBERT
    sbert_model = SentenceTransformer(model_name)
    if preprocesamiento == "False":
      # Crear embeddings para el conjunto de entrenamiento
      df['embedding'] = df['pregunta'].apply(lambda x: sbert_model.encode(x))
    else:
      df['embedding'] = df['texto_preprocesado'].apply(lambda x: sbert_model.encode(x))

    X = np.array(df['embedding'].tolist())

    # Crear embeddings para el conjunto de prueba
    if os.path.exists(f'{output_dir_emb}/test_embeddings_SBERT.csv'):
        X_test = None
    else:
        if preprocesamiento == "False":
          test_data['embedding'] = test_data['pregunta'].apply(lambda x: sbert_model.encode(x))
        else:
          test_data['embedding'] = test_data['texto_preprocesado'].apply(lambda x: sbert_model.encode(x))
        X_test = np.array(test_data['embedding'].tolist())

    return X, X_test, df['final_pregunta_isAcusatoria'], test_data['final_pregunta_isAcusatoria']

# Función para crear embeddings
def create_embeddings(df, test_data, embedding_type, preprocesamiento="False"):
    y = df['final_pregunta_isAcusatoria']
    y_test = test_data['final_pregunta_isAcusatoria']

    if embedding_type == 'BoW':
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['texto_preprocesado']).toarray()
        X_test = vectorizer.transform(test_data['texto_preprocesado']).toarray()
    elif embedding_type == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['texto_preprocesado']).toarray()
        X_test = vectorizer.transform(test_data['texto_preprocesado']).toarray()
    elif embedding_type == 'Doc2Vec':
        documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(df['texto_preprocesado'])]
        model = Doc2Vec(documents, vector_size=200, window=2, min_count=1, workers=4)
        df['embedding'] = df['texto_preprocesado'].apply(lambda x: model.infer_vector(x.split()))
        test_data['embedding'] = test_data['texto_preprocesado'].apply(lambda x: model.infer_vector(x.split()))
        X = np.array(df['embedding'].tolist())
        X_test = np.array(test_data['embedding'].tolist())
    elif embedding_type == 'FastText':
        results = df['texto_preprocesado'].apply(lambda text: text_to_vector(text, model_ff))
        df['ff_average_vector'] = results.apply(lambda x: x[0])
        X = np.array(df['ff_average_vector'].tolist())
        if os.path.exists(f'{output_dir_emb}/test_embeddings_{embedding}.csv'):
            X_test = None
        else:
            results_test = test_data['texto_preprocesado'].apply(lambda text: text_to_vector(text, model_ff))
            test_data['ff_average_vector'] = results_test.apply(lambda x: x[0])
            X_test = np.array(test_data['ff_average_vector'].tolist())
    elif embedding_type == 'text-embedding-3-large':
        df['embedding'] = df['pregunta'].apply(lambda x: get_embedding(x, model="text-embedding-3-large"))
        X = np.array(df['embedding'].tolist())
        if os.path.exists(f'{output_dir_emb}/test_embeddings_{embedding}.csv'):
            X_test = None
        else:
            test_data['embedding'] = test_data['pregunta'].apply(lambda x: get_embedding(x, model="text-embedding-3-large"))
            X_test = np.array(test_data['embedding'].tolist())
    elif embedding_type == 'SBERT':
        X, X_test, y, y_test = create_sbert_embeddings(df, test_data, preprocesamiento=preprocesamiento)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    return X, X_test, y, y_test

# Aplicar la función de preprocesamiento

test_data['lista_preprocesada'],test_data['texto_preprocesado'] = zip(*test_data['pregunta'].apply(preprocesar))
print("Data de test GENERAL preprocesada\n")

# Cell 44
# Generar y guardar embeddings en archivos CSV separados

# Definir los embeddings y modelos a utilizar
embeddings = ['BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'text-embedding-3-large']
datasets_keys = ['original_3886_118', 'undersampled_118_118', 'unbalanced_1000_118', 'balanced_500_sinonimos',
                 'balanced_500_insert_BERT', 'balanced_500_substitute_BERT', 'balanced_500_sentence_GPT2',
                 'balanced_500_sentence_prompt_GPT2', 'balanced_500_sentence_prompt_GPT4o_mini', 'balanced_1000_sinonimos',
                 'balanced_1000_insert_BERT', 'balanced_1000_substitute_BERT', 'balanced_1000_sentence_GPT2', 'balanced_1000_sentence_prompt_GPT2',
                 'balanced_1000_sentence_prompt_GPT4o_mini', 'balanced_2500_sinonimos', 'balanced_2500_insert_BERT', 'balanced_2500_substitute_BERT',
                 'balanced_2500_sentence_GPT2', 'balanced_2500_sentence_prompt_GPT2', 'balanced_2500_sentence_prompt_GPT4o_mini',
                 'balanced_total_sinonimos', 'balanced_total_insert_BERT', 'balanced_total_substitute_BERT', 'balanced_total_sentence_GPT2',
                 'balanced_total_sentence_prompt_GPT2', 'balanced_total_sentence_prompt_GPT4o_mini', 'unbalanced_3886_2500_sinonimos',
                 'unbalanced_3886_2500_insert_BERT', 'unbalanced_3886_2500_substitute_BERT', 'unbalanced_3886_2500_sentence_GPT2',
                 'unbalanced_3886_2500_sentence_prompt_GPT2', 'unbalanced_3886_2500_sentence_prompt_GPT4o_mini', 'total_augmented_data',
                 'balanced_AUG_GPT4o_mini', 'balanced_AUG_GPT4o_mini_total_augmented']

for dataset_key in datasets_keys:
    df = datasets[dataset_key]  # datasets es un diccionario con los DataFrames
    df['lista_preprocesada'],df['texto_preprocesado'] = zip(*df['pregunta'].apply(preprocesar))
    print(f"\nData de train {dataset_key} preprocesada\n")
    for embedding in embeddings:
        print(f"Generando embeddings {embedding} para {dataset_key}...")
        train_embeddings, test_embeddings, train_y, test_y = create_embeddings(df, test_data, embedding)

        # Guardar los embeddings de entrenamiento en un archivo CSV separado
        train_df = pd.DataFrame(train_embeddings)
        train_df['label'] = train_y.values
        train_df.to_csv(f'{output_dir_emb}/train_embeddings_{dataset_key}_{embedding}.csv', index=False)

        # Guardar los embeddings de prueba en un archivo CSV separado
        if embedding in ['BoW', 'TF-IDF', 'Doc2Vec']:
            test_df = pd.DataFrame(test_embeddings)
            test_df['label'] = test_y.values
            test_df.to_csv(f'{output_dir_emb}/test_embeddings_{dataset_key}_{embedding}.csv', index=False)
        else:
            if not os.path.exists(f'{output_dir_emb}/test_embeddings_{embedding}.csv'):
                test_df = pd.DataFrame(test_embeddings)
                test_df['label'] = test_y.values
                test_df.to_csv(f'{output_dir_emb}/test_embeddings_{embedding}.csv', index=False)

        print(f"Embeddings {embedding} generados y guardados para {dataset_key}")

    print("\nEmbeddings generados y guardados en archivos CSV separados")

# Cell 46
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
from sklearn.impute import SimpleImputer
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from itertools import product

# Filtro para advertencias de runtime
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuración de carpetas
output_dir = '/home/froh/FinalProject_cl'
output_dir_emb = '/home/froh/FinalProject_cl/embeddings'

# Definir los embeddings y modelos a utilizar
embeddings = ['BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'text-embedding-3-large', 'SBERT']
model_types = ['Logistic_Regression', 'Naive_Bayes', 'Random_Forest_Classifier', 'SVC']
datasets_keys = ['original_3886_118', 'undersampled_118_118', 'unbalanced_1000_118', 'balanced_500_sinonimos',
                 'balanced_500_insert_BERT', 'balanced_500_substitute_BERT', 'balanced_500_sentence_GPT2',
                 'balanced_500_sentence_prompt_GPT2', 'balanced_500_sentence_prompt_GPT4o_mini', 'balanced_1000_sinonimos',
                 'balanced_1000_insert_BERT', 'balanced_1000_substitute_BERT', 'balanced_1000_sentence_GPT2', 'balanced_1000_sentence_prompt_GPT2',
                 'balanced_1000_sentence_prompt_GPT4o_mini', 'balanced_2500_sinonimos', 'balanced_2500_insert_BERT', 'balanced_2500_substitute_BERT',
                 'balanced_2500_sentence_GPT2', 'balanced_2500_sentence_prompt_GPT2', 'balanced_2500_sentence_prompt_GPT4o_mini',
                 'balanced_total_sinonimos', 'balanced_total_insert_BERT', 'balanced_total_substitute_BERT', 'balanced_total_sentence_GPT2',
                 'balanced_total_sentence_prompt_GPT2', 'balanced_total_sentence_prompt_GPT4o_mini', 'unbalanced_3886_2500_sinonimos',
                 'unbalanced_3886_2500_insert_BERT', 'unbalanced_3886_2500_substitute_BERT', 'unbalanced_3886_2500_sentence_GPT2',
                 'unbalanced_3886_2500_sentence_prompt_GPT2', 'unbalanced_3886_2500_sentence_prompt_GPT4o_mini', 'total_augmented_data',
                 'balanced_AUG_GPT4o_mini', 'balanced_AUG_GPT4o_mini_total_augmented']



# Parámetros para GridSearch
param_grid = {
    'Logistic_Regression': {
        'C': [0.01, 1, 100],
        'penalty': ['l2', 'elasticnet'], #'l1',
        'solver': ['saga']
    },
    'Naive_Bayes': {
        'var_smoothing': np.logspace(0, -9, num=100)
    },
    'Random_Forest_Classifier': {
        'n_estimators': [80, 200],
        'max_depth': [None, 10, 30],
        #'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4]
    },
    'SVC': {
        'C': [1, 100]#0.01,
        #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #'gamma': ['scale', 'auto']
    }
}

# cv para datos desbalanceados
#rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

# cv para datos desbalanceados usando StratifiedKFold
rskf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Función para contar combinaciones de parámetros de un modelo específico
def count_combinations_for_model(params):
    param_combinations = product(*params.values())
    num_combinations = len(list(param_combinations))
    return num_combinations

# Función para realizar GridSearch y encontrar el mejor modelo
def grid_search_model(model_type, X_train, y_train):
    if model_type == 'Logistic_Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'Naive_Bayes':
        model = GaussianNB()
    elif model_type == 'Random_Forest_Classifier':
        model = RandomForestClassifier()
    elif model_type == 'SVC':
        model = SVC(probability=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    param_grid_for_model = param_grid[model_type]
    num_combinations = count_combinations_for_model(param_grid_for_model)

    grid = GridSearchCV(estimator=model, param_grid=param_grid_for_model, scoring='f1', cv=rskf, return_train_score=True, n_jobs=8)
    start_time = time.time()
    grid.fit(X_train, y_train)
    runtime = (time.time() - start_time) / 60
    avg_runtime = runtime / num_combinations

    print(f"Mejor modelo para {model_type}: {grid.best_estimator_}. Mejores parámetros para {model_type}: {grid.best_params_}. Mejor puntuación F1 para {model_type}: {grid.best_score_}.")
    print(f"Tiempo promedio de ejecución para {model_type}: {avg_runtime} minutos. (Tiempo total: {runtime} minutos para {num_combinations} combinaciones de param.)")

    return grid.best_estimator_, grid.best_params_, grid.cv_results_, runtime, avg_runtime, num_combinations


def plot_learning_curve(estimator, title, X, y, output_dir, dataset_key, model_type, embedding, cv=rskf, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(7, 5))
    plt.title(title, fontsize=10)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, val_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes, return_times=True, n_jobs=8)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")

    learning_curve_path = os.path.join(output_dir, f'learningcurve_{dataset_key}_{model_type}_{embedding}.png')
    plt.savefig(learning_curve_path)
    plt.show()
    plt.close()

    return learning_curve_path

def save_plot(fpr, tpr, roc_auc, precision, recall, average_precision, cm, title_prefix, output_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title_prefix}\nROC Curve')
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, f'roc_curve_{title_prefix}.png')
    plt.savefig(roc_path)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (area = {average_precision:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix}\nPrecision-Recall Curve')
    plt.legend(loc="lower left")
    prc_path = os.path.join(output_dir, f'prc_curve_{title_prefix}.png')
    plt.savefig(prc_path)
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], annot_kws={"size": 70}, cbar=False)
    plt.title(f'{title_prefix}\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(output_dir, f'conf_matrix_{title_prefix}.png')
    plt.savefig(cm_path)
    plt.close()

    return roc_path, prc_path, cm_path

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, output_dir, dataset_key, model_type, embedding, runtime, avg_runtime, num_combinations):
    if not model_type == 'SVC':
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]

    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    def get_metrics(y_true, y_pred, y_prob):
        return {
            'f1': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc_curve': roc_curve(y_true, y_prob),
            'auc': auc(roc_curve(y_true, y_prob)[0], roc_curve(y_true, y_prob)[1]),
            'precision_recall_curve': precision_recall_curve(y_true, y_prob),
            'average_precision': average_precision_score(y_true, y_prob)
        }
    if not model_type == 'SVC':
        train_metrics = get_metrics(y_train, y_train_pred, y_train_prob)
    val_metrics = get_metrics(y_val, y_val_pred, y_val_prob)
    test_metrics = get_metrics(y_test, y_test_pred, y_test_prob)

    if not model_type == 'SVC':
        train_roc_path, train_prc_path, train_cm_path = save_plot(
            train_metrics['roc_curve'][0], train_metrics['roc_curve'][1], train_metrics['auc'],
            train_metrics['precision_recall_curve'][1], train_metrics['precision_recall_curve'][0], train_metrics['average_precision'],
            train_metrics['confusion_matrix'], f'{dataset_key}_{model_type}_{embedding}_train', output_dir
        )
    val_roc_path, val_prc_path, val_cm_path = save_plot(
        val_metrics['roc_curve'][0], val_metrics['roc_curve'][1], val_metrics['auc'],
        val_metrics['precision_recall_curve'][1], val_metrics['precision_recall_curve'][0], val_metrics['average_precision'],
        val_metrics['confusion_matrix'], f'{dataset_key}_{model_type}_{embedding}_val', output_dir
    )

    test_roc_path, test_prc_path, test_cm_path = save_plot(
        test_metrics['roc_curve'][0], test_metrics['roc_curve'][1], test_metrics['auc'],
        test_metrics['precision_recall_curve'][1], test_metrics['precision_recall_curve'][0], test_metrics['average_precision'],
        test_metrics['confusion_matrix'], f'{dataset_key}_{model_type}_{embedding}_test', output_dir
    )
    learning_curve_path = plot_learning_curve(model, f"{dataset_key} - {model_type} - {embedding}\nLearning Curve", X_train, y_train, output_dir, dataset_key, model_type, embedding, cv=rskf)
    print(f"F1 en Test: {test_metrics['f1']}")
    result = {
        'dataset': dataset_key,
        'model': model_type,
        'embedding': embedding,
        'avg_runtime_minutes': avg_runtime,
        'test_f1': test_metrics['f1'],
        'test_conf_matrix': test_cm_path,
        'learning_curve': learning_curve_path,
        'test_roc_curve': test_roc_path,
        'test_prc_curve': test_prc_path,
        'test_acc': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'runtime_minutes': runtime,
        'num_combinations': num_combinations,
        'val_conf_matrix': val_cm_path,
        'val_roc_curve': val_roc_path,
        'val_prc_curve': val_prc_path,
        'val_f1': val_metrics['f1'],
        'val_acc': val_metrics['accuracy'],
        'val_precision': val_metrics['precision'],
        'val_recall': val_metrics['recall']
    }

    if model_type != 'SVC':
        result.update({
            'train_conf_matrix': train_cm_path,
            'train_roc_curve': train_roc_path,
            'train_prc_curve': train_prc_path,
            'train_f1': train_metrics['f1'],
            'train_acc': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall']
        })

    return result

# Función para cargar embeddings desde archivos CSV
def load_embeddings(dataset, embedding_type):
    # Cargar los embeddings de entrenamiento desde el archivo CSV correspondiente
    train_df = pd.read_csv(f'{output_dir_emb}/train_embeddings_{dataset_key}_{embedding}.csv')
    train_embeddings = train_df.drop(['label'], axis=1).values
    y_train = train_df['label'].values

    # Cargar los embeddings de prueba desde el archivo CSV correspondiente
    if embedding_type in ['BoW', 'TF-IDF', 'Doc2Vec']:
        test_df = pd.read_csv(f'{output_dir_emb}/test_embeddings_{dataset_key}_{embedding}.csv')
    else:
        test_df = pd.read_csv(f'{output_dir_emb}/test_embeddings_{embedding_type}.csv')
    test_embeddings = test_df.drop(['label'], axis=1).values
    y_test = test_df['label'].values

    return train_embeddings, test_embeddings, y_train, y_test

# Función para guardar resultados incrementalmente
def save_results_incremental(results, output_file):
    if os.path.exists(output_file):
        existing_results = pd.read_csv(output_file)
        results_df = pd.concat([existing_results, pd.DataFrame([results])], ignore_index=True)
    else:
        results_df = pd.DataFrame([results])
    results_df.to_csv(output_file, index=False)

# Leer resultados existentes
output_file = f'{output_dir}/results_summary_cl.csv'
if os.path.exists(output_file):
    results_cl_df = pd.read_csv(output_file)
    existing_results = results_cl_df.to_dict(orient='records')
else:
    existing_results = []

# Unificar todo el código para realizar embeddings, entrenamiento y evaluación
results = []

for dataset_key in datasets_keys:
    print(f"\nDataset: {dataset_key}")
    for embedding in embeddings:
        print(f"\nCargando embeddings {embedding} para {dataset_key}")
        X_train, X_test, y_train, y_test = load_embeddings(dataset_key, embedding)
        # Estratificar la división de los datos
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
        print(f"Embeddings {embedding} cargados para {dataset_key}, registros (Train/Val/Test): ", len(X_train), len(X_val), len(X_test))

        for model_type in model_types:
            print(f"\nEntrenando... -> Dataset: {dataset_key}, Embedding: {embedding}, Model: {model_type}")
            best_model, best_params, grid_results, runtime, avg_runtime, num_combinations = grid_search_model(model_type, X_train, y_train)

            # Guardar los resultados del GridSearchCV
            grid_results_df = pd.DataFrame(grid_results)
            grid_results_df.to_csv(f'{output_dir}/gridsearch_results_{dataset_key}_{embedding}_{model_type}.csv', index=False)

            result = evaluate_model(best_model, X_train, y_train, X_val, y_val, X_test, y_test, output_dir, dataset_key, model_type, embedding, runtime, avg_runtime, num_combinations)
            result['best_params'] = best_params
            results.append(result)
            save_results_incremental(result, output_file)
            print(f"Evaluación completada -> Dataset: {dataset_key}, Embedding: {embedding}, Model: {model_type}")

print("\nResultados guardados en 'results_summary_cl.csv'")

# Cell 48
output_dir = '/home/froh/FinalProject_SBERT_AUG_total_cl_PRE'

#Verifica directorios
if not os.path.exists(output_dir): os.makedirs(output_dir)

output_dir_emb = '/home/froh/FinalProject_SBERT_AUG_total_cl_PRE/embeddings'
if not os.path.exists(output_dir_emb): os.makedirs(output_dir_emb)


# Definir los embeddings y modelos a utilizar
embeddings = ['SBERT']
model_types = ['Logistic_Regression', 'Naive_Bayes', 'Random_Forest_Classifier', 'SVC']
datasets_keys = ['original_3886_118', 'undersampled_118_118', 'unbalanced_1000_118', 'balanced_500_sinonimos',
                 'balanced_500_insert_BERT', 'balanced_500_substitute_BERT', 'balanced_500_sentence_GPT2',
                 'balanced_500_sentence_prompt_GPT2', 'balanced_500_sentence_prompt_GPT4o_mini', 'balanced_1000_sinonimos',
                 'balanced_1000_insert_BERT', 'balanced_1000_substitute_BERT', 'balanced_1000_sentence_GPT2', 'balanced_1000_sentence_prompt_GPT2',
                 'balanced_1000_sentence_prompt_GPT4o_mini', 'balanced_2500_sinonimos', 'balanced_2500_insert_BERT', 'balanced_2500_substitute_BERT',
                 'balanced_2500_sentence_GPT2', 'balanced_2500_sentence_prompt_GPT2', 'balanced_2500_sentence_prompt_GPT4o_mini',
                 'balanced_total_sinonimos', 'balanced_total_insert_BERT', 'balanced_total_substitute_BERT', 'balanced_total_sentence_GPT2',
                 'balanced_total_sentence_prompt_GPT2', 'balanced_total_sentence_prompt_GPT4o_mini', 'unbalanced_3886_2500_sinonimos',
                 'unbalanced_3886_2500_insert_BERT', 'unbalanced_3886_2500_substitute_BERT', 'unbalanced_3886_2500_sentence_GPT2',
                 'unbalanced_3886_2500_sentence_prompt_GPT2', 'unbalanced_3886_2500_sentence_prompt_GPT4o_mini', 'total_augmented_data',
                 'balanced_AUG_GPT4o_mini', 'balanced_AUG_GPT4o_mini_total_augmented']


# Generar y guardar embeddings en archivos CSV separados

for dataset_key in datasets_keys:
    df = datasets[dataset_key]  # datasets es un diccionario con los DataFrames
    df['lista_preprocesada'],df['texto_preprocesado'] = zip(*df['pregunta'].apply(preprocesar))
    print(f"\nData de train {dataset_key} preprocesada\n")
    for embedding in embeddings:
        print(f"Generando embeddings {embedding} para {dataset_key}...")
        train_embeddings, test_embeddings, train_y, test_y = create_embeddings(df, test_data, embedding, preprocesamiento="True")

        # Guardar los embeddings de entrenamiento en un archivo CSV separado
        train_df = pd.DataFrame(train_embeddings)
        train_df['label'] = train_y.values
        train_df.to_csv(f'{output_dir_emb}/train_embeddings_{dataset_key}_{embedding}.csv', index=False)

        # Guardar los embeddings de prueba en un archivo CSV separado
        if embedding in ['BoW', 'TF-IDF', 'Doc2Vec']:
            test_df = pd.DataFrame(test_embeddings)
            test_df['label'] = test_y.values
            test_df.to_csv(f'{output_dir_emb}/test_embeddings_{dataset_key}_{embedding}.csv', index=False)
        else:
            if not os.path.exists(f'{output_dir_emb}/test_embeddings_{embedding}.csv'):
                test_df = pd.DataFrame(test_embeddings)
                test_df['label'] = test_y.values
                test_df.to_csv(f'{output_dir_emb}/test_embeddings_{embedding}.csv', index=False)

        print(f"Embeddings {embedding} generados y guardados para {dataset_key}")

    print("\nEmbeddings generados y guardados en archivos CSV separados")


# Cell 49
output_dir = '/home/froh/FinalProject_SBERT_AUG_total_cl_PRE'

#Verifica directorios
if not os.path.exists(output_dir): os.makedirs(output_dir)

output_dir_emb = '/home/froh/FinalProject_SBERT_AUG_total_cl_PRE/embeddings'
if not os.path.exists(output_dir_emb): os.makedirs(output_dir_emb)


# Definir los embeddings y modelos a utilizar
embeddings = ['SBERT']
model_types = ['Logistic_Regression', 'Naive_Bayes', 'Random_Forest_Classifier', 'SVC']
datasets_keys = ['original_3886_118', 'undersampled_118_118', 'unbalanced_1000_118', 'balanced_500_sinonimos', 'balanced_500_insert_BERT', 'balanced_500_substitute_BERT',
                 'balanced_500_sentence_GPT2', 'balanced_500_sentence_prompt_GPT2', 'balanced_500_sentence_prompt_GPT4o_mini', 'balanced_1000_sinonimos',
                 'balanced_1000_insert_BERT', 'balanced_1000_substitute_BERT', 'balanced_1000_sentence_GPT2', 'balanced_1000_sentence_prompt_GPT2',
                 'balanced_1000_sentence_prompt_GPT4o_mini', 'balanced_2500_sinonimos', 'balanced_2500_insert_BERT', 'balanced_2500_substitute_BERT',
                 'balanced_2500_sentence_GPT2', 'balanced_2500_sentence_prompt_GPT2', 'balanced_2500_sentence_prompt_GPT4o_mini', 'balanced_total_sinonimos',
                 'balanced_total_insert_BERT', 'balanced_total_substitute_BERT', 'balanced_total_sentence_GPT2', 'balanced_total_sentence_prompt_GPT2',
                 'balanced_total_sentence_prompt_GPT4o_mini', 'unbalanced_3886_2500_sinonimos', 'unbalanced_3886_2500_insert_BERT', 'unbalanced_3886_2500_substitute_BERT',
                 'unbalanced_3886_2500_sentence_GPT2', 'unbalanced_3886_2500_sentence_prompt_GPT2', 'unbalanced_3886_2500_sentence_prompt_GPT4o_mini',
                 'total_augmented_data', 'balanced_AUG_GPT4o_mini', 'balanced_AUG_GPT4o_mini_total_augmented']

# Filtro para advertencias de runtime
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Parámetros para GridSearch
param_grid = {
    'Logistic_Regression': {
        'C': [0.01, 1, 100],
        'penalty': ['l2', 'elasticnet'], #'l1',
        'solver': ['saga']
    },
    'Naive_Bayes': {
        'var_smoothing': np.logspace(0, -9, num=100)
    },
    'Random_Forest_Classifier': {
        'n_estimators': [80, 200],
        'max_depth': [None, 10, 30],
        #'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4]
    },
    'SVC': {
        'C': [1, 100]#0.01,
        #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #'gamma': ['scale', 'auto']
    }
}

# cv para datos desbalanceados
#rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

# cv para datos desbalanceados usando StratifiedKFold
rskf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# Leer resultados existentes
output_file = f'{output_dir}/results_summary_SBERT_AUG_cl.csv'
if os.path.exists(output_file):
    results_cl_df = pd.read_csv(output_file)
    existing_results = results_cl_df.to_dict(orient='records')
else:
    existing_results = []

# Unificar todo el código para realizar embeddings, entrenamiento y evaluación
results = []

for dataset_key in datasets_keys:
    print(f"\nDataset: {dataset_key}")
    for embedding in embeddings:
        print(f"\nCargando embeddings {embedding} para {dataset_key}")
        X_train, X_test, y_train, y_test = load_embeddings(dataset_key, embedding)
        # Estratificar la división de los datos
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
        print(f"Embeddings {embedding} cargados para {dataset_key}, registros (Train/Val/Test): ", len(X_train), len(X_val), len(X_test))

        for model_type in model_types:
            print(f"\nEntrenando... -> Dataset: {dataset_key}, Embedding: {embedding}, Model: {model_type}")
            best_model, best_params, grid_results, runtime, avg_runtime, num_combinations = grid_search_model(model_type, X_train, y_train)

            # Guardar los resultados del GridSearchCV
            grid_results_df = pd.DataFrame(grid_results)
            grid_results_df.to_csv(f'{output_dir}/gridsearch_results_{dataset_key}_{embedding}_{model_type}.csv', index=False)

            result = evaluate_model(best_model, X_train, y_train, X_val, y_val, X_test, y_test, output_dir, dataset_key, model_type, embedding, runtime, avg_runtime, num_combinations)
            result['best_params'] = best_params
            results.append(result)
            save_results_incremental(result, output_file)
            print(f"Evaluación completada -> Dataset: {dataset_key}, Embedding: {embedding}, Model: {model_type}")

print("\nResultados guardados en 'results_summary_SBERT_AUG_cl.csv'")


# Cell 52
import re

def make_valid_filename(s):
    """
    Convierte una cadena en un nombre de archivo válido:
    - Reemplaza espacios con guiones bajos
    - Elimina caracteres que no son letras, números, guiones bajos o guiones
    """
    s = str(s)
    s = s.replace(' ', '_')  # Reemplaza espacios con guiones bajos
    return re.sub(r'[^A-Za-z0-9_\-]', '', s)  # Elimina caracteres no válidos

# Cell 53
import pandas as pd
from torch.utils.data import Dataset

class PreguntaDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.drop_duplicates()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pregunta = self.dataframe.iloc[idx]['pregunta']
        label = self.dataframe.iloc[idx]['final_pregunta_isAcusatoria']
        return pregunta, label

# Cell 55
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, RobertaTokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import math
import torchtext
import re
torchtext.disable_torchtext_deprecation_warning()

class PreguntaDataModule(pl.LightningDataModule):
    def __init__(self, dataframe, model_name, batch_size=32, embedding_dim=512):
        super().__init__()
        self.dataframe = dataframe
        self.test_dataset = PreguntaDataset(test_df)  # Crear un dataset para el test
        self.model_name = model_name.lower()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.num_workers = NUM_WORKERS
        self.tokenizers = {
            'bert': BertTokenizer.from_pretrained('bert-base-uncased'),
            'roberta': RobertaTokenizer.from_pretrained('roberta-base'),
            'auto_bert': BertTokenizer.from_pretrained('bert-base-uncased'),
            'auto_roberta': RobertaTokenizer.from_pretrained('roberta-base')
            }

        if self.model_name not in self.tokenizers and self.model_name not in ['rnn_gru', 'rnn_lstm']:
            raise ValueError("Model name must be 'bert', 'roberta', 'auto_bert', 'auto_roberta', 'GRU' or 'LSTM'")

        if self.model_name in ['rnn_gru', 'rnn_lstm']:
            self.text_tokenizer = word_tokenize
            self.vocab = self.build_vocab(self.dataframe['pregunta'])
            self.vocab_size = len(self.vocab)
            if embedding_dim is None:
                self.embedding_dim = int(math.sqrt(self.vocab_size))  # Calcular automáticamente embedding_dim
            else:
                self.embedding_dim = embedding_dim
        else:
            self.vocab_size = None

    def build_vocab(self, texts):
        token_generator = (self.text_tokenizer(text) for text in texts)
        vocab = build_vocab_from_iterator(token_generator, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def setup(self, stage=None):

        train_df, val_df = train_test_split(self.dataframe, test_size=0.2, stratify=self.dataframe['final_pregunta_isAcusatoria'], random_state=42)
        self.train_dataset = PreguntaDataset(train_df)
        self.val_dataset = PreguntaDataset(val_df)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Eliminar caracteres no alfanuméricos excepto espacios
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Eliminar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_embed_batch(self, batch):
        texts, labels = zip(*batch)
        texts = [self.clean_text(text) for text in texts]  # Limpiar textos

        if self.model_name in ['bert', 'roberta', 'auto_bert', 'auto_roberta']:
            tokenizer = self.tokenizers[self.model_name]
            inputs = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
        else:  # For GRU or LSTM
            tokenized_texts = [self.text_tokenizer(text) for text in texts]
            sequences = [torch.tensor(self.vocab(tokens), dtype=torch.long) for tokens in tokenized_texts]
            max_length = max(len(seq) for seq in sequences)
            inputs = pad_sequence(sequences, batch_first=True, padding_value=0)

        # Convertir labels a enteros si son decimales y luego a tensores
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
        return inputs, labels

    def collate_fn(self, batch):
        return self.tokenize_and_embed_batch(batch)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

# Cell 57
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from transformers import BertModel, RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


class TextClassifierBase(pl.LightningModule):
    def __init__(self, model_type, n_classes, n_training_steps=None, n_warmup_steps=None, hidden_dim=None, n_layers=None, bidirectional=None, dropout=None, embedding_dim=None,
                 vocab_size=None, unfreeze_layers=12, use_pretrained_embeddings=False, pretrained_embeddings=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.n_classes = n_classes
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.unfreeze_layers = unfreeze_layers
        self.last_train_preds = []
        self.last_train_labels = []
        self.last_val_preds = []
        self.last_val_labels = []
        self.test_probs = []
        self.test_preds = []
        self.test_labels = []
        self.train_image_path = None
        self.val_image_path = None
        self.test_image_path = None
        self.test_roc_path = None
        self.test_prc_path = None
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_embeddings = pretrained_embeddings  # Embeddings preentrenados si es necesario

        if model_type == "bert":
            model_name = "dccuchile/bert-base-spanish-wwm-cased"  # Mejor modelo BERT en español
            self.model = BertModel.from_pretrained(model_name, return_dict=True)
            self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        elif model_type == "auto_bert":
            model_name = "dccuchile/bert-base-spanish-wwm-cased"  # Mejor modelo BERT en español
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
        elif model_type == "roberta":
            model_name = "PlanTL-GOB-ES/roberta-base-bne"  # Mejor modelo RoBERTa en español
            self.model = RobertaModel.from_pretrained(model_name, return_dict=True)
            self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        elif model_type == "auto_roberta":
            model_name = "PlanTL-GOB-ES/roberta-base-bne"  # Mejor modelo RoBERTa en español
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
        elif model_type in ["rnn_lstm", "rnn_gru"]:
            # RNN con opción de cargar embeddings preentrenados
            if self.use_pretrained_embeddings:
                self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
            else:
                self.embedding = nn.Embedding(vocab_size, embedding_dim)

            if model_type == "rnn_lstm":
                self.model = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
            else:
                self.model = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
            self.classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, n_classes)



        # Congelar capas dependiendo de freeze_layers
        if model_type in ["bert", "roberta", "auto_bert", "auto_roberta"]:
            for name, param in self.model.named_parameters():
                if self.unfreeze_layers > 0 and not any(f'encoder.layer.{i}' in name for i in range(12 - self.unfreeze_layers, 12)):
                    param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, average='macro')
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, average='macro')
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, average='macro')
        self.train_precision = torchmetrics.Precision(task='multiclass', num_classes=n_classes, average='macro')
        self.valid_precision = torchmetrics.Precision(task='multiclass', num_classes=n_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task='multiclass', num_classes=n_classes, average='macro')
        self.train_recall = torchmetrics.Recall(task='multiclass', num_classes=n_classes, average='macro')
        self.valid_recall = torchmetrics.Recall(task='multiclass', num_classes=n_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task='multiclass', num_classes=n_classes, average='macro')
        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='macro')
        self.valid_f1 = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='macro')

    def forward(self, inputs, attention_mask=None, labels=None):
        if self.model_type in ["bert", "roberta"]:
            input_ids = inputs['input_ids']
            output = self.model(input_ids, attention_mask=attention_mask)
            logits = self.classifier(output.pooler_output)
        elif self.model_type in ["auto_bert", "auto_roberta"]:
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
        else:  # RNN models
            embedded = self.embedding(inputs)
            rnn_out, _ = self.model(embedded)
            if self.hparams.bidirectional:
                hidden = torch.cat((rnn_out[:, -1, :self.hparams.hidden_dim], rnn_out[:, 0, self.hparams.hidden_dim:]), dim=1)
            else:
                hidden = rnn_out[:, -1, :]
            logits = self.classifier(hidden)

        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        attention_mask = None
        if isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask", None)
            inputs = inputs["input_ids"]

        loss, outputs = self(inputs, attention_mask, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        if batch_idx == 0:
            self.last_train_preds = []
            self.last_train_labels = []
        self.last_train_preds.extend(preds.cpu().numpy())
        self.last_train_labels.extend(labels.cpu().numpy())

        self.train_acc(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)

        self.log("train_loss", loss, on_epoch=True, on_step=False, logger=True)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_precision", self.train_precision, on_epoch=True, on_step=False)
        self.log("train_recall", self.train_recall, on_epoch=True, on_step=False)
        self.log("train_f1", self.train_f1, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        attention_mask = None
        if isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask", None)
            inputs = inputs["input_ids"]

        loss, outputs = self(inputs, attention_mask, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        if batch_idx == 0:
            self.last_val_preds = []
            self.last_val_labels = []
        self.last_val_preds.extend(preds.cpu().numpy())
        self.last_val_labels.extend(labels.cpu().numpy())

        self.valid_acc(preds, labels)
        self.valid_precision(preds, labels)
        self.valid_recall(preds, labels)
        self.valid_f1(preds, labels)


        self.log("valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("valid_precision", self.valid_precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log("valid_recall", self.valid_recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log("valid_f1", self.valid_f1, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        attention_mask = None
        if isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask", None)
            inputs = inputs["input_ids"]

        loss, outputs = self(inputs, attention_mask, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.test_probs.extend(probs.cpu().numpy())
        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())

        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_precision", self.test_precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_recall", self.test_recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.model_type in ["bert", "roberta", "auto_bert", "auto_roberta"]:
            optimizer = AdamW(self.parameters(), lr=2e-5)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_training_steps
            )
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval='step'
                )
            )
        else:  # RNN models
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
            return optimizer

    def visualize_confusion_matrix(self, labels, predictions, all_categories, title):
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_categories, yticklabels=all_categories, annot_kws={"size": 70}, cbar=False)
        plt.title(title)
        plt.xlabel("Predicción")
        plt.ylabel("Etiqueta Verdadera")
        image_path = os.path.join(output_dir, f'{make_valid_filename(title)}_{make_valid_filename(self.model_type)}_{make_valid_filename(dataset_key)}.jpg')
        plt.savefig(image_path)
        plt.close()
        return image_path

    def visualize_roc_curve(self, labels, probabilities, title):
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        image_path = os.path.join(output_dir, f'{make_valid_filename(title)}_{make_valid_filename(self.model_type)}_{make_valid_filename(dataset_key)}.jpg')
        plt.savefig(image_path)
        plt.close()
        return image_path

    def visualize_precision_recall_curve(self, labels, probabilities, title):
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        average_precision = average_precision_score(labels, probabilities)
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (area = {average_precision:0.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        image_path = os.path.join(output_dir, f'{make_valid_filename(title)}_{make_valid_filename(self.model_type)}_{make_valid_filename(dataset_key)}.jpg')
        plt.savefig(image_path)
        plt.close()
        return image_path


    #def on_validation_end(self):
    #def on_train_end(self):
    #def on_fit_end(self):

    def on_test_end(self):
        self.train_image_path = self.visualize_confusion_matrix(self.last_train_labels, self.last_train_preds, list(range(self.n_classes)), "Matriz Confusión  - Entrenamiento")
        self.val_image_path = self.visualize_confusion_matrix(self.last_val_labels, self.last_val_preds, list(range(self.n_classes)), "Matriz Confusión  - Validación")
        self.test_image_path = self.visualize_confusion_matrix(self.test_labels, self.test_preds, list(range(self.n_classes)), "Matriz Confusión  - Test")

        # Generar las curvas ROC y precisión-recall para el conjunto de test
        probabilities = [prob[1] for prob in self.test_probs]
        self.test_roc_path = self.visualize_roc_curve(self.test_labels, probabilities, "Curva ROC - Test")
        self.test_prc_path = self.visualize_precision_recall_curve(self.test_labels, probabilities, "Curva Precision-Recall - Test")


# Cell 59
import time
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import RichProgressBar
import os
import base64
from IPython.display import HTML

output_dir = '/home/froh/FinalProject'

# Definición de los parámetros
BATCH_SIZE = 32
NUM_EPOCHS = 30
WARMUP_PARTICION = 6
NUM_WORKERS = 4

for PATIENCE in [10, 5]:
    print(f"Models Patience: {PATIENCE}")

    # Lista de modelos y datasets
    model_types = ['bert', 'roberta','rnn_lstm', 'rnn_gru', 'auto_bert', 'auto_roberta']
    datasets_keys = ['original_3886_118', 'undersampled_118_118', 'unbalanced_1000_118', 'balanced_500_sinonimos',
                 'balanced_500_insert_BERT', 'balanced_500_substitute_BERT', 'balanced_500_sentence_GPT2',
                 'balanced_500_sentence_prompt_GPT2', 'balanced_500_sentence_prompt_GPT4o_mini', 'balanced_1000_sinonimos',
                 'balanced_1000_insert_BERT', 'balanced_1000_substitute_BERT', 'balanced_1000_sentence_GPT2', 'balanced_1000_sentence_prompt_GPT2',
                 'balanced_1000_sentence_prompt_GPT4o_mini', 'balanced_2500_sinonimos', 'balanced_2500_insert_BERT', 'balanced_2500_substitute_BERT',
                 'balanced_2500_sentence_GPT2', 'balanced_2500_sentence_prompt_GPT2', 'balanced_2500_sentence_prompt_GPT4o_mini',
                 'balanced_total_sinonimos', 'balanced_total_insert_BERT', 'balanced_total_substitute_BERT', 'balanced_total_sentence_GPT2',
                 'balanced_total_sentence_prompt_GPT2', 'balanced_total_sentence_prompt_GPT4o_mini', 'unbalanced_3886_2500_sinonimos',
                 'unbalanced_3886_2500_insert_BERT', 'unbalanced_3886_2500_substitute_BERT', 'unbalanced_3886_2500_sentence_GPT2',
                 'unbalanced_3886_2500_sentence_prompt_GPT2', 'unbalanced_3886_2500_sentence_prompt_GPT4o_mini', 'total_augmented_data',
                 'balanced_AUG_GPT4o_mini', 'balanced_AUG_GPT4o_mini_total_augmented']


    # DataFrame para almacenar resultados
    results_df = []


    # Proceso de entrenamiento y evaluación para cada dataset y modelo
    for dataset_key in datasets_keys:
        df = datasets[dataset_key]  # Asume que datasets es un diccionario con los DataFrames

        for model_type in model_types:
            print(f"\nEntrenando modelo: {model_type} con dataset: {dataset_key}...\n")
            # Inicialización del DataModule
            data_module = PreguntaDataModule(dataframe=df, model_name=model_type, batch_size=BATCH_SIZE)

            # Calcular los pasos totales de entrenamiento y los pasos de calentamiento
            total_training_steps = len(df) // BATCH_SIZE * NUM_EPOCHS
            warmup_steps = total_training_steps // WARMUP_PARTICION

            # Inicializar el modelo
            model = TextClassifierBase(
                model_type=model_type,
                n_classes=2,
                n_warmup_steps=warmup_steps,
                n_training_steps=total_training_steps,
                hidden_dim=128,  # Agregar otros parámetros según el modelo
                n_layers=2,
                bidirectional=True,
                dropout=0.3,
                embedding_dim=data_module.embedding_dim,  # Pasar embedding_dim
                vocab_size=data_module.vocab_size  # Pasar vocab_size
            )

            # Callbacks para el entrenamiento
            #Para el ejercicio no se se considera crear checkpoints debido a la cantidad de modelos y almacenamiento requerido
            callback_check = ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_f1")
            callback_tqdm = RichProgressBar(leave=True)
            callback_early = EarlyStopping(monitor="valid_f1", mode="max", patience=PATIENCE)

            # Logger para registrar los resultados del entrenamiento
            logger = CSVLogger(save_dir=f"{output_dir}/logs/{dataset_key}/{model_type}/", name=f"{model_type}_{dataset_key}")

            # Inicializar el Trainer de PyTorch Lightning
            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                callbacks=[callback_tqdm, callback_early,callback_check], # no se considera
                accelerator="auto",  # Uses GPUs or TPUs if available
                devices="auto",  # Uses all available GPUs/TPUs if applicable
                logger=logger,
            )

            # Entrenamiento del modelo
            start_time = time.time()
            trainer.fit(model, data_module)
            runtime = (time.time() - start_time) / 60
            print(f"Tiempo de entrenamiento para {model_type} con {dataset_key} en minutos: {runtime:.2f}")

            # Curvas de aprendizaje del modelo
            metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

            aggreg_metrics = []
            agg_col = "epoch"
            for i, dfg in metrics.groupby(agg_col):
                agg = dict(dfg.mean())
                agg[agg_col] = i
                aggreg_metrics.append(agg)

            df_metrics = pd.DataFrame(aggreg_metrics)

            # Guardar las curvas de aprendizaje para cada métrica
            metrics_to_plot = {
                ("train_loss", "valid_loss"): "Loss",
                ("train_acc", "valid_acc"): "Accuracy",
                ("train_precision", "valid_precision"): "Precision",
                ("train_recall", "valid_recall"): "Recall",
                ("train_f1", "valid_f1"): "F1 Score"
            }
            learning_curves_paths = {}

            for (train_metric, val_metric), ylabel in metrics_to_plot.items():
                if train_metric in df_metrics.columns and val_metric in df_metrics.columns:
                    df_metrics[[train_metric, val_metric]].plot(grid=True, legend=True, xlabel="Epoch", ylabel=ylabel)
                    learning_curve_path = os.path.join(output_dir, f'learningcurve_{make_valid_filename(model_type)}_{make_valid_filename(dataset_key)}_{make_valid_filename(train_metric)}_{make_valid_filename(val_metric)}.png')
                    plt.savefig(learning_curve_path)
                    plt.show()
                    plt.close()
                    learning_curves_paths[f"{train_metric}_{val_metric}"] = learning_curve_path

            # Evaluar el conjunto de test
            test_results = trainer.test(model=model, datamodule=data_module, ckpt_path='best')  # cargamos el mejor checkpoint del modelo


            # Guardar las matrices de confusión y curvas
            train_conf_matrix_path = model.train_image_path
            val_conf_matrix_path = model.val_image_path
            test_conf_matrix_path = model.test_image_path
            test_roc_path = model.test_roc_path
            test_prc_path = model.test_prc_path

            # Agregar resultados al DataFrame
            result_df = {
                'dataset': dataset_key,
                'model': model_type,
                'runtime_minutes': runtime,
                'patience': PATIENCE,
                'test_f1': test_results[0]['test_f1'],
                'test_conf_matrix': test_conf_matrix_path,
                'test_roc_curve': test_roc_path,
                'test_prc_curve': test_prc_path,
                'learning_curves_loss': learning_curves_paths.get("train_loss_valid_loss"),
                'learning_curves_f1': learning_curves_paths.get("train_f1_valid_f1"),
                'train_conf_matrix': train_conf_matrix_path,
                'val_conf_matrix': val_conf_matrix_path,
                'learning_curves_acc': learning_curves_paths.get("train_acc_valid_acc"),
                'learning_curves_precision': learning_curves_paths.get("train_precision_valid_precision"),
                'learning_curves_recall': learning_curves_paths.get("train_recall_valid_recall"),
                'test_acc': test_results[0]['test_acc'],
                'test_precision': test_results[0]['test_precision'],
                'test_recall': test_results[0]['test_recall'],
                'test_loss': test_results[0]['test_loss'],
                'valid_acc': df_metrics["valid_acc"].iloc[-1] if "valid_acc" in df_metrics.columns else None,
                'valid_precision': df_metrics["valid_precision"].iloc[-1] if "valid_precision" in df_metrics.columns else None,
                'valid_recall': df_metrics["valid_recall"].iloc[-1] if "valid_recall" in df_metrics.columns else None,
                'valid_f1': df_metrics["valid_f1"].iloc[-1] if "valid_f1" in df_metrics.columns else None,
                'valid_loss': df_metrics["valid_loss"].iloc[-1] if "valid_loss" in df_metrics.columns else None,
                'train_acc': df_metrics["train_acc"].iloc[-1] if "train_acc" in df_metrics.columns else None,
                'train_precision': df_metrics["train_precision"].iloc[-1] if "train_precision" in df_metrics.columns else None,
                'train_recall': df_metrics["train_recall"].iloc[-1] if "train_recall" in df_metrics.columns else None,
                'train_f1': df_metrics["train_f1"].iloc[-1] if "train_f1" in df_metrics.columns else None,
                'train_loss': df_metrics["train_loss"].iloc[-1] if "train_loss" in df_metrics.columns else None,
            }
            results_df.append(result_df)

# Guardar resultados en un archivo CSV
results_df = pd.DataFrame(results_df)
results_df.to_csv(f'{output_dir}/results_summary.csv', index=False)

# Cell 61
import time
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import RichProgressBar
import os
import base64
from IPython.display import HTML

output_dir = '/home/froh/FinalProject_Freezed_models_5p'

#Verifica directorios
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Definición de los parámetros
BATCH_SIZE = 128
NUM_EPOCHS = 30
WARMUP_PARTICION = 6
PATIENCE = 5
NUM_WORKERS = 4

UNFREEZEDL = [4, 2, 0]

# Lista de modelos y datasets
model_types = ['bert', 'roberta','auto_bert', 'auto_roberta']
datasets_keys = ['original_3886_118', 'undersampled_118_118', 'unbalanced_1000_118', 'balanced_500_sinonimos',
                 'balanced_500_insert_BERT', 'balanced_500_substitute_BERT', 'balanced_500_sentence_GPT2',
                 'balanced_500_sentence_prompt_GPT2', 'balanced_500_sentence_prompt_GPT4o_mini', 'balanced_1000_sinonimos',
                 'balanced_1000_insert_BERT', 'balanced_1000_substitute_BERT', 'balanced_1000_sentence_GPT2', 'balanced_1000_sentence_prompt_GPT2',
                 'balanced_1000_sentence_prompt_GPT4o_mini', 'balanced_2500_sinonimos', 'balanced_2500_insert_BERT', 'balanced_2500_substitute_BERT',
                 'balanced_2500_sentence_GPT2', 'balanced_2500_sentence_prompt_GPT2', 'balanced_2500_sentence_prompt_GPT4o_mini',
                 'balanced_total_sinonimos', 'balanced_total_insert_BERT', 'balanced_total_substitute_BERT', 'balanced_total_sentence_GPT2',
                 'balanced_total_sentence_prompt_GPT2', 'balanced_total_sentence_prompt_GPT4o_mini', 'unbalanced_3886_2500_sinonimos',
                 'unbalanced_3886_2500_insert_BERT', 'unbalanced_3886_2500_substitute_BERT', 'unbalanced_3886_2500_sentence_GPT2',
                 'unbalanced_3886_2500_sentence_prompt_GPT2', 'unbalanced_3886_2500_sentence_prompt_GPT4o_mini', 'total_augmented_data',
                 'balanced_AUG_GPT4o_mini', 'balanced_AUG_GPT4o_mini_total_augmented']


# DataFrame para almacenar resultados
results_df = []

for UNFREEZED in UNFREEZEDL:
    # Proceso de entrenamiento y evaluación para cada dataset y modelo
    for dataset_key in datasets_keys:
        df = datasets[dataset_key]  # Asume que datasets es un diccionario con los DataFrames

        for model_type in model_types:
            print(f"\nEntrenando modelo: {model_type}, con {12-UNFREEZED} capas congeladas, con dataset: {dataset_key}...\n")
            # Inicialización del DataModule
            data_module = PreguntaDataModule(dataframe=df, model_name=model_type, batch_size=BATCH_SIZE)

            # Calcular los pasos totales de entrenamiento y los pasos de calentamiento
            total_training_steps = len(df) // BATCH_SIZE * NUM_EPOCHS
            warmup_steps = total_training_steps // WARMUP_PARTICION

            # Inicializar el modelo
            model = TextClassifierBase(
                model_type=model_type,
                n_classes=2,
                n_warmup_steps=warmup_steps,
                n_training_steps=total_training_steps,
                hidden_dim=128,  # Agregar otros parámetros según el modelo
                n_layers=2,
                bidirectional=True,
                dropout=0.3,
                embedding_dim=data_module.embedding_dim,  # Pasar embedding_dim
                vocab_size=data_module.vocab_size,  # Pasar vocab_size
                unfreeze_layers=UNFREEZED
            )

            # Callbacks para el entrenamiento
            #Para el ejercicio no se se considera crear checkpoints debido a la cantidad de modelos y almacenamiento requerido
            callback_check = ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_f1")
            callback_tqdm = RichProgressBar(leave=True)
            callback_early = EarlyStopping(monitor="valid_f1", mode="max", patience=PATIENCE)

            # Logger para registrar los resultados del entrenamiento
            logger = CSVLogger(save_dir=f"{output_dir}/logs/{dataset_key}/{model_type}_{12-UNFREEZED}_f_layers/", name=f"{model_type}_{dataset_key}_{12-UNFREEZED}_f_layers")

            # Inicializar el Trainer de PyTorch Lightning
            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                callbacks=[callback_tqdm, callback_early,callback_check], # no se considera
                accelerator="gpu",  # Uses GPUs or TPUs if available
                devices=[1],  # Uses all available GPUs/TPUs if applicable
                logger=logger,
            )

            # Entrenamiento del modelo
            start_time = time.time()
            trainer.fit(model, data_module)
            runtime = (time.time() - start_time) / 60
            print(f"Tiempo de entrenamiento para {model_type} con {dataset_key} en minutos: {runtime:.2f}")

            # Curvas de aprendizaje del modelo
            metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

            aggreg_metrics = []
            agg_col = "epoch"
            for i, dfg in metrics.groupby(agg_col):
                agg = dict(dfg.mean())
                agg[agg_col] = i
                aggreg_metrics.append(agg)

            df_metrics = pd.DataFrame(aggreg_metrics)

            # Guardar las curvas de aprendizaje para cada métrica
            metrics_to_plot = {
                ("train_loss", "valid_loss"): "Loss",
                ("train_acc", "valid_acc"): "Accuracy",
                ("train_precision", "valid_precision"): "Precision",
                ("train_recall", "valid_recall"): "Recall",
                ("train_f1", "valid_f1"): "F1 Score"
            }
            learning_curves_paths = {}

            for (train_metric, val_metric), ylabel in metrics_to_plot.items():
                if train_metric in df_metrics.columns and val_metric in df_metrics.columns:
                    df_metrics[[train_metric, val_metric]].plot(grid=True, legend=True, xlabel="Epoch", ylabel=ylabel)
                    learning_curve_path = os.path.join(output_dir, f'learningcurve_{make_valid_filename(model_type)}_{make_valid_filename(dataset_key)}_{make_valid_filename(train_metric)}_{make_valid_filename(val_metric)}.png')
                    plt.savefig(learning_curve_path)
                    plt.show()
                    plt.close()
                    learning_curves_paths[f"{train_metric}_{val_metric}"] = learning_curve_path

            # Evaluar el conjunto de test
            test_results = trainer.test(model=model, datamodule=data_module, ckpt_path='best')  # cargamos el mejor checkpoint del modelo


            # Guardar las matrices de confusión y curvas
            train_conf_matrix_path = model.train_image_path
            val_conf_matrix_path = model.val_image_path
            test_conf_matrix_path = model.test_image_path
            test_roc_path = model.test_roc_path
            test_prc_path = model.test_prc_path

            # Agregar resultados al DataFrame
            result_df = {
                'dataset': dataset_key,
                'model': model_type,
                'freezed_layers': (12-UNFREEZED),
                'patience': PATIENCE,
                'runtime_minutes': runtime,
                'test_f1': test_results[0]['test_f1'],
                'test_conf_matrix': test_conf_matrix_path,
                'test_roc_curve': test_roc_path,
                'test_prc_curve': test_prc_path,
                'learning_curves_loss': learning_curves_paths.get("train_loss_valid_loss"),
                'learning_curves_f1': learning_curves_paths.get("train_f1_valid_f1"),
                'train_conf_matrix': train_conf_matrix_path,
                'val_conf_matrix': val_conf_matrix_path,
                'learning_curves_acc': learning_curves_paths.get("train_acc_valid_acc"),
                'learning_curves_precision': learning_curves_paths.get("train_precision_valid_precision"),
                'learning_curves_recall': learning_curves_paths.get("train_recall_valid_recall"),
                'test_acc': test_results[0]['test_acc'],
                'test_precision': test_results[0]['test_precision'],
                'test_recall': test_results[0]['test_recall'],
                'test_loss': test_results[0]['test_loss'],
                'valid_acc': df_metrics["valid_acc"].iloc[-1] if "valid_acc" in df_metrics.columns else None,
                'valid_precision': df_metrics["valid_precision"].iloc[-1] if "valid_precision" in df_metrics.columns else None,
                'valid_recall': df_metrics["valid_recall"].iloc[-1] if "valid_recall" in df_metrics.columns else None,
                'valid_f1': df_metrics["valid_f1"].iloc[-1] if "valid_f1" in df_metrics.columns else None,
                'valid_loss': df_metrics["valid_loss"].iloc[-1] if "valid_loss" in df_metrics.columns else None,
                'train_acc': df_metrics["train_acc"].iloc[-1] if "train_acc" in df_metrics.columns else None,
                'train_precision': df_metrics["train_precision"].iloc[-1] if "train_precision" in df_metrics.columns else None,
                'train_recall': df_metrics["train_recall"].iloc[-1] if "train_recall" in df_metrics.columns else None,
                'train_f1': df_metrics["train_f1"].iloc[-1] if "train_f1" in df_metrics.columns else None,
                'train_loss': df_metrics["train_loss"].iloc[-1] if "train_loss" in df_metrics.columns else None,
            }
            results_df.append(result_df)


# Guardar resultados en un archivo CSV
results_df = pd.DataFrame(results_df)
results_df.to_csv(f'{output_dir}/results_summary.csv', index=False)

# Cell 65
import easyocr
from PIL import Image, ImageEnhance
import numpy as np

# Cargar los datasets de resultados
resultados_redes = pd.read_csv('/home/froh/FinalProject/results_summary.csv')
resultados_redes_aug = pd.read_csv('/home/froh/FinalProject_AUG/results_summary_aug.csv')
resultados_redes_aug_total = pd.read_csv('/home/froh/FinalProject_AUG_total/results_summary_aug_total.csv')
resultados_redes_autoc_aug_total = pd.read_csv('/home/froh/FinalProject_AutoC_AUGtotal/results_summary_aug_total.csv')

res_red=pd.concat([resultados_redes, resultados_redes_aug, resultados_redes_aug_total, resultados_redes_autoc_aug_total], ignore_index=True) #resultados_redes_freezed


resultados_redes_p5 = pd.read_csv('/home/froh/FinalProject_p5/results_summary_p5.csv')
resultados_redes_aug_5p = pd.read_csv('/home/froh/FinalProject_AUG_5p/results_summary_aug_5p.csv')
resultados_redes_aug_total_5p = pd.read_csv('/home/froh/FinalProject_AUG_total_5p/results_summary_aug_total_5p.csv')
resultados_redes_autoc_aug_total_5p = pd.read_csv('/home/froh/FinalProject_AutoC_AUGtotal_5p/results_summary_aug_total_5p.csv')
resultados_redes_freezed_5p = pd.read_csv('/home/froh/FinalProject_Freezed_models_5p/results_summary1.csv')

res_red_5p=pd.concat([resultados_redes_p5, resultados_redes_aug_5p, resultados_redes_aug_total_5p, resultados_redes_autoc_aug_total_5p, resultados_redes_freezed_5p], ignore_index=True)


res_redes=pd.concat([res_red, res_red_5p], ignore_index=True)
res_redes=res_redes.rename(columns={'valid_acc': 'val_acc', 'valid_precision': 'val_precision', 'valid_recall': 'val_recall', 'valid_f1': 'val_f1', 'valid_loss': 'val_loss'})

# Definir una función para extraer el número de capas congeladas
def extract_freezed_layers(model_name):
    if ' - Freezed 8 layers' in model_name:
        return 8
    elif ' - Freezed 10 layers' in model_name:
        return 10
    elif ' - Freezed 12 layers' in model_name:
        return 12
    else:
        return 0

# Crear la columna 'freezed_layers'
res_redes['freezed_layers'] = res_redes['model'].apply(extract_freezed_layers)
# Eliminar registros donde 'freezed_layers' es 12
res_redes = res_redes[res_redes['freezed_layers'] != 12]
# Eliminar textos específicos de la columna 'model'
texts_to_remove = [' - Freezed 8 layers', ' - Freezed 10 layers', ' - Freezed 12 layers']
for text in texts_to_remove:
    #res_redes['model'] = res_redes['model'].str.replace(text, '', regex=False)
    res_redes.loc[:, 'model'] = res_redes['model'].str.replace(text, '', regex=False)


resultados_cl = pd.read_csv('/home/froh/FinalProject_cl/results_summary_cl.csv')
resultados_aug_cl = pd.read_csv('/home/froh/FinalProject_AUG_cl/results_summary_AUG_cl.csv')
resultados_aug_total_cl = pd.read_csv('/home/froh/FinalProject_AUG_total_cl/results_summary_AUG_total_cl.csv')
resultados_svc_cl = pd.read_csv('/home/froh/FinalProject_SVC_cl/results_summary_SVC_cl.csv')
resultados_sbert_cl = pd.read_csv('/home/froh/FinalProject_SBERT_AUG_total_cl/results_summary_SBERT_AUG_cl.csv')
resultados_sbert_pre_cl = pd.read_csv('/home/froh/FinalProject_SBERT_AUG_total_cl_PRE/results_summary_SBERT_AUG_cl.csv')
resultados_sbert_pre_cl['embedding'] = 'prepr_SBERT'

res_cl=pd.concat([resultados_cl, resultados_aug_cl,resultados_aug_total_cl, resultados_svc_cl, resultados_sbert_cl, resultados_sbert_pre_cl], ignore_index=True)

res_cl=res_cl.rename(columns={'runtime_minutes': 'grid_runtime_minutes', 'avg_runtime_minutes': 'runtime_minutes', 'learning_curve': 'learning_curves_f1'})

results_df = pd.concat([res_redes, res_cl], ignore_index=True)

results_df.drop_duplicates(subset=['dataset', 'model', 'Patience', 'freezed_layers', 'embedding'], inplace=True)


# Inicializar el lector de EasyOCR
reader = easyocr.Reader(['en'])

# Definir las coordenadas basadas en el tamaño de la imagen para incluir todo el cuadrante
tn_coords = (227, 167, 409, 250)  # Coordenadas para TN
fp_coords = (600, 167, 800, 250)  # Coordenadas para FP
fn_coords = (227, 440, 409, 519)  # Coordenadas para FN
tp_coords = (600, 440, 800, 519)  # Coordenadas para TP

from PIL import ImageFilter, ImageOps

def extract_number_from_image_easyocr(image_path, coords, gray=12, radius=3, scale_factor=1):
    try:
        # Cargar y recortar la imagen
        image = Image.open(image_path)
        cropped_image = image.crop(coords)

        # Escalar el recorte para amplificarlo
        if scale_factor != 1.0:
            new_width = int(cropped_image.width * scale_factor)
            new_height = int(cropped_image.height * scale_factor)
            cropped_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)

        # Convertir la imagen a escala de grises
        gray_image = cropped_image.convert('L')

        # Aumentar el contraste
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(gray)  # Ajuste el factor de contraste si es necesario

        # Aplicar desenfoque gaussiano
        blurred_image = enhanced_image.filter(ImageFilter.GaussianBlur(radius=radius))

        # Convertir la imagen mejorada a un array de numpy para EasyOCR
        cropped_image_np = np.array(blurred_image)

        # Usar EasyOCR para extraer el texto
        results = reader.readtext(cropped_image_np, detail=0)

        # Si hay algún resultado, devolver el primer valor
        if results:
            text = results[0].strip()
            return text #if text.isdigit() else None
        else:
            return None
    except Exception as e:
        return None


# Aplicar la extracción de los valores de cada cuadrante usando las coordenadas
results_df['TP'] = results_df['test_conf_matrix'].apply(lambda x: extract_number_from_image_easyocr(x, tp_coords)).astype(int)
results_df['FN'] = results_df['test_conf_matrix'].apply(lambda x: extract_number_from_image_easyocr(x, fn_coords)).astype(int)
results_df['FP'] = results_df['test_conf_matrix'].apply(lambda x: extract_number_from_image_easyocr(x, fp_coords)).astype(int)
results_df['TN'] = results_df['test_conf_matrix'].apply(lambda x: extract_number_from_image_easyocr(x, tn_coords)).astype(int)

# Calcular métricas para la clase positiva
results_df['recall_positive'] = results_df['TP'] / (results_df['TP'] + results_df['FN'])
results_df['precision_positive'] = results_df['TP'] / (results_df['TP'] + results_df['FP'])
results_df['f1_positive'] = 2 * (results_df['precision_positive'] * results_df['recall_positive']) / (results_df['precision_positive'] + results_df['recall_positive'])

# Calcular métricas para la clase negativa
results_df['recall_negative'] = results_df['TN'] / (results_df['TN'] + results_df['FP'])
results_df['precision_negative'] = results_df['TN'] / (results_df['TN'] + results_df['FN'])
results_df['f1_negative'] = 2 * (results_df['precision_negative'] * results_df['recall_negative']) / (results_df['precision_negative'] + results_df['recall_negative'])

# Cálculos de métricas macro (promedio simple entre clases)
results_df['test_macro_recall'] = (results_df['recall_positive'] + results_df['recall_negative']) / 2
results_df['test_macro_f1'] = (results_df['f1_positive'] + results_df['f1_negative']) / 2

# Calcular el número de ejemplos positivos y negativos
support_positive = results_df['TP'] + results_df['FN']
support_negative = results_df['TN'] + results_df['FP']
total_support = support_positive + support_negative

# Cálculos de métricas weighted (promedio ponderado por el número de ejemplos en cada clase)
results_df['test_weighted_recall'] = (results_df['recall_positive'] * support_positive + results_df['recall_negative'] * support_negative) / total_support
results_df['test_weighted_f1'] = (results_df['f1_positive'] * support_positive + results_df['f1_negative'] * support_negative) / total_support



# Definir las coordenadas basadas en el tamaño de la imagen para incluir todo el cuadrante ROC
roc_coords = (521, 395, 559, 418)  # Coordenadas para ROC

# Aplicar la extracción de los valores de cada cuadrante usando las coordenadas
results_df['ROC_area'] = results_df['test_roc_curve'].apply(lambda x: extract_number_from_image_easyocr(x, roc_coords, gray=12, radius=0, scale_factor=2)).astype(float)

nan_count = results_df['TP'].isna().sum()
print(f"Número de NaN en la columna 'TP': {nan_count}")
nan_count = results_df['FN'].isna().sum()
print(f"Número de NaN en la columna 'FN': {nan_count}")
nan_count = results_df['FP'].isna().sum()
print(f"Número de NaN en la columna 'FP': {nan_count}")
nan_count = results_df['TN'].isna().sum()
print(f"Número de NaN en la columna 'TN': {nan_count}")
nan_count = results_df['ROC_area'].isna().sum()
print(f"Número de NaN en la columna 'ROC_area': {nan_count}")


results_df['freezed_layers'].fillna(0, inplace=True)

# Definir la tabla de mapeo
mapeo = {
    'balanced_1000_insert_BERT': ['04 insert BERT', '05. 1000 - 1000', '03. balanced'],
    'balanced_1000_sentence_GPT2': ['05 sentence GPT2', '05. 1000 - 1000', '03. balanced'],
    'balanced_1000_sentence_prompt_GPT2': ['06 sentence prompt GPT2', '05. 1000 - 1000', '03. balanced'],
    'balanced_1000_sentence_prompt_GPT4o_mini': ['07 sentence prompt GPT4o mini', '05. 1000 - 1000', '03. balanced'],
    'balanced_1000_sinonimos': ['02 sinónimos', '05. 1000 - 1000', '03. balanced'],
    'balanced_1000_substitute_BERT': ['03 substitute BERT', '05. 1000 - 1000', '03. balanced'],
    'balanced_2500_insert_BERT': ['04 insert BERT', '06. 2500 - 2500', '03. balanced'],
    'balanced_2500_sentence_GPT2': ['05 sentence GPT2', '06. 2500 - 2500', '03. balanced'],
    'balanced_2500_sentence_prompt_GPT2': ['06 sentence prompt GPT2', '06. 2500 - 2500', '03. balanced'],
    'balanced_2500_sentence_prompt_GPT4o_mini': ['07 sentence prompt GPT4o mini', '06. 2500 - 2500', '03. balanced'],
    'balanced_2500_sinonimos': ['02 sinónimos', '06. 2500 - 2500', '03. balanced'],
    'balanced_2500_substitute_BERT': ['03 substitute BERT', '06. 2500 - 2500', '03. balanced'],
    'balanced_500_insert_BERT': ['04 insert BERT', '04. 500 - 500', '03. balanced'],
    'balanced_500_sentence_GPT2': ['05 sentence GPT2', '04. 500 - 500', '03. balanced'],
    'balanced_500_sentence_prompt_GPT2': ['06 sentence prompt GPT2', '04. 500 - 500', '03. balanced'],
    'balanced_500_sentence_prompt_GPT4o_mini': ['07 sentence prompt GPT4o mini', '04. 500 - 500', '03. balanced'],
    'balanced_500_sinonimos': ['02 sinónimos', '04. 500 - 500', '03. balanced'],
    'balanced_500_substitute_BERT': ['03 substitute BERT', '04. 500 - 500', '03. balanced'],
    'balanced_total_insert_BERT': ['04 insert BERT', '07. 3785 - 3879', '03. balanced'],
    'balanced_total_sentence_prompt_GPT2': ['06 sentence prompt GPT2', '07. 3785 - 3879', '03. balanced'],
    'balanced_total_sentence_prompt_GPT4o_mini': ['07 sentence prompt GPT4o mini', '07. 3785 - 3879', '03. balanced'],
    'balanced_total_sinonimos': ['02 sinónimos', '07. 3785 - 3879', '03. balanced'],
    'balanced_total_substitute_BERT': ['03 substitute BERT', '07. 3785 - 3879', '03. balanced'],
    'balanced_total_sentence_GPT2': ['05 sentence GPT2', '07. 3785 - 3879', '03. balanced'],
    'original_3886_118': ['00 original data', '00. 3785 - 117', '00. original'],
    'unbalanced_1000_118': ['00 original data', '02. 1000 - 117', '02. unbalanced - Class 0'],
    'unbalanced_3886_2500_insert_BERT': ['04 insert BERT', '03. 3785 - 2500', '02. unbalanced - Class 0'],
    'unbalanced_3886_2500_sentence_GPT2': ['05 sentence GPT2', '03. 3785 - 2500', '02. unbalanced - Class 0'],
    'unbalanced_3886_2500_sentence_prompt_GPT2': ['06 sentence prompt GPT2', '03. 3785 - 2500', '02. unbalanced - Class 0'],
    'unbalanced_3886_2500_sentence_prompt_GPT4o_mini': ['07 sentence prompt GPT4o mini', '03. 3785 - 2500', '02. unbalanced - Class 0'],
    'unbalanced_3886_2500_sinonimos': ['02 sinónimos', '03. 3785 - 2500', '02. unbalanced - Class 0'],
    'unbalanced_3886_2500_substitute_BERT': ['03 substitute BERT', '03. 3785 - 2500', '02. unbalanced - Class 0'],
    'undersampled_118_118': ['01 undersampling', '01. 117 - 117', '01. undersampling'],
    'balanced_AUG_GPT4o_mini': ['08 augmented prompt GPT4o mini', '08. 3785 - 5145', '04. unbalanced  - Class 1'],
    'total_augmented_data': ['09 All augmented data', '09. 3785 - 21800', '04. unbalanced  - Class 1'],
    'balanced_AUG_GPT4o_mini_total_augmented': ['10 All augmented + aug gtp4 data', '10. 3785 - 26460', '04. unbalanced  - Class 1']
}

# Crear las tres nuevas columnas basadas en el dataset
results_df[['data_augmentation', 'tamaño_dataset', 'balance']] = results_df['dataset'].map(mapeo).apply(pd.Series)

print(results_df.shape)
results_df.to_csv('/home/froh/results_final_summary_modelos.csv', index=False)
print("Resultados guardados en 'results_final_summary_modelos.csv'")

results_df.head()

# Cell 67
import os
import base64
from IPython.display import HTML

# Función para convertir el DataFrame a HTML con imágenes embebidas
def render_df_with_images(df, image_columns):
    df_copy = df.copy()

    # Asegurarse de que image_columns sea una lista
    if isinstance(image_columns, str):
        image_columns = [image_columns]

    for idx, row in df_copy.iterrows():
        for image_column in image_columns:
            image_path = row[image_column]
            if pd.notna(image_path) and image_path and os.path.exists(image_path):  # Verificar que no sea NaN, no sea None y exista
                img_html = f'<img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="150" height="100">'
                df_copy.at[idx, image_column] = img_html
            elif pd.isna(image_path):  # Si es NaN, dejar el campo en blanco
                df_copy.at[idx, image_column] = ''
            else:
                df_copy.at[idx, image_column] = 'Image not found'

    return HTML(df_copy.to_html(escape=False))

# Cell 68
path = '/home/froh/results_final_summary_modelos.csv'
results_df = pd.read_csv(path)

# Lista con el nuevo orden de las columnas
column_order = [
    'dataset', 'model', 'Patience', 'embedding', 'freezed_layers', 'runtime_minutes',
    'test_f1', 'test_recall', 'test_conf_matrix', 'TP', 'FP', 'FN', 'TN', 'test_roc_curve', 'test_prc_curve',
    'learning_curves_f1', 'learning_curves_recall', 'learning_curves_loss', 'test_acc', 'test_precision',
    'train_conf_matrix', 'val_conf_matrix', 'learning_curves_acc',
    'learning_curves_precision', 'test_loss', 'val_acc', 'val_precision',
    'val_recall', 'val_f1', 'val_loss', 'train_acc', 'train_precision',
    'train_recall', 'train_f1', 'train_loss', 'grid_runtime_minutes',
    'num_combinations', 'val_roc_curve', 'val_prc_curve',
    'train_roc_curve', 'train_prc_curve', 'best_params'
]

# Reordenar las columnas del DataFrame
results_df = results_df.reindex(columns=column_order)

# Ordenar el DataFrame de resultados por la columna 'test_f1' de manera descendente
results_df_sorted = results_df.sort_values(by='test_f1', ascending=False)

# Especificar las columnas que contienen rutas de imágenes
image_columns = ['test_conf_matrix', 'test_roc_curve', 'test_prc_curve',
                'learning_curves_f1', 'learning_curves_recall', 'learning_curves_loss'
                ]
# Renderizar el DataFrame con imágenes embebidas
html_output = render_df_with_images(results_df_sorted[0:150], image_columns)

# Mostrar el DataFrame ordenado
print(f"************************ TOP MODELOS OPTIMIZADOS ************************\nModelos calculados: {len(results_df_sorted)}\nOrdenado por: F1 score")
# Mostrar el HTML generado
display(html_output)

# Cell 69
path = '/home/froh/results_final_summary_modelos.csv'
results_df = pd.read_csv(path)

# Lista con el nuevo orden de las columnas
column_order = [
    'dataset', 'model', 'Patience', 'embedding', 'freezed_layers', 'runtime_minutes',
    'test_f1', 'test_recall', 'test_conf_matrix', 'TP', 'FP', 'FN', 'TN', 'test_roc_curve', 'test_prc_curve',
    'learning_curves_f1', 'learning_curves_recall', 'learning_curves_loss', 'test_acc', 'test_precision',
    'train_conf_matrix', 'val_conf_matrix', 'learning_curves_acc',
    'learning_curves_precision', 'test_loss', 'val_acc', 'val_precision',
    'val_recall', 'val_f1', 'val_loss', 'train_acc', 'train_precision',
    'train_recall', 'train_f1', 'train_loss', 'grid_runtime_minutes',
    'num_combinations', 'val_roc_curve', 'val_prc_curve',
    'train_roc_curve', 'train_prc_curve', 'best_params'
]

# Reordenar las columnas del DataFrame
results_df = results_df.reindex(columns=column_order)

# Ordenar el DataFrame de resultados por la columna 'test_recall' de manera descendente
results_df_sorted = results_df.sort_values(by='test_recall', ascending=False)

# Especificar las columnas que contienen rutas de imágenes
image_columns = ['test_conf_matrix', 'test_roc_curve', 'test_prc_curve',
                'learning_curves_f1', 'learning_curves_recall', 'learning_curves_loss'
                ]
# Renderizar el DataFrame con imágenes embebidas
html_output = render_df_with_images(results_df_sorted[0:150], image_columns)

# Mostrar el DataFrame ordenado
print(f"************************ TOP MODELOS OPTIMIZADOS ************************\nModelos calculados: {len(results_df_sorted)}\nOrdenado por: Recall")
# Mostrar el HTML generado
display(html_output)

# Cell 71
#data = pd.read_csv('/home/froh/results_final_summary_modelos.csv')
data = pd.read_csv('results_final_summary_modelos.csv')
data['text representation'] = data['embedding']
df = data[data['embedding'].isna()]
df_em = data[~data['embedding'].isna()]
df.head()

# Cell 73
# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Patience
plt.figure(figsize=(14, 6))

# Gráfico de caja y bigote con outliers
plt.subplot(2, 2, 1)
sns.boxplot(x=df['Patience'], y=df['f1_positive'], data=df, showfliers=True)
plt.title('Distribución de Binary F1 Score por Patience')

# Gráfico de caja y bigote sin outliers
plt.subplot(2, 2, 2)
sns.boxplot(x=df['Patience'], y=df['recall_positive'], data=df, showfliers=False)
plt.title('Distribución de Binary Recall por Patience')

# Gráfico de caja y bigote sin outliers
plt.subplot(2, 2, 3)
sns.boxplot(x=df['Patience'], y=df['runtime_minutes'], data=df, showfliers=True)
plt.title('Distribución de tiempo de ejecución por Modelo NN y Patience (Con Outliers)')

# Gráfico de caja y bigote sin outliers
plt.subplot(2, 2, 4)
sns.boxplot(x=df['Patience'], y=df['runtime_minutes'], data=df, showfliers=False)
plt.title('Distribución de tiempo de ejecución por Modelo NN y Patience (Sin Outliers)')


# Colocar un título general (supertítulo)
plt.suptitle('Comparación métricas por Patience en modelos NN', fontsize=16, y=0.98)

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 75
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, shapiro

# Cargar el archivo CSV
file_path = '/home/froh/results_final_summary_modelos.csv'
data = pd.read_csv(file_path)

# Parte 1: Prueba de hipótesis comparando f1_positive entre Patience = 10 y Patience = 5, filtrando por freezed_layers = 0
# Filtramos los datos con Patience = 5 y Patience = 10, y con freezed_layers = 0
patience_5 = data[(data['Patience'] == 5) & (data['freezed_layers'] == 0)]
patience_10 = data[(data['Patience'] == 10) & (data['freezed_layers'] == 0)]

# Unimos ambos conjuntos de datos basándonos en el modelo y dataset para comparar f1_positive
merged_data_p5_p10 = pd.merge(patience_5, patience_10, on=['model', 'dataset'], suffixes=('_p5', '_p10'))

# Eliminamos valores nulos en f1_positive antes de la prueba de hipótesis
clean_data_p5_p10 = merged_data_p5_p10.dropna(subset=['f1_positive_p5', 'f1_positive_p10'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_p5_p10.empty:
    differences = clean_data_p5_p10['f1_positive_p5'] - clean_data_p5_p10['f1_positive_p10']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_p5_p10, p_value_p5_p10 = ttest_rel(clean_data_p5_p10['f1_positive_p5'], clean_data_p5_p10['f1_positive_p10'])
        print(f"Resultados de la prueba t pareada para Patience 5 vs 10: t_statistic = {t_statistic_p5_p10}, p_value = {p_value_p5_p10}")
        if p_value_p5_p10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Patience 5 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Patience 5 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_p5_p10, p_value_p5_p10 = wilcoxon(clean_data_p5_p10['f1_positive_p5'], clean_data_p5_p10['f1_positive_p10'])
        print(f"Resultados de la prueba de Wilcoxon para Patience 5 vs 10: w_statistic = {w_statistic_p5_p10}, p_value = {p_value_p5_p10}")
        if p_value_p5_p10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Patience 5 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Patience 5 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Patience 5 y 10\n----------------------------------------------------\n")



# Eliminamos valores nulos en recall_positive antes de la prueba de hipótesis
clean_data_p5_p10 = merged_data_p5_p10.dropna(subset=['recall_positive_p5', 'recall_positive_p10'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_p5_p10.empty:
    differences = clean_data_p5_p10['recall_positive_p5'] - clean_data_p5_p10['recall_positive_p10']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_p5_p10, p_value_p5_p10 = ttest_rel(clean_data_p5_p10['recall_positive_p5'], clean_data_p5_p10['recall_positive_p10'])
        print(f"Resultados de la prueba t pareada para Patience 5 vs 10: t_statistic = {t_statistic_p5_p10}, p_value = {p_value_p5_p10}")
        if p_value_p5_p10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Patience 5 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Patience 5 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_p5_p10, p_value_p5_p10 = wilcoxon(clean_data_p5_p10['recall_positive_p5'], clean_data_p5_p10['recall_positive_p10'])
        print(f"Resultados de la prueba de Wilcoxon para Patience 5 vs 10: w_statistic = {w_statistic_p5_p10}, p_value = {p_value_p5_p10}")
        if p_value_p5_p10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Patience 5 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Patience 5 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Patience 5 y 10\n----------------------------------------------------\n")


# Eliminamos valores nulos en runtime_minutes antes de la prueba de hipótesis
clean_data_p5_p10 = merged_data_p5_p10.dropna(subset=['runtime_minutes_p5', 'runtime_minutes_p10'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_p5_p10.empty:
    differences = clean_data_p5_p10['runtime_minutes_p5'] - clean_data_p5_p10['runtime_minutes_p10']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_p5_p10, p_value_p5_p10 = ttest_rel(clean_data_p5_p10['runtime_minutes_p5'], clean_data_p5_p10['runtime_minutes_p10'])
        print(f"Resultados de la prueba t pareada para Patience 5 vs 10: t_statistic = {t_statistic_p5_p10}, p_value = {p_value_p5_p10}")
        if p_value_p5_p10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Patience 5 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Patience 5 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_p5_p10, p_value_p5_p10 = wilcoxon(clean_data_p5_p10['runtime_minutes_p5'], clean_data_p5_p10['runtime_minutes_p10'])
        print(f"Resultados de la prueba de Wilcoxon para Patience 5 vs 10: w_statistic = {w_statistic_p5_p10}, p_value = {p_value_p5_p10}")
        if p_value_p5_p10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Patience 5 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Patience 5 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Patience 5 y 10\n----------------------------------------------------\n")

# Cell 77
# Configuración de los gráficos con etiquetas de los ejes
plt.figure(figsize=(14, 10))

# Gráfico de caja y bigote de f1_positive
plt.subplot(2, 2, 1)
sns.boxplot(x='model', y='f1_positive', hue='Patience', data=df, order=order, showfliers=True, palette=palette)
plt.title('Binary F1 Score')
plt.xlabel('Model')
plt.ylabel('Binary F1')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de recall_positive
plt.subplot(2, 2, 2)
sns.boxplot(x='model', y='recall_positive', hue='Patience', data=df, order=order, showfliers=True, palette=palette)
plt.title('Binary Recall Score')
plt.xlabel('Model')
plt.ylabel('Binary Recall')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de runtime_minutes con outliers
plt.subplot(2, 2, 3)
sns.boxplot(x='model', y='runtime_minutes', hue='Patience', data=df, order=order, showfliers=True, palette=palette)
plt.title('Runtime (with Outliers)')
plt.xlabel('Model')
plt.ylabel('Runtime (minutes)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de runtime_minutes sin outliers
plt.subplot(2, 2, 4)
sns.boxplot(x='model', y='runtime_minutes', hue='Patience', data=df, order=order, showfliers=False, palette=palette)
plt.title('Runtime (Without Outliers)')
plt.xlabel('Model')
plt.ylabel('Runtime (minutes)')
plt.xticks(rotation=45)

# Colocar un título general
plt.suptitle('Comparison of Metrics by Patience in Neural Network Models', fontsize=16, y=0.98)

# Ajustar el diseño y guardar como SVG
plt.tight_layout()
plt.savefig("metrics_by_patience.svg", format='svg')

# Mostrar los gráficos
plt.show()

# Cell 79
# Ajustar el orden personalizado
order = ['bert', 'auto_bert', 'roberta', 'auto_roberta', 'rnn_lstm', 'rnn_gru']

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df['Patience'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Patience
plt.figure(figsize=(14, 6))

# Gráfico de caja y bigote de test_f1
plt.subplot(1, 2, 1)
sns.boxplot(x=df['model'], y=df['test_f1'], hue=df['Patience'], data=df, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo NN y Patience\n(Macro F1 Score)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de test_recall
plt.subplot(1, 2, 2)
sns.boxplot(x=df['model'], y=df['test_recall'], hue=df['Patience'], data=df, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo NN y Patience\n(Macro Recall Score)')
plt.xticks(rotation=45)

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 81
# Filtrar el DataFrame para incluir solo 'bert' y 'roberta' y 'Patience' igual a 5
df2 = df[(df['model'].isin(['bert', 'roberta', 'auto_bert', 'auto_roberta'])) & (df['Patience'] == 5)]

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Patience
plt.figure(figsize=(14, 13))

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 2, 1)
sns.boxplot(x=df2['freezed_layers'], y=df2['f1_positive'], data=df2, showfliers=True)
plt.title('Distribución de Test f1 Score y freezed_layers\n(Binary F1 Score)')

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 2, 2)
sns.boxplot(x=df2['freezed_layers'], y=df2['recall_positive'], data=df2, showfliers=True)
plt.title('Distribución de Test Recall y freezed_layers\n(Binary Recall Score)')

# Gráfico de caja y bigote de runtime_minutes
plt.subplot(2, 2, 3)
sns.boxplot(x=df2['freezed_layers'], y=df2['runtime_minutes'], data=df2, showfliers=True)
plt.title('Distribución de runtime_minutes por Modelo NN y freezed_layers\n(minutos)')

# Gráfico de caja y bigote de runtime_minutes
plt.subplot(2, 2, 4)
sns.boxplot(x=df2['freezed_layers'], y=df2['runtime_minutes'], data=df2, showfliers=False)
plt.title('Distribución de runtime_minutes por Modelo NN y freezed_layers\n(minutos)')

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 83
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, shapiro

# Cargar el archivo CSV
file_path = '/home/froh/results_final_summary_modelos.csv'
data = pd.read_csv(file_path)


# Filtramos los datos eliminando los modelos 'rnn_gru' y 'rnn_lstm' y considerando solo Patience = 5
filtered_data = data[(data['Patience'] == 5) & (~data['model'].isin(['rnn_gru', 'rnn_lstm']))]

# Filtrar modelos con freezed_layers = 0, 8 y 10
freezed_0 = filtered_data[filtered_data['freezed_layers'] == 0]
freezed_8 = filtered_data[filtered_data['freezed_layers'] == 8]
freezed_10 = filtered_data[filtered_data['freezed_layers'] == 10]

# Comparar f1_positive entre freezed_layers = 0 y freezed_layers = 8
merged_data_0_8 = pd.merge(freezed_0, freezed_8, on=['model', 'dataset'], suffixes=('_f0', '_f8'))
clean_data_0_8 = merged_data_0_8.dropna(subset=['f1_positive_f0', 'f1_positive_f8'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_0_8.empty:
    differences = clean_data_0_8['f1_positive_f0'] - clean_data_0_8['f1_positive_f8']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_f0_f8, p_value_f0_f8 = ttest_rel(clean_data_0_8['f1_positive_f0'], clean_data_0_8['f1_positive_f8'])
        print(f"Resultados de la prueba t pareada para Freezed layers 0 vs. 8: t_statistic = {t_statistic_f0_f8}, p_value = {p_value_f0_f8}")
        if p_value_f0_f8 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 8 en términos de f1_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 8 en términos de f1_positive.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_f0_f8, p_value_f0_f8 = wilcoxon(clean_data_0_8['f1_positive_f0'], clean_data_0_8['f1_positive_f8'])
        print(f"Resultados de la prueba de Wilcoxon para Freezed layers 0 vs. 8: w_statistic = {w_statistic_f0_f8}, p_value = {p_value_f0_f8}")
        if p_value_f0_f8 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 8 en términos de f1_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 8 en términos de f1_positive.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Freezed layers 0 y 8\n----------------------------------------------------\n")



# Eliminamos valores nulos en recall_positive antes de la prueba de hipótesis
clean_data_0_8 = merged_data_0_8.dropna(subset=['recall_positive_f0', 'recall_positive_f8'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_0_8.empty:
    differences = clean_data_0_8['recall_positive_f0'] - clean_data_0_8['recall_positive_f8']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_f0_f8, p_value_f0_f8 = ttest_rel(clean_data_0_8['recall_positive_f0'], clean_data_0_8['recall_positive_f8'])
        print(f"Resultados de la prueba t pareada para Freezed layers 0 vs. 8: t_statistic = {t_statistic_f0_f8}, p_value = {p_value_f0_f8}")
        if p_value_f0_f8 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 8 en términos de recall_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 8 en términos de recall_positive.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_f0_f8, p_value_f0_f8 = wilcoxon(clean_data_0_8['recall_positive_f0'], clean_data_0_8['recall_positive_f8'])
        print(f"Resultados de la prueba de Wilcoxon para Freezed layers 0 vs. 8: w_statistic = {w_statistic_f0_f8}, p_value = {p_value_f0_f8}")
        if p_value_f0_f8 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 8 en términos de recall_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 8 en términos de recall_positive.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Freezed layers 0 y 8\n----------------------------------------------------\n")



# Comparar f1_positive entre freezed_layers = 0 y freezed_layers = 10
merged_data_0_10 = pd.merge(freezed_0, freezed_10, on=['model', 'dataset'], suffixes=('_f0', '_f10'))
clean_data_0_10 = merged_data_0_10.dropna(subset=['f1_positive_f0', 'f1_positive_f10'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_0_10.empty:
    differences = clean_data_0_10['f1_positive_f0'] - clean_data_0_10['f1_positive_f10']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_f0_f10, p_value_f0_f10 = ttest_rel(clean_data_0_10['f1_positive_f0'], clean_data_0_10['f1_positive_f10'])
        print(f"Resultados de la prueba t pareada para Freezed layers 0 vs. 10: t_statistic = {t_statistic_f0_f10}, p_value = {p_value_f0_f10}")
        if p_value_f0_f10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_f0_f10, p_value_f0_f10 = wilcoxon(clean_data_0_10['f1_positive_f0'], clean_data_0_10['f1_positive_f10'])
        print(f"Resultados de la prueba de Wilcoxon para Freezed layers 0 vs. 10: w_statistic = {w_statistic_f0_f10}, p_value = {p_value_f0_f10}")
        if p_value_f0_f10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 10 en términos de f1_positive.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Freezed layers 0 y 10\n----------------------------------------------------\n")



# Eliminamos valores nulos en recall_positive antes de la prueba de hipótesis
clean_data_0_10 = merged_data_0_10.dropna(subset=['recall_positive_f0', 'recall_positive_f10'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_0_10.empty:
    differences = clean_data_0_10['recall_positive_f0'] - clean_data_0_10['recall_positive_f10']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_f0_f10, p_value_f0_f10 = ttest_rel(clean_data_0_10['recall_positive_f0'], clean_data_0_10['recall_positive_f10'])
        print(f"Resultados de la prueba t pareada para Freezed layers 0 vs. 10: t_statistic = {t_statistic_f0_f10}, p_value = {p_value_f0_f10}")
        if p_value_f0_f10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_f0_f10, p_value_f0_f10 = wilcoxon(clean_data_0_10['recall_positive_f0'], clean_data_0_10['recall_positive_f10'])
        print(f"Resultados de la prueba de Wilcoxon para Freezed layers 0 vs. 10: w_statistic = {w_statistic_f0_f10}, p_value = {p_value_f0_f10}")
        if p_value_f0_f10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 10 en términos de recall_positive.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Freezed layers 0 y 10\n----------------------------------------------------\n")



# Comparar f1_positive entre freezed_layers = 0 y freezed_layers = 8
merged_data_0_8 = pd.merge(freezed_0, freezed_8, on=['model', 'dataset'], suffixes=('_f0', '_f8'))
clean_data_0_8 = merged_data_0_8.dropna(subset=['runtime_minutes_f0', 'runtime_minutes_f8'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_0_8.empty:
    differences = clean_data_0_8['runtime_minutes_f0'] - clean_data_0_8['runtime_minutes_f8']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_f0_f8, p_value_f0_f8 = ttest_rel(clean_data_0_8['runtime_minutes_f0'], clean_data_0_8['runtime_minutes_f8'])
        print(f"Resultados de la prueba t pareada para Freezed layers 0 vs. 8: t_statistic = {t_statistic_f0_f8}, p_value = {p_value_f0_f8}")
        if p_value_f0_f8 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 8 en términos de runtime_minutes.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 8 en términos de runtime_minutes.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_f0_f8, p_value_f0_f8 = wilcoxon(clean_data_0_8['runtime_minutes_f0'], clean_data_0_8['runtime_minutes_f8'])
        print(f"Resultados de la prueba de Wilcoxon para Freezed layers 0 vs. 8: w_statistic = {w_statistic_f0_f8}, p_value = {p_value_f0_f8}")
        if p_value_f0_f8 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 8 en términos de runtime_minutes.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 8 en términos de runtime_minutes.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Freezed layers 0 y 8\n----------------------------------------------------\n")


# Comparar runtime_minutes entre freezed_layers = 0 y freezed_layers = 10
merged_data_0_10 = pd.merge(freezed_0, freezed_10, on=['model', 'dataset'], suffixes=('_f0', '_f10'))
clean_data_0_10 = merged_data_0_10.dropna(subset=['runtime_minutes_f0', 'runtime_minutes_f10'])

# Realizamos la prueba de normalidad de Shapiro-Wilk para las diferencias
if not clean_data_0_10.empty:
    differences = clean_data_0_10['runtime_minutes_f0'] - clean_data_0_10['runtime_minutes_f10']
    shapiro_stat, shapiro_p_value = shapiro(differences)
    print(f"Resultados de la prueba de Shapiro-Wilk para normalidad: statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Conclusión: Las diferencias siguen una distribución normal.\n")
    else:
        print("Conclusión: Las diferencias no siguen una distribución normal.\n")

    # Si la distribución es normal (p_value > 0.05), utilizamos el t-test pareado
    if shapiro_p_value > 0.05:
        t_statistic_f0_f10, p_value_f0_f10 = ttest_rel(clean_data_0_10['runtime_minutes_f0'], clean_data_0_10['runtime_minutes_f10'])
        print(f"Resultados de la prueba t pareada para Freezed layers 0 vs. 10: t_statistic = {t_statistic_f0_f10}, p_value = {p_value_f0_f10}")
        if p_value_f0_f10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
    else:
        # Si la distribución no es normal (p_value <= 0.05), utilizamos la prueba de Wilcoxon
        w_statistic_f0_f10, p_value_f0_f10 = wilcoxon(clean_data_0_10['runtime_minutes_f0'], clean_data_0_10['runtime_minutes_f10'])
        print(f"Resultados de la prueba de Wilcoxon para Freezed layers 0 vs. 10: w_statistic = {w_statistic_f0_f10}, p_value = {p_value_f0_f10}")
        if p_value_f0_f10 < 0.05:
            print("Conclusión: Existe una diferencia significativa entre Freezed layers 0 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
        else:
            print("Conclusión: No existe una diferencia significativa entre Freezed layers 0 y 10 en términos de runtime_minutes.\n----------------------------------------------------\n")
else:
    print("No hay suficientes datos para realizar la prueba de hipótesis entre Freezed layers 0 y 10\n----------------------------------------------------\n")

# Cell 85
import pandas as pd
from scipy.stats import shapiro, friedmanchisquare
from statsmodels.stats.anova import AnovaRM
import numpy as np

# Cargar el archivo CSV
file_path = '/home/froh/results_final_summary_modelos.csv'
data = pd.read_csv(file_path)

# Filtrar los datos eliminando los modelos 'rnn_gru' y 'rnn_lstm' y considerando solo Patience = 5
filtered_data = data[(data['Patience'] == 5) & (~data['model'].isin(['rnn_gru', 'rnn_lstm']))]

# Filtrar modelos con freezed_layers = 0, 8 y 10
freezed_layers = [0, 8, 10]
filtered_data = filtered_data[filtered_data['freezed_layers'].isin(freezed_layers)]

# Pivotar los datos para tener una columna por cada valor de freezed_layers
pivot_cols = ['f1_positive', 'recall_positive', 'runtime_minutes']
id_vars = ['model', 'dataset']

data_pivot = filtered_data.pivot_table(index=id_vars, columns='freezed_layers', values=pivot_cols)

# Verificar los nombres de las columnas
print("Columnas después de pivotar los datos:")
print(data_pivot.columns)

# Aplanar el MultiIndex de las columnas
data_pivot.columns = ['{}_f{}'.format(var, fl) for var, fl in data_pivot.columns]

# Verificar los nombres de las columnas después de aplanar
print("Columnas después de aplanar el MultiIndex:")
print(data_pivot.columns)

# Resetear el índice para tener 'model' y 'dataset' como columnas
data_pivot = data_pivot.reset_index()

# Lista de variables a analizar
variables = ['f1_positive', 'recall_positive', 'runtime_minutes']

for var in variables:
    print(f"\nAnálisis de la variable: {var}")

    # Verificar si las columnas necesarias están presentes
    required_columns = [f'{var}_f0.0', f'{var}_f8.0', f'{var}_f10.0']
    missing_columns = [col for col in required_columns if col not in data_pivot.columns]
    if missing_columns:
        print(f"Faltan las siguientes columnas para el análisis: {missing_columns}")
        print("Saltar esta variable.\n")
        continue

    # Preparar los datos para la prueba
    data_var = data_pivot[['model', 'dataset', f'{var}_f0.0', f'{var}_f8.0', f'{var}_f10.0']]
    data_var.columns = ['model', 'dataset', 'f0', 'f8', 'f10']

    # Eliminar filas con valores nulos
    data_var = data_var.dropna()

    if data_var.empty:
        print("No hay suficientes datos para realizar el análisis para esta variable.\n")
        continue

    # Reestructurar los datos para ANOVA de medidas repetidas
    data_melt = pd.melt(data_var, id_vars=['model', 'dataset'], value_vars=['f0', 'f8', 'f10'],
                        var_name='freezed_layers', value_name=var)
    data_melt['freezed_layers'] = data_melt['freezed_layers'].map({'f0': 0, 'f8': 8, 'f10': 10})

    # Crear una columna de sujeto único combinando 'model' y 'dataset'
    data_melt['subject'] = data_melt['model'] + '_' + data_melt['dataset']

    # Verificar normalidad de las diferencias entre cada par de configuraciones
    differences = {}
    pairs = [('f0', 'f8'), ('f0', 'f10'), ('f8', 'f10')]
    normal = True
    for (a, b) in pairs:
        diff = data_var[a] - data_var[b]
        differences[f'{a}-{b}'] = diff
        shapiro_stat, shapiro_p_value = shapiro(diff)
        print(f"Prueba de Shapiro-Wilk para la normalidad de las diferencias ({a} vs {b}): statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
        if shapiro_p_value > 0.05:
            print(f"Conclusión: Las diferencias entre {a} y {b} siguen una distribución normal.\n")
        else:
            print(f"Conclusión: Las diferencias entre {a} y {b} no siguen una distribución normal.\n")
            normal = False  # Si alguna comparación no es normal, usamos prueba no paramétrica

    # Seleccionar la prueba estadística según la normalidad
    if normal:
        print("Todas las diferencias siguen una distribución normal. Usaremos ANOVA de medidas repetidas.\n")
        # Convertir 'freezed_layers' a categórica
        data_melt['freezed_layers'] = data_melt['freezed_layers'].astype('category')
        # Realizar el ANOVA de medidas repetidas
        aovrm = AnovaRM(data_melt, depvar=var, subject='subject', within=['freezed_layers'])
        res = aovrm.fit()
        print(res)
        # Si el resultado es significativo, realizar pruebas post-hoc
        p_value = res.anova_table['Pr > F'][0]
        if p_value < 0.05:
            print("Conclusión: Existe una diferencia significativa entre las configuraciones.\n")
            # Realizar pruebas post-hoc (ejemplo usando t-tests pareados con corrección de Bonferroni)
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey = pairwise_tukeyhsd(endog=data_melt[var], groups=data_melt['freezed_layers'], alpha=0.05)
            print(tukey)
        else:
            print("Conclusión: No hay diferencias significativas entre las configuraciones.\n")
    else:
        print("No se cumple la normalidad. Usaremos la prueba de Friedman.\n")
        # Preparar los datos para la prueba de Friedman
        data_friedman = data_var[['f0', 'f8', 'f10']]
        # Realizar la prueba de Friedman
        friedman_stat, friedman_p_value = friedmanchisquare(data_friedman['f0'], data_friedman['f8'], data_friedman['f10'])
        print(f"Resultado de la prueba de Friedman: statistic = {friedman_stat}, p_value = {friedman_p_value}")
        if friedman_p_value < 0.05:
            print("Conclusión: Existe una diferencia significativa entre las configuraciones.\n")
            # Realizar pruebas post-hoc (por ejemplo, la prueba de Nemenyi)
            import scikit_posthocs as sp
            nemenyi_results = sp.posthoc_nemenyi_friedman(data_friedman)
            nemenyi_results.index = ['f0', 'f8', 'f10']
            nemenyi_results.columns = ['f0', 'f8', 'f10']
            print("Resultados de la prueba post-hoc de Nemenyi:")
            print(nemenyi_results)
        else:
            print("Conclusión: No hay diferencias significativas entre las configuraciones.\n")

# Cell 87
# Filtrar el DataFrame para incluir solo 'bert' y 'roberta' y 'Patience' igual a 5
df2 = df[(df['model'].isin(['bert', 'roberta', 'auto_bert', 'auto_roberta'])) & (df['Patience'] == 5)]

# Ajustar el orden personalizado
order = ['bert', 'auto_bert', 'roberta', 'auto_roberta']

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df['freezed_layers'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Patience
plt.figure(figsize=(14, 13))

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 2, 1)
sns.boxplot(x=df2['model'], y=df2['test_f1'], hue=df2['freezed_layers'], data=df2, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo NN y freezed_layers\n(Macro F1 Score)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 2, 2)
sns.boxplot(x=df2['model'], y=df2['test_recall'], hue=df2['freezed_layers'], data=df2, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo NN y freezed_layers\n(Macro Recall Score)')
plt.xticks(rotation=45)

# Colocar la leyenda del segundo gráfico en la parte superior derecha
plt.legend(loc='upper right')

# Gráfico de caja y bigote de runtime_minutes
plt.subplot(2, 2, 3)
sns.boxplot(x=df2['model'], y=df2['runtime_minutes'], hue=df2['freezed_layers'], data=df2, order=order, showfliers=True, palette=palette)
plt.title('Distribución de runtime_minutes por Modelo NN y freezed_layers\n(minutos)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de runtime_minutes
plt.subplot(2, 2, 4)
sns.boxplot(x=df2['model'], y=df2['runtime_minutes'], hue=df2['freezed_layers'], data=df2, order=order, showfliers=False, palette=palette)
plt.title('Distribución de runtime_minutes por Modelo NN y freezed_layers\n(minutos)')
plt.xticks(rotation=45)

# Colocar la leyenda del segundo gráfico en la parte superior derecha
plt.legend(loc='upper right')

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 90
# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']
# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_em.loc[:, 'embedding'] = pd.Categorical(df_em['embedding'], categories=embedding_categories, ordered=True)

# Ajustar el orden personalizado
df_em2 = df_em[df_em['model'] != 'SVC']
# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_em2.loc[:, 'embedding'] = pd.Categorical(df_em2['embedding'], categories=embedding_categories, ordered=True)
df_em3 = df_em2[(df_em2['embedding'] != 'BoW') & (df_em2['embedding'] != 'TF-IDF')]
# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_em3.loc[:, 'embedding'] = pd.Categorical(df_em3['embedding'], categories=embedding_categories, ordered=True)

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df_em['embedding'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn
palette2 = sns.color_palette("Set2", len(df_em2['embedding'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn
palette3 = sns.color_palette("Set2", len(df_em3['embedding'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes embedding
plt.figure(figsize=(14, 18))

# Gráfico de caja y bigote con outliers
plt.subplot(3, 2, 1)
sns.boxplot(x=df_em['model'], y=df_em['runtime_minutes'], hue=df_em['embedding'], data=df_em, showfliers=True, palette=palette)
plt.title('Distribución de tiempo de ejecución por Modelo ML y embedding (Con Outliers)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote sin outliers
plt.subplot(3, 2, 2)
sns.boxplot(x=df_em['model'], y=df_em['runtime_minutes'], hue=df_em['embedding'], data=df_em, showfliers=False, palette=palette)
plt.title('Distribución de tiempo de ejecución por Modelo ML y embedding (Sin Outliers)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote con outliers
plt.subplot(3, 2, 3)
sns.boxplot(x=df_em2['model'], y=df_em2['runtime_minutes'], hue=df_em2['embedding'], data=df_em2, showfliers=True, palette=palette2)
plt.title('Distribución de tiempo de ejecución por Modelo ML y embedding (Con Outliers)\n(sin modelos SVC)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote sin outliers
plt.subplot(3, 2, 4)
sns.boxplot(x=df_em2['model'], y=df_em2['runtime_minutes'], hue=df_em2['embedding'], data=df_em2, showfliers=False, palette=palette2)
plt.title('Distribución de tiempo de ejecución por Modelo ML y embedding (Sin Outliers)\n(sin modelos SVC)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote con outliers
plt.subplot(3, 2, 5)
sns.boxplot(x=df_em3['model'], y=df_em3['runtime_minutes'], hue=df_em3['embedding'], data=df_em3, showfliers=True, palette=palette3)
plt.title('Distribución de tiempo de ejecución por Modelo ML y embedding (Con Outliers)\n(sin modelos SVC y embeddings BoW y TF-IDF)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote sin outliers
plt.subplot(3, 2, 6)
sns.boxplot(x=df_em3['model'], y=df_em3['runtime_minutes'], hue=df_em3['embedding'], data=df_em3, showfliers=False, palette=palette3)
plt.title('Distribución de tiempo de ejecución por Modelo ML y embedding (Sin Outliers)\n(sin modelos SVC y embeddings BoW y TF-IDF')
plt.xticks(rotation=45)

# Colocar un título general (supertítulo)
plt.suptitle('Comparación de Tiempo de Ejecución por Modelo ML y embedding', fontsize=16, y=0.99)

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 92
# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df_em['embedding'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes embedding
plt.figure(figsize=(16, 10))

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 1, 1)
sns.boxplot(x=df_em['model'], y=df_em['test_f1'], hue=df_em['embedding'], data=df_em, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo ML y embedding\n(Binary F1 Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 1, 2)
sns.boxplot(x=df_em['model'], y=df_em['test_recall'], hue=df_em['embedding'], data=df_em, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo ML y embedding\n(Binary Recall Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Ampliar el ancho de los gráficos para que las cajas no queden tan compactas
plt.subplots_adjust(wspace=0.4)  # Aumentamos el espacio entre los subplots

# Mostrar los gráficos
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustamos el layout para dar espacio a la leyenda en la derecha
plt.show()


# Cell 94
# Ajustar el orden personalizado
order = ['bert', 'auto_bert', 'roberta', 'auto_roberta', 'rnn_lstm', 'rnn_gru']

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df['balance'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes balance
plt.figure(figsize=(16, 10))

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 1, 1)
sns.boxplot(x=df['model'], y=df['test_f1'], hue=df['balance'], data=df, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo NN y balance\n(Macro F1 Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 1, 2)
sns.boxplot(x=df['model'], y=df['test_recall'], hue=df['balance'], data=df, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo NN y balance\n(Macro Recall Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Ampliar el ancho de los gráficos para que las cajas no queden tan compactas
plt.subplots_adjust(wspace=0.4)  # Aumentamos el espacio entre los subplots

# Mostrar los gráficos
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustamos el layout para dar espacio a la leyenda en la derecha
plt.show()


# Cell 96
# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df_em['balance'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes balance
plt.figure(figsize=(16, 10))

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 1, 1)
sns.boxplot(x=df_em['model'], y=df_em['test_f1'], hue=df_em['balance'], data=df_em, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo ML y balance\n(Binary F1 Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 1, 2)
sns.boxplot(x=df_em['model'], y=df_em['test_recall'], hue=df_em['balance'], data=df_em, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo ML y balance\n(Binary Recall Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Ampliar el ancho de los gráficos para que las cajas no queden tan compactas
plt.subplots_adjust(wspace=0.4)  # Aumentamos el espacio entre los subplots

# Mostrar los gráficos
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustamos el layout para dar espacio a la leyenda en la derecha
plt.show()


# Cell 98
# Ajustar el orden personalizado
order = [
          '00. 3785 - 117', '01. 117 - 117', '02. 1000 - 117', '03. 3785 - 2500',
          '04. 500 - 500', '05. 1000 - 1000', '06. 2500 - 2500', '07. 3785 - 3879', #'08. 3785 - 5145',
          '09. 3785 - 21800', '10. 3785 - 26460']

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df['tamaño_dataset'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Pacience
plt.figure(figsize=(14, 6))

# Gráfico de caja y bigote de test_f1
plt.subplot(1, 2, 1)
sns.boxplot(x=df['tamaño_dataset'], y=df['test_f1'], data=df, order=order, showfliers=True)
plt.title('Distribución de Test f1 Score por tamaño_dataset en modelos NN\n(Macro F1 Score)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de test_recall
plt.subplot(1, 2, 2)
sns.boxplot(x=df['tamaño_dataset'], y=df['test_recall'], data=df, order=order, showfliers=True)
plt.title('Distribución de Test Recall por tamaño_dataset en modelos NN\n(Macro Recall Score)')
plt.xticks(rotation=45)

# Colocar la leyenda del segundo gráfico en la parte superior derecha
#plt.legend(loc='upper right')

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 100
# Ajustar el orden personalizado
order = [
          '00. 3785 - 117', '01. 117 - 117', '02. 1000 - 117', '03. 3785 - 2500',
          '04. 500 - 500', '05. 1000 - 1000', '06. 2500 - 2500', '07. 3785 - 3879', #'08. 3785 - 5145',
          '09. 3785 - 21800', '10. 3785 - 26460']

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df_em['tamaño_dataset'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Pacience
plt.figure(figsize=(14, 6))

# Gráfico de caja y bigote de test_f1
plt.subplot(1, 2, 1)
sns.boxplot(x=df_em['tamaño_dataset'], y=df_em['test_f1'], data=df_em, order=order, showfliers=True)
plt.title('Distribución de Test f1 Score por tamaño_dataset en modelos ML\n(Binary F1 Score)')
plt.xticks(rotation=45)

# Gráfico de caja y bigote de test_recall
plt.subplot(1, 2, 2)
sns.boxplot(x=df_em['tamaño_dataset'], y=df_em['test_recall'], data=df_em, order=order, showfliers=True)
plt.title('Distribución de Test Recall por  tamaño_dataset en modelos ML\n(Binary Recall Score)')
plt.xticks(rotation=45)

# Colocar la leyenda del segundo gráfico en la parte superior derecha
#plt.legend(loc='upper right')

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 102
# Ajustar el orden personalizado
order = [
        '00 original data', '01 undersampling', '02 sinónimos', '03 substitute BERT', '04 insert BERT',
        '05 sentence GPT2', '06 sentence prompt GPT2', '07 sentence prompt GPT4o mini', #'08 augmented prompt GPT4o mini',
        '09 All augmented data', '10 All augmented + aug gtp4 data']

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df['data_augmentation'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Pacience
plt.figure(figsize=(14, 6))

# Gráfico de caja y bigote de test_f1
plt.subplot(1, 2, 1)
sns.boxplot(x=df['data_augmentation'], y=df['test_f1'], data=df, order=order, showfliers=True)
plt.title('Distribución de Test f1 Score por data_augmentation en modelos NN\n(Macro F1 Score)')
plt.xticks(rotation=90)

# Gráfico de caja y bigote de test_recall
plt.subplot(1, 2, 2)
sns.boxplot(x=df['data_augmentation'], y=df['test_recall'], data=df, order=order, showfliers=True)
plt.title('Distribución de Test Recall por  data_augmentation en modelos NN\n(Macro Recall Score)')
plt.xticks(rotation=90)

# Colocar la leyenda del segundo gráfico en la parte superior derecha
#plt.legend(loc='upper right')

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 104
# Ajustar el orden personalizado
order = [
        '00 original data', '01 undersampling', '02 sinónimos', '03 substitute BERT', '04 insert BERT',
        '05 sentence GPT2', '06 sentence prompt GPT2', '07 sentence prompt GPT4o mini', #'08 augmented prompt GPT4o mini',
        '09 All augmented data', '10 All augmented + aug gtp4 data']

# Definir una paleta de colores personalizada (puedes elegir entre muchas paletas disponibles en Seaborn)
palette = sns.color_palette("Set2", len(df_em['data_augmentation'].unique()))  # "Set2" es una de las paletas disponibles en Seaborn

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Pacience
plt.figure(figsize=(14, 6))

# Gráfico de caja y bigote de test_f1
plt.subplot(1, 2, 1)
sns.boxplot(x=df_em['data_augmentation'], y=df_em['test_f1'], data=df_em, order=order, showfliers=True)
plt.title('Distribución de Test f1 Score por data_augmentation en modelos ML\n(Binary F1 Score)')
plt.xticks(rotation=90)

# Gráfico de caja y bigote de test_recall
plt.subplot(1, 2, 2)
sns.boxplot(x=df_em['data_augmentation'], y=df_em['test_recall'], data=df_em, order=order, showfliers=True)
plt.title('Distribución de Test Recall por  data_augmentation en modelos ML\n(Binary Recall Score)')
plt.xticks(rotation=90)

# Colocar la leyenda del segundo gráfico en la parte superior derecha
#plt.legend(loc='upper right')

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 106
# Ajustar el orden personalizado
order = ['bert', 'auto_bert', 'roberta', 'auto_roberta', 'rnn_lstm', 'rnn_gru']

# Definir una paleta de colores personalizada
palette = sns.color_palette("Set2", len(df['data_augmentation'].unique()))

# Generar gráficos de caja y bigote para comparar test_f1 y test_recall por model
plt.figure(figsize=(16, 10))  # Aumentamos el tamaño general del gráfico

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 1, 1)
sns.boxplot(x=df['model'], y=df['test_f1'], hue=df['data_augmentation'], data=df, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo NN y data_augmentation\n(Macro F1 Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 1, 2)
sns.boxplot(x=df['model'], y=df['test_recall'], hue=df['data_augmentation'], data=df, order=order, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo NN y data_augmentation\n(Macro Recall Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Ampliar el ancho de los gráficos para que las cajas no queden tan compactas
plt.subplots_adjust(wspace=0.4)  # Aumentamos el espacio entre los subplots

# Mostrar los gráficos
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustamos el layout para dar espacio a la leyenda en la derecha
plt.show()

# Cell 108
# Definir una paleta de colores personalizada
palette = sns.color_palette("Set2", len(df_em['data_augmentation'].unique()))

# Generar gráficos de caja y bigote para comparar test_f1 y test_recall por model
plt.figure(figsize=(16, 10))  # Aumentamos el tamaño general del gráfico

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 1, 1)
sns.boxplot(x=df_em['model'], y=df_em['test_f1'], hue=df_em['data_augmentation'], data=df_em, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo ML y data_augmentation\n(Binary F1 Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 1, 2)
sns.boxplot(x=df_em['model'], y=df_em['test_recall'], hue=df_em['data_augmentation'], data=df_em, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo ML y data_augmentation\n(Binary Recall Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)  # ncol=1 para tener solo una columna

# Ampliar el ancho de los gráficos para que las cajas no queden tan compactas
plt.subplots_adjust(wspace=0.4)  # Aumentamos el espacio entre los subplots

# Mostrar los gráficos
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustamos el layout para dar espacio a la leyenda en la derecha
plt.show()

# Cell 110
# Ajustar el orden personalizado de los modelos
order = ['bert', 'auto_bert', 'roberta', 'auto_roberta', 'rnn_lstm', 'rnn_gru']

# Ordenar el hue alfabéticamente
sorted_hue = sorted(df['tamaño_dataset'].unique())  # Ordenar los valores únicos de tamaño_dataset

# Definir una paleta de colores personalizada para que coincida con el orden alfabético de tamaño_dataset
palette = sns.color_palette("Set2", len(sorted_hue))  # Crear paleta con el mismo número de colores que las clases de tamaño_dataset

# Generar gráficos de caja y bigote para comparar test_f1 y test_recall por modelo con tamaño_dataset como hue
plt.figure(figsize=(16, 10))  # Aumentamos el tamaño general del gráfico

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 1, 1)
sns.boxplot(x=df['model'], y=df['test_f1'], hue=df['tamaño_dataset'], data=df, order=order, hue_order=sorted_hue, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo NN y tamaño_dataset\n(Macro F1 Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 1, 2)
sns.boxplot(x=df['model'], y=df['test_recall'], hue=df['tamaño_dataset'], data=df, order=order, hue_order=sorted_hue, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo NN y tamaño_dataset\n(Macro Recall Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)

# Ampliar el ancho de los gráficos para que las cajas no queden tan compactas
plt.subplots_adjust(wspace=0.4)

# Mostrar los gráficos
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# Cell 112
# Ordenar el hue alfabéticamente
sorted_hue = sorted(df_em['tamaño_dataset'].unique())  # Ordenar los valores únicos de tamaño_dataset

# Definir una paleta de colores personalizada para que coincida con el orden alfabético de tamaño_dataset
palette = sns.color_palette("Set2", len(sorted_hue))  # Crear paleta con el mismo número de colores que las clases de tamaño_dataset

# Generar gráficos de caja y bigote para comparar test_f1 y test_recall por modelo con tamaño_dataset como hue
plt.figure(figsize=(16, 10))  # Aumentamos el tamaño general del gráfico

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 1, 1)
sns.boxplot(x=df_em['model'], y=df_em['test_f1'], hue=df_em['tamaño_dataset'], data=df_em, hue_order=sorted_hue, showfliers=True, palette=palette)
plt.title('Distribución de Test f1 Score por Modelo ML y tamaño_dataset\n(Binary F1 Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 1, 2)
sns.boxplot(x=df_em['model'], y=df_em['test_recall'], hue=df_em['tamaño_dataset'], data=df_em, hue_order=sorted_hue, showfliers=True, palette=palette)
plt.title('Distribución de Test Recall por Modelo ML y tamaño_dataset\n(Binary Recall Score)')
plt.xticks(rotation=45)

# Ajustar la leyenda para que aparezca afuera del gráfico a la derecha en una sola columna
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)

# Ampliar el ancho de los gráficos para que las cajas no queden tan compactas
plt.subplots_adjust(wspace=0.4)

# Mostrar los gráficos
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# Cell 115
import matplotlib.pyplot as plt
import seaborn as sns


# Ordenar los datos por 'Tamaño dataset' de manera alfabética
data_sorted = df.sort_values(by='tamaño_dataset')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_recall',
    y='test_f1',
    hue='tamaño_dataset',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='Tamaño dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN con Tamaño de Dataset\n(Macro metrics)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 117
import matplotlib.pyplot as plt
import seaborn as sns

# Ordenar los datos por 'Tamaño dataset' de manera alfabética
data_sorted = df_em.sort_values(by='tamaño_dataset')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_recall',
    y='test_f1',
    hue='tamaño_dataset',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='Tamaño dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo ML con Tamaño de Dataset\n(Binary metrics)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 119
import matplotlib.pyplot as plt
import seaborn as sns

df_total = pd.concat([df, df_em])

# Ordenar los datos por 'Tamaño dataset' de manera alfabética
data_sorted = df_total.sort_values(by='tamaño_dataset')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='recall_positive',
    y='f1_positive',
    hue='tamaño_dataset',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='Tamaño dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN y ML con Tamaño de Dataset\n(Métricas binary)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 121
import matplotlib.pyplot as plt
import seaborn as sns

df_total = pd.concat([df, df_em])

# Ordenar los datos por 'Tamaño dataset' de manera alfabética
data_sorted = df_total.sort_values(by='tamaño_dataset')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_macro_recall',
    y='test_macro_f1',
    hue='tamaño_dataset',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='Tamaño dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN y ML con Tamaño de Dataset\n(Métricas macro)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 123
import matplotlib.pyplot as plt
import seaborn as sns



# Ordenar los datos por 'data_augmentation' de manera alfabética
data_sorted = df.sort_values(by='data_augmentation')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_recall',
    y='test_f1',
    hue='data_augmentation',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='data_augmentation', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN con Tipo data_augmentation\n(Métricas Macro)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 125
import matplotlib.pyplot as plt
import seaborn as sns



# Ordenar los datos por 'data_augmentation' de manera alfabética
data_sorted = df_em.sort_values(by='data_augmentation')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_recall',
    y='test_f1',
    hue='data_augmentation',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='data_augmentation', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo ML con Tipo data augmentation\n(Métricas binary)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 127
import matplotlib.pyplot as plt
import seaborn as sns

df_total = pd.concat([df, df_em])

# Ordenar los datos por 'data_augmentation' de manera alfabética
data_sorted = df_total.sort_values(by='data_augmentation')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='recall_positive',
    y='f1_positive',
    hue='data_augmentation',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='data_augmentation', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN y ML con Tipo data augmentation\n(Métricas binary)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 129
import matplotlib.pyplot as plt
import seaborn as sns

df_total = pd.concat([df, df_em])

# Ordenar los datos por 'data_augmentation' de manera alfabética
data_sorted = df_total.sort_values(by='data_augmentation')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_macro_recall',
    y='test_macro_f1',
    hue='data_augmentation',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='data_augmentation', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN y ML con Tipo data augmentation\n(Métricas macro)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 131
import matplotlib.pyplot as plt
import seaborn as sns

# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['0 sin embedding', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_em.loc[:, 'embedding'] = pd.Categorical(df_em['embedding'], categories=embedding_categories, ordered=True)

# Ordenar los datos por 'embedding' de manera alfabética
data_sorted = df_em.sort_values(by='embedding')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_recall',
    y='test_f1',
    hue='embedding',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='embedding', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo ML con Tipo de embedding\n(Métricas binary)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 133
import matplotlib.pyplot as plt
import seaborn as sns

df_tot = pd.concat([df, df_em])
df_tot['embedding'] = df_tot['embedding'].fillna('0 sin embedding')

# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['0 sin embedding', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_tot['embedding'] = pd.Categorical(df_tot['embedding'], categories=embedding_categories, ordered=True)

# Ordenar los datos por 'embedding' de manera alfabética
data_sorted = df_tot.sort_values(by='embedding')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='recall_positive',
    y='f1_positive',
    hue='embedding',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='embedding', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN y ML con Tipo data augmentation\n(Métricas binary)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 135
import matplotlib.pyplot as plt
import seaborn as sns

df_tot = pd.concat([df, df_em])
df_tot['embedding'] = df_tot['embedding'].fillna('0 sin embedding')

# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['0 sin embedding', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_tot['embedding'] = pd.Categorical(df_tot['embedding'], categories=embedding_categories, ordered=True)

# Ordenar los datos por 'embedding' de manera alfabética
data_sorted = df_tot.sort_values(by='embedding')

# Crear un gráfico de dispersión con el tamaño del dataset como color y el modelo como símbolo
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data_sorted,
    x='test_macro_recall',
    y='test_macro_f1',
    hue='embedding',
    style='model',
    palette='coolwarm',
    s=100
)

# Ajustar la leyenda para que aparezca en orden alfabético y añadir títulos
plt.legend(title='embedding', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Relación Entre Test_f1 y Test_recall Por Modelo NN y ML con Tipo data augmentation\n(Métricas macro)')
plt.xlabel('Test Recall')
plt.ylabel('Test F1')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 137
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

df_total = pd.concat([df, df_em])

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_total['tamaño_dataset_numeric'] = df_total['tamaño_dataset'].map(dataset_size_mapping)

# Crear la figura en 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Mapear los colores a 'data_augmentation'
colors = sns.color_palette('coolwarm', len(df_total['data_augmentation'].unique()))
color_mapping = {aug: colors[i] for i, aug in enumerate(df_total['data_augmentation'].unique())}
df_total['color'] = df_total['data_augmentation'].map(color_mapping)

# Crear el gráfico de dispersión 3D
scatter = ax.scatter(
    df_total['recall_positive'],
    df_total['f1_positive'],
    df_total['tamaño_dataset_numeric'],
    c=df_total['color'],
    s=50,  # Tamaño de los puntos
    alpha=0.7
)

# Ajustar las etiquetas de los ejes y el título
ax.set_xlabel('Test Recall')
ax.set_ylabel('Test F1')
ax.set_zlabel('Tamaño Dataset (numérico)')
plt.title('Relación Entre Test_f1, Test_recall y Tamaño Dataset en 3D\n(Métricas binary)')

# Colocar la leyenda
scatter_proxy = [plt.Line2D([0], [0], linestyle="none", marker="o", markersize=10, color=color) for color in colors]
ax.legend(scatter_proxy, df_total['data_augmentation'].unique(), title='Data Augmentation', bbox_to_anchor=(1.05, 1), loc='upper left')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 138
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames
df_total = pd.concat([df, df_em])

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Mapeo de etiquetas actuales a textos nuevos para la columna 'data_augmentation'
augmentation_label_mapping = {
    '01 undersampling': 'Undersampled',
    '02 sinónimos': 'Synonims',
    '00 original data': 'Original data',
    '04 insert BERT': 'Word insertions - BERT',
    '03 substitute BERT': 'Word substitutions - BERT',
    '05 sentence GPT2': 'GPT2 sentences',
    '06 sentence prompt GPT2': 'GPT2 prompts',
    '07 sentence prompt GPT4o mini': 'GPT4o mini prompts',
    '09 All augmented data': 'All augmented methods',
    '08 augmented prompt GPT4o mini': 'GPT4o mini prompts v2',
    '10 All augmented + aug gtp4 data': 'Consolidated data augmentation',
    'Unknown': 'Unknown'  # Para cubrir cualquier valor no mapeado
}

# Aplicar el mapeo a la columna 'data_augmentation'
df_total['data_augmentation'] = df_total['data_augmentation'].map(augmentation_label_mapping)

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_total['tamaño_dataset_numeric'] = df_total['tamaño_dataset'].map(dataset_size_mapping)

# Especificar el orden de las categorías
categoria_orden = [
    'Original data', 'Undersampled', 'Synonims', 'Word insertions - BERT', 'Word substitutions - BERT',
    'GPT2 sentences', 'GPT2 prompts', 'GPT4o mini prompts',
    'All augmented methods', 'GPT4o mini prompts v2', 'Consolidated data augmentation'
]

# Crear la figura en 3D con mayor espacio para la leyenda
fig = plt.figure(figsize=(12, 8))  # Aumenté el tamaño de la figura
ax = fig.add_subplot(111, projection='3d')

# Mapear los colores a 'data_augmentation'
colors = sns.color_palette('coolwarm', len(df_total['data_augmentation'].unique()))
color_mapping = {aug: colors[i] for i, aug in enumerate(categoria_orden)}
df_total['color'] = df_total['data_augmentation'].map(color_mapping)

# Mapear las formas a 'model'
markers = ['o', '^', 's', 'P', '*', 'X', 'D']
marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(df_total['model'].unique())}
df_total['marker'] = df_total['model'].map(marker_mapping)

# Crear el gráfico de dispersión 3D
for model in df_total['model'].unique():
    subset = df_total[df_total['model'] == model]
    ax.scatter(
        subset['recall_positive'],
        subset['f1_positive'],
        subset['tamaño_dataset_numeric'],
        c=subset['color'],
        marker=marker_mapping[model],
        s=50,  # Tamaño de los puntos
        alpha=0.7,
        label=model
    )

# Ajustar las etiquetas de los ejes y el título
ax.set_xlabel('Test Recall')
ax.set_ylabel('Test F1')
ax.set_zlabel('Dataset size')
plt.title('Test F1 Score, Test Recall and Dataset size\n(Binary Metrics)')

# Leyenda de Model (formas)
scatter_proxy_shapes = [plt.Line2D([0], [0], linestyle="none", marker=marker_mapping[model], markersize=10, color='black') for model in df_total['model'].unique()]
legend_patches = [plt.Line2D([0], [0], linestyle="none", marker="o", markersize=10, color=color_mapping[aug]) for aug in categoria_orden]

# Combinar ambas leyendas con figlegend
plt.figlegend(
    scatter_proxy_shapes + legend_patches,  # Formas de los modelos y colores de data augmentation
    list(df_total['model'].unique()) + list(categoria_orden),  # Etiquetas combinadas
    loc='upper center', bbox_to_anchor=(0.89, 0.81), title='Model and Data Augmentation types'
)

# Ajustar espacio manualmente para que no se corte
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

# Guardar imagen como svg, utilizando bbox_inches para evitar cortes
plt.savefig('/home/froh/f1_recall_size_data_aug_aligned.svg', format='svg', bbox_inches='tight')

plt.show()

# Cell 139
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames
df_total = pd.concat([df, df_em])

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Mapeo de etiquetas actuales a textos nuevos para la columna 'data_augmentation'
augmentation_label_mapping = {
    '01 undersampling': 'Undersampled',
    '02 sinónimos': 'Synonims',
    '00 original data': 'Original data',
    '04 insert BERT': 'Word insertions - BERT',
    '03 substitute BERT': 'Word substitutions - BERT',
    '05 sentence GPT2': 'GPT2 sentences',
    '06 sentence prompt GPT2': 'GPT2 prompts',
    '07 sentence prompt GPT4o mini': 'GPT4o mini prompts',
    '09 All augmented data': 'All augmented methods',
    '08 augmented prompt GPT4o mini': 'GPT4o mini prompts v2',
    '10 All augmented + aug gtp4 data': 'Consolidated data aug.',
}

# Aplicar el mapeo a la columna 'data_augmentation'
df_total['data_augmentation'] = df_total['data_augmentation'].map(augmentation_label_mapping)

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_total['tamaño_dataset_numeric'] = df_total['tamaño_dataset'].map(dataset_size_mapping)

# Especificar el orden de las categorías
categoria_orden = [
    'Original data', 'Undersampled', 'Synonims', 'Word insertions - BERT', 'Word substitutions - BERT',
    'GPT2 sentences', 'GPT2 prompts', 'GPT4o mini prompts',
    'All augmented methods', 'GPT4o mini prompts v2', 'Consolidated data aug.'
]

# Crear la figura en 3D con mayor espacio para la leyenda
fig = plt.figure(figsize=(12, 8))  # Aumenté el tamaño de la figura
ax = fig.add_subplot(111, projection='3d')

# Mapear los colores a 'data_augmentation'
colors = sns.color_palette('coolwarm', len(df_total['data_augmentation'].unique()))
color_mapping = {aug: colors[i] for i, aug in enumerate(categoria_orden)}
df_total['color'] = df_total['data_augmentation'].map(color_mapping)

# Mapear las formas a 'model'
markers = ['o', '^', 's', 'P', '*', 'X', 'D']
marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(df_total['model'].unique())}
df_total['marker'] = df_total['model'].map(marker_mapping)

# Crear el gráfico de dispersión 3D
for model in df_total['model'].unique():
    subset = df_total[df_total['model'] == model]
    ax.scatter(
        subset['recall_positive'],
        subset['f1_positive'],
        subset['tamaño_dataset_numeric'],
        c=subset['color'],
        marker=marker_mapping[model],
        s=50,  # Tamaño de los puntos
        alpha=0.7,
        label=model
    )

# Ajustar las etiquetas de los ejes, aumentar tamaño y alejar un milímetro
ax.set_xlabel('Test Recall', fontsize=18, labelpad=10)  # Ajustar labelpad (distancia) de la etiqueta del eje X
ax.set_ylabel('Test F1', fontsize=18, labelpad=10)      # Ajustar labelpad (distancia) de la etiqueta del eje Y
ax.set_zlabel('Dataset size', fontsize=16, labelpad=10)  # Ajustar labelpad (distancia) de la etiqueta del eje Z
plt.title('Test F1 Score, Test Recall and Dataset size\n(Binary Metrics)', fontsize=12)

# Aumentar el tamaño de los números (ticks) en los ejes
ax.tick_params(axis='both', which='major', labelsize=16)  # Para el eje X e Y
ax.tick_params(axis='z', which='major', labelsize=12)  # Para el eje Z

# Leyenda de Model (formas)
scatter_proxy_shapes = [plt.Line2D([0], [0], linestyle="none", marker=marker_mapping[model], markersize=10, color='black') for model in df_total['model'].unique()]
legend_patches = [plt.Line2D([0], [0], linestyle="none", marker="o", markersize=10, color=color_mapping[aug]) for aug in categoria_orden]

# Combinar ambas leyendas con figlegend
plt.figlegend(
    scatter_proxy_shapes + legend_patches,  # Formas de los modelos y colores de data augmentation
    list(df_total['model'].unique()) + list(categoria_orden),  # Etiquetas combinadas
    loc='upper center', bbox_to_anchor=(0.91, 0.96), title='Model and Data Augmentation types', fontsize=16  # Tamaño de la fuente de la leyenda
)

# Ajustar espacio manualmente para que no se corte
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

# Guardar imagen como svg, utilizando bbox_inches para evitar cortes
plt.savefig('/home/froh/f1_recall_size_data_aug_aligned.svg', format='svg', bbox_inches='tight')

plt.show()

# Cell 141
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

df_total = pd.concat([df, df_em])

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_total['tamaño_dataset_numeric'] = df_total['tamaño_dataset'].map(dataset_size_mapping)

# Crear la figura en 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Mapear los colores a 'data_augmentation'
colors = sns.color_palette('coolwarm', len(df_total['data_augmentation'].unique()))
color_mapping = {aug: colors[i] for i, aug in enumerate(df_total['data_augmentation'].unique())}
df_total['color'] = df_total['data_augmentation'].map(color_mapping)

# Crear el gráfico de dispersión 3D
scatter = ax.scatter(
    df_total['test_macro_recall'],
    df_total['test_macro_f1'],
    df_total['tamaño_dataset_numeric'],
    c=df_total['color'],
    s=50,  # Tamaño de los puntos
    alpha=0.7
)

# Ajustar las etiquetas de los ejes y el título
ax.set_xlabel('Test Recall')
ax.set_ylabel('Test F1')
ax.set_zlabel('Tamaño Dataset (numérico)')
plt.title('Relación Entre Test_f1, Test_recall y Tamaño Dataset en 3D\n(Métricas macro)')

# Colocar la leyenda
scatter_proxy = [plt.Line2D([0], [0], linestyle="none", marker="o", markersize=10, color=color) for color in colors]
ax.legend(scatter_proxy, df_total['data_augmentation'].unique(), title='Data Augmentation', bbox_to_anchor=(1.05, 1), loc='upper left')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Cell 143
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['text representation'] = df_tot['text representation'].fillna('No text representation')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['dataset size'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'text representation' en el orden deseado
text_representation_categories = ['No text representation', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_tot['text representation'] = pd.Categorical(df_tot['text representation'], categories=text_representation_categories, ordered=True)

# Crear un gráfico de dispersión combinando 'hue', 'style' y 'size'
plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(
    data=df_tot,
    x='recall_positive',
    y='f1_positive',
    hue='text representation',               # Usar 'text representation' para color
    style='model',                 # Símbolos por 'model'
    size='dataset size', # Tamaño por 'dataset_size'
    sizes=(50, 200),               # Ajuste del tamaño de los puntos
    palette='coolwarm'             # Usar la paleta de colores 'coolwarm'
)

# Ajustar la leyenda y los títulos
plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=11.5)
plt.title('Test F1 Score vs. Test Recall\n(Binary Metrics)')
plt.xlabel('Test Recall', fontsize=18)
plt.ylabel('Test F1 Score', fontsize=18)

# Ajustar el tamaño de los números en los ejes
plt.tick_params(axis='both', which='major', labelsize=16)  # Ajustar el tamaño de los ticks de los ejes X e Y

# Mostrar el gráfico
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/f1_recall_binary.svg', format='svg')

plt.show()

# Cell 145
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['text representation'] = df_tot['text representation'].fillna('No text representation')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['dataset size'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'text representation' en el orden deseado
text_representation_categories = ['No text representation', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'text representation' en un tipo categórico con el orden específico
df_tot['text representation'] = pd.Categorical(df_tot['text representation'], categories=text_representation_categories, ordered=True)

# Crear un gráfico de dispersión combinando 'hue', 'style' y 'size'
plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(
    data=df_tot,
    x='test_macro_recall',
    y='test_macro_f1',
    hue='text representation',               # Usar 'text representation' para color
    style='model',                 # Símbolos por 'model'
    size='dataset size', # Tamaño por 'dataset_size'
    sizes=(50, 200),               # Ajuste del tamaño de los puntos
    palette='coolwarm'             # Usar la paleta de colores 'coolwarm'
)

# Ajustar la leyenda y los títulos
plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=11.5)
plt.title('Test F1 Score vs. Test Recall\n(Macro Metrics)')
plt.xlabel('Test Recall', fontsize=18)
plt.ylabel('Test F1 Score', fontsize=18)

# Ajustar el tamaño de los números en los ejes
plt.tick_params(axis='both', which='major', labelsize=16)  # Ajustar el tamaño de los ticks de los ejes X e Y

# Mostrar el gráfico
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/f1_recall_macro.svg', format='svg')

plt.show()

# Cell 147
print(df_tot['text representation'].unique())

# Cell 148
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['text representation'] = df_tot['text representation'].fillna('No text representation')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['dataset size'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'text_representation' en el orden deseado
text_representation_categories = ['No text representation', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'text representation' en un tipo categórico con el orden específico
df_tot['text representation'] = pd.Categorical(df_tot['text representation'], categories=text_representation_categories, ordered=True)

# Crear un gráfico de dispersión combinando 'hue', 'style' y 'size'
plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(
    data=df_tot,
    x='TP',
    y='FP',
    hue='text representation',               # Usar 'text representation' para color
    style='model',                 # Símbolos por 'model'
    size='dataset size', # Tamaño por 'dataset_size'
    sizes=(50, 200),               # Ajuste del tamaño de los puntos
    palette='coolwarm'             # Usar la paleta de colores 'coolwarm'
)

# Ajustar la leyenda y los títulos
plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=11.85)
plt.title('Test - True Positives vs. Test - False Positives')
plt.xlabel('True Positives', fontsize=18)
plt.ylabel('False Positives', fontsize=18)

# Ajustar el tamaño de los números en los ejes
plt.tick_params(axis='both', which='major', labelsize=16)  # Ajustar el tamaño de los ticks de los ejes X e Y

# Mostrar el gráfico
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/TP_FP.svg', format='svg')

plt.show()

# Cell 151
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['text representation'] = df_tot['embedding'].fillna('No text representation')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['tamaño_dataset_numeric'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'embedding' en el orden deseado
text_representation_categories = ['No text representation', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'text representation' en un tipo categórico con el orden específico
df_tot['text representation'] = pd.Categorical(df_tot['text representation'], categories=text_representation_categories, ordered=True)

# Crear un gráfico de barras para mostrar el 'ROC_area'
plt.figure(figsize=(10, 4))  # Reducir el tamaño de la figura para hacerla menos ancha
ax = sns.barplot(
    data=df_tot,
    x='text representation',
    y='ROC_area',
    hue='model',                # Leyenda basada en 'model'
    palette='coolwarm',         # Usar la paleta de colores 'coolwarm'
    width=1.3                   # Reducir el ancho de las columnas para acercarlas entre sí
)

# Colocar las etiquetas del eje X dentro de las barras
for tick in ax.get_xticks():
    label = ax.get_xticklabels()[tick].get_text()  # Obtener el texto de la etiqueta
    if label == 'No text representation':
        ax.text(tick + 0.3, 0.05, label, ha='center', va='bottom', fontsize=12, rotation=90, color='black')  # Mover ligeramente la etiqueta a la derecha
    else:
        ax.text(tick, 0.05, label, ha='center', va='bottom', fontsize=12, rotation=90, color='black')  # Ajustar la posición y rotación de las etiquetas

# Eliminar las etiquetas originales del eje X
ax.set_xticklabels([])

# Ajustar el número de divisiones en el eje Y
plt.yticks([i/10 for i in range(0, 11)])  # Divisiones cada 0.1 de 0 a 1 en el eje Y

# Mover la leyenda a la derecha del gráfico, empezar desde la altura del título y usar 2 columnas
plt.legend(title='Model', bbox_to_anchor=(1, 1), loc='upper left', fontsize=14, title_fontsize=14, ncol=1)

# Ajustar títulos y etiquetas
plt.title('Test ROC area by Text representation and Model')
plt.xlabel('Text representation', fontsize=16)
plt.ylabel('Test ROC area', fontsize=16)

# Ajustar el tamaño de los valores en los ejes
plt.tick_params(axis='both', which='major', labelsize=14)

# Ajustar el gráfico para no cortar la leyenda ni las etiquetas giradas
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/ROC_embedding_model.svg', format='svg')

plt.show()

# Cell 153
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['embedding'] = df_tot['embedding'].fillna('No embedding')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['tamaño_dataset_numeric'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['No embedding', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_tot['embedding'] = pd.Categorical(df_tot['embedding'], categories=embedding_categories, ordered=True)

# Crear un gráfico de barras para mostrar el 'ROC_area'
plt.figure(figsize=(12, 3))
sns.barplot(
    data=df_tot,
    x='embedding',
    y='recall_positive',
    hue='model',                # Leyenda basada en 'model'
    palette='coolwarm'          # Usar la paleta de colores 'coolwarm'
)

# Ajustar la leyenda y los títulos
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Test Binary Recall Score by Embedding and Model')
plt.xlabel('Embedding')
plt.ylabel('Test Binary Recall')

# Mostrar el gráfico
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/recall_embedding_model.svg', format='svg')

plt.show()

# Cell 154
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['text representation'] = df_tot['embedding'].fillna('No text representation')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['tamaño_dataset_numeric'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'text_representation' en el orden deseado
text_representation_categories = ['No text representation', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'text_representation' en un tipo categórico con el orden específico
df_tot['text representation'] = pd.Categorical(df_tot['text representation'], categories=text_representation_categories, ordered=True)

# Crear un gráfico de barras para mostrar el 'recall_positive'
plt.figure(figsize=(10, 4))  # Ajustar el tamaño de la figura
ax = sns.barplot(
    data=df_tot,
    x='text representation',
    y='recall_positive',
    hue='model',                # Leyenda basada en 'model'
    palette='coolwarm',         # Usar la paleta de colores 'coolwarm'
    width=1.3                   # Ajustar el ancho de las barras para que se parezcan al gráfico de referencia
)

# Colocar las etiquetas del eje X dentro de las barras
for tick in ax.get_xticks():
    label = ax.get_xticklabels()[tick].get_text()  # Obtener el texto de la etiqueta
    if label == 'No text representation':
        ax.text(tick + 0.3, 0.05, label, ha='center', va='bottom', fontsize=12, rotation=90, color='black')  # Mover ligeramente la etiqueta a la derecha
    else:
        ax.text(tick, 0.05, label, ha='center', va='bottom', fontsize=12, rotation=90, color='black')  # Ajustar la posición y rotación de las etiquetas

# Eliminar las etiquetas originales del eje X
ax.set_xticklabels([])

# Ajustar el número de divisiones en el eje Y
# plt.yticks([i/10 for i in range(0, 11)])  # Divisiones cada 0.1 de 0 a 1 en el eje Y

# Mover la leyenda a la derecha del gráfico, empezar desde la altura del título y usar 2 columnas
plt.legend(title='Model', bbox_to_anchor=(1, 1), loc='upper left', fontsize=14, title_fontsize=14, ncol=1)

# Ajustar títulos y etiquetas
plt.xlabel('Text representation', fontsize=16)
plt.ylabel('Test Binary Recall', fontsize=16)
plt.title('Test Binary Recall by Text representation and Model')

# Ajustar el tamaño de los valores en los ejes
plt.tick_params(axis='both', which='major', labelsize=14)

# Ajustar el gráfico para no cortar la leyenda ni las etiquetas giradas
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/recall_embedding_model.svg', format='svg')

plt.show()

# Cell 156
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['embedding'] = df_tot['embedding'].fillna('No embedding')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['tamaño_dataset_numeric'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['No embedding', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_tot['embedding'] = pd.Categorical(df_tot['embedding'], categories=embedding_categories, ordered=True)

# Crear un gráfico de barras para mostrar el 'ROC_area'
plt.figure(figsize=(12, 4))
sns.barplot(
    data=df_tot,
    x='embedding',
    y='test_macro_recall',
    hue='model',                # Leyenda basada en 'model'
    palette='coolwarm'          # Usar la paleta de colores 'coolwarm'
)

# Ajustar la leyenda y los títulos
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Distribución de Test Macro Recall Score por Embedding y Modelo')
plt.xlabel('Embedding')
plt.ylabel('Test Macro recall')

# Mostrar el gráfico
plt.tight_layout()


plt.show()

# Cell 158
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['embedding'] = df_tot['embedding'].fillna('No embedding')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['tamaño_dataset_numeric'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'embedding' en el orden deseado
embedding_categories = ['No embedding', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'embedding' en un tipo categórico con el orden específico
df_tot['embedding'] = pd.Categorical(df_tot['embedding'], categories=embedding_categories, ordered=True)
df_tot= df_tot.dropna(subset=['f1_positive'])

# Crear un gráfico de barras para mostrar el 'ROC_area'
plt.figure(figsize=(12, 3))
sns.barplot(
    data=df_tot,
    x='embedding',
    y='f1_positive',
    hue='model',                # Leyenda basada en 'model'
    palette='coolwarm'          # Usar la paleta de colores 'coolwarm'
)

# Ajustar la leyenda y los títulos
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Test Binary F1 Score by Embedding and Model')
plt.xlabel('Embedding')
plt.ylabel('Test Binary F1 Score')

# Mostrar el gráfico
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/f1_embedding_model.svg', format='svg')

plt.show()

# Cell 159
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Concatenar los DataFrames y manejar los valores NaN
df_tot = pd.concat([df, df_em])
df_tot['text representation'] = df_tot['embedding'].fillna('No text representation')

# Mapeo de tamaño_dataset a valores numéricos
dataset_size_mapping = {
    '00. 3785 - 117': 3902,
    '01. 117 - 117': 234,
    '02. 1000 - 117': 1117,
    '03. 3785 - 2500': 6285,
    '04. 500 - 500': 1000,
    '05. 1000 - 1000': 2000,
    '06. 2500 - 2500': 5000,
    '07. 3785 - 3879': 7664,
    '08. 3785 - 5145': 8930,
    '09. 3785 - 21800': 25585,
    '10. 3785 - 26460': 30245
}

# Crear una nueva columna en el DataFrame con los tamaños numéricos
df_tot['tamaño_dataset_numeric'] = df_tot['tamaño_dataset'].map(dataset_size_mapping)

# Definir las categorías específicas para 'text_representation' en el orden deseado
text_representation_categories = ['No text representation', 'BoW', 'TF-IDF', 'Doc2Vec', 'FastText', 'prepr_SBERT', 'SBERT', 'text-embedding-3-large']

# Convertir la columna 'text representation' en un tipo categórico con el orden específico
df_tot['text representation'] = pd.Categorical(df_tot['text representation'], categories=text_representation_categories, ordered=True)

# Filtrar para asegurarse de que no haya valores NaN en 'f1_positive'
df_tot = df_tot.dropna(subset=['f1_positive'])

# Crear un gráfico de barras para mostrar el 'f1_positive'
plt.figure(figsize=(10, 4))  # Ajustar el tamaño de la figura
ax = sns.barplot(
    data=df_tot,
    x='text representation',
    y='f1_positive',
    hue='model',                # Leyenda basada en 'model'
    palette='coolwarm',         # Usar la paleta de colores 'coolwarm'
    width=1.3                   # Ajustar el ancho de las barras
)

# Colocar las etiquetas del eje X dentro de las barras
for tick in ax.get_xticks():
    label = ax.get_xticklabels()[tick].get_text()  # Obtener el texto de la etiqueta
    if label == 'No text representation':
        ax.text(tick + 0.3, 0.05, label, ha='center', va='bottom', fontsize=12, rotation=90, color='black')  # Mover ligeramente la etiqueta a la derecha
    else:
        ax.text(tick, 0.05, label, ha='center', va='bottom', fontsize=12, rotation=90, color='black')  # Ajustar la posición y rotación de las etiquetas

# Eliminar las etiquetas originales del eje X
ax.set_xticklabels([])

# Mover la leyenda a la derecha del gráfico, empezar desde la altura del título y usar 2 columnas
plt.legend(title='Model', bbox_to_anchor=(1, 1), loc='upper left', fontsize=14, title_fontsize=14, ncol=1)

# Ajustar títulos y etiquetas
plt.xlabel('Text representation', fontsize=16)
plt.ylabel('Test Binary F1 Score', fontsize=16)
plt.title('Test Binary F1 Score by Text representation and Model')

# Ajustar el tamaño de los valores en los ejes
plt.tick_params(axis='both', which='major', labelsize=14)

# Ajustar el gráfico para no cortar la leyenda ni las etiquetas giradas
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('/home/froh/f1_embedding_model.svg', format='svg')

plt.show()

# Cell 161
# Ejemplo de uso
#file_path = '/home/froh/results_final_summary_modelos.csv'
file_path = 'results_final_summary_modelos.csv'

data = pd.read_csv(file_path)

# Llenar valores nulos en 'embedding' con '0 sin embedding'
data['embedding'] = data['embedding'].fillna('0 sin embedding')
df = data

df_red = data[((data['Patience'] == 5) | (data['Patience'].isna())) & ((data['freezed_layers'] == 0) | (data['freezed_layers'].isna()))  & ((data['embedding'] == 'text-embedding-3-large') | (data['embedding']=='0 sin embedding'))]

# Ajustar el orden personalizado de model
order = ['bert', 'auto_bert', 'roberta', 'auto_roberta', 'rnn_lstm', 'rnn_gru', 'Logistic_Regression', 'Naive_Bayes', 'Random_Forest_Classifier','SVC']

# Generar gráficos de caja y bigote para comparar runtime_minutes por model con diferentes Pacience
plt.figure(figsize=(14, 13))

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 2, 1)
sns.boxplot(x=df['model'], y=df['test_f1'], data=df, order=order, showfliers=True)
plt.title('Distribución de Test f1 Score de todos los modelos\n(Binary F1 Score)')
plt.xticks(rotation=90)

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 2, 2)
sns.boxplot(x=df['model'], y=df['test_recall'], data=df, order=order, showfliers=True)
plt.title('Distribución de Test Recall de todos los modelos\n(Binary Recall Score)')
plt.xticks(rotation=90)

# Gráfico de caja y bigote de test_f1
plt.subplot(2, 2, 3)
sns.boxplot(x=df_red['model'], y=df_red['test_f1'], data=df_red, order=order, showfliers=True)
plt.title('Distribución de Test f1 Score por mejores modelos de cada tipología\n(Binary F1 Score)')
plt.xticks(rotation=90)

# Gráfico de caja y bigote de test_recall
plt.subplot(2, 2, 4)
sns.boxplot(x=df_red['model'], y=df_red['test_recall'], data=df_red, order=order, showfliers=True)
plt.title('Distribución de Test Recall por  mejores modelos de cada tipología\n(Binary Recall Score)')
plt.xticks(rotation=90)

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Cell 163
# Ejemplo de uso
#file_path = '/home/froh/results_final_summary_modelos.csv'
file_path = 'results_final_summary_modelos.csv'

data = pd.read_csv(file_path)

# Llenar valores nulos en 'embedding' con '0 sin embedding'
data['embedding'] = data['embedding'].fillna('0 sin embedding')

df_red = data[((data['Patience'] == 5) | (data['Patience'].isna())) & ((data['freezed_layers'] == 0) | (data['freezed_layers'].isna()))  & ((data['embedding'] == 'text-embedding-3-large') | (data['embedding']=='0 sin embedding'))]

# Ajustar el orden personalizado de model
order = ['bert', 'auto_bert', 'roberta', 'auto_roberta', 'rnn_lstm', 'rnn_gru', 'Logistic_Regression', 'Naive_Bayes', 'Random_Forest_Classifier','SVC']


# Seleccionar las columnas relevantes y renombrarlas para mayor claridad
df_selected = df_red[['model', 'test_f1', 'test_weighted_recall']].copy()
df_selected.columns = ['Modelo', 'f1_positive', 'recall_positive']

# Transformar el DataFrame a formato largo
df_long = pd.melt(df_selected, id_vars=['Modelo'], value_vars=['f1_positive', 'recall_positive'],
                  var_name='Metric', value_name='Valor')

# Crear dos subplots, uno para cada métrica
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Gráfico de f1_positive
sns.boxplot(data=df_red, x='model', y='f1_positive', ax=ax1, color='orange')
ax1.set_title('Test Binary F1 Score by model')
ax1.set_ylabel('Test Binary F1 Score')
#ax1.set_ylim(0, 0.8)

# Gráfico de recall_positive
sns.boxplot(data=df_red, x='model', y='recall_positive', ax=ax2, color='green')
ax2.set_title('Test Binary Recall by model')
ax2.set_ylabel('Test Binary Recall')

# Personalización de etiquetas
ax2.set_xlabel('Model')
plt.xticks(rotation=45)

# Ajustar el layout
plt.tight_layout()

# Guardar la gráfica en formato SVG
plt.savefig('resumen_modelos.svg', format='svg')

plt.show()

# Cell 165
import pandas as pd
from scipy.stats import shapiro, friedmanchisquare
from statsmodels.stats.anova import AnovaRM
import numpy as np

def analyze_data(df, id_vars, pivot_col, metric_columns):
    # Llenar valores nulos en la columna categórica con '0 sin valor'
    df.loc[:, pivot_col] = df[pivot_col].fillna('0 sin valor')

    # Obtener los valores únicos de la columna categórica
    category_values = df[pivot_col].dropna().unique().tolist()

    # Convertir la columna categórica en un tipo categórico con el orden específico
    df.loc[:, pivot_col] = pd.Categorical(df[pivot_col], categories=category_values, ordered=True)

    # Filtrar registros con valores nulos en las métricas importantes
    filtered_data = df.dropna(subset=metric_columns)

    # Pivotar los datos para tener una columna por cada valor de la columna categórica
    data_pivot = filtered_data.pivot_table(index=id_vars, columns=pivot_col, values=metric_columns, aggfunc='mean')

    # Asegurarse de que todas las categorías estén presentes en las columnas, incluso si están vacías
    all_columns = pd.MultiIndex.from_product([metric_columns, category_values], names=[None, pivot_col])
    data_pivot = data_pivot.reindex(columns=all_columns, fill_value=np.nan)

    # Aplanar el MultiIndex de las columnas
    data_pivot.columns = ['{}_{}'.format(var, cat) for var, cat in data_pivot.columns]

    # Resetear el índice para tener 'model' y 'dataset' como columnas
    data_pivot = data_pivot.reset_index()

    # Lista de variables a analizar
    for var in metric_columns:
        print(f"\nAnálisis de la variable: {var}\n")

        # Verificar si las columnas necesarias están presentes
        required_columns = [f'{var}_{cat}' for cat in category_values]
        available_columns = [col for col in required_columns if col in data_pivot.columns]
        missing_columns = [col for col in required_columns if col not in data_pivot.columns]

        if len(available_columns) < 2:
            print(f"No hay suficientes columnas para realizar comparaciones para la variable {var}. Faltan: {missing_columns}")
            continue

        # Preparar los datos para la prueba
        data_var = data_pivot[id_vars + available_columns]

        # Eliminar filas con valores nulos en cualquiera de las columnas seleccionadas
        #data_var = data_var.dropna()

        if data_var.empty:
            print(f"No hay suficientes datos para realizar el análisis para la variable {var} después de eliminar NaN.\n")
            continue

        print(f"Datos listos para el análisis de la variable {var}:")
        print(data_var.head())

        # Reestructurar los datos para ANOVA de medidas repetidas
        data_melt = pd.melt(data_var, id_vars=id_vars, value_vars=available_columns, var_name=pivot_col, value_name=var)

        # Crear una columna de sujeto único combinando 'model' y 'dataset'
        data_melt['subject'] = data_melt['model'] + '_' + data_melt['dataset']

        # Verificar normalidad de las diferencias entre cada par de configuraciones
        differences = {}
        pairs = [(available_columns[i], available_columns[j]) for i in range(len(available_columns)) for j in range(i+1, len(available_columns))]
        normal = True
        for (a, b) in pairs:
            diff = data_var[a] - data_var[b]

            # Verificar si la diferencia es NaN antes de la prueba de Shapiro
            if diff.isnull().all():
                print(f"Diferencia entre {a} y {b} no puede ser calculada, todos son NaN.")
                continue

            shapiro_stat, shapiro_p_value = shapiro(diff.dropna())  # Eliminar NaN antes de aplicar la prueba
            print(f"Prueba de Shapiro-Wilk para la normalidad de las diferencias ({a} vs {b}): statistic = {shapiro_stat}, p_value = {shapiro_p_value}")
            if shapiro_p_value > 0.05:
                print(f"Conclusión: Las diferencias entre {a} y {b} siguen una distribución normal.\n")
            else:
                print(f"Conclusión: Las diferencias entre {a} y {b} no siguen una distribución normal.\n")
                normal = False  # Si alguna comparación no es normal, usamos prueba no paramétrica

        # Seleccionar la prueba estadística según la normalidad
        if normal:
            print("Todas las diferencias siguen una distribución normal. Usaremos ANOVA de medidas repetidas.\n")
            data_melt[pivot_col] = data_melt[pivot_col].astype('category')
            aovrm = AnovaRM(data_melt, depvar=var, subject='subject', within=[pivot_col])
            res = aovrm.fit()
            print(res)
            p_value = res.anova_table['Pr > F'][0]
            if p_value < 0.05:
                print("Conclusión: Existe una diferencia significativa entre las categorías.\n")
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukey = pairwise_tukeyhsd(endog=data_melt[var], groups=data_melt[pivot_col], alpha=0.05)
                print(tukey)
            else:
                print("Conclusión: No hay diferencias significativas entre las categorías.\n")
        else:
            print("No se cumple la normalidad. Usaremos la prueba de Friedman.\n")
            # Eliminar filas con valores nulos antes de aplicar la prueba de Friedman
            data_friedman = data_var[available_columns].dropna()
            friedman_stat, friedman_p_value = friedmanchisquare(*[data_friedman[col] for col in data_friedman.columns])
            print(f"Resultado de la prueba de Friedman: statistic = {friedman_stat}, p_value = {friedman_p_value}")
            if friedman_p_value < 0.05:
                print("Conclusión: Existe una diferencia significativa entre las categorías.\n")
                import scikit_posthocs as sp
                nemenyi_results = sp.posthoc_nemenyi_friedman(data_friedman)
                nemenyi_results.index = available_columns
                nemenyi_results.columns = available_columns
                print("Resultados de la prueba post-hoc de Nemenyi:")
                print(nemenyi_results)
            else:
                print("Conclusión: No hay diferencias significativas entre las categorías.\n")

# Ejemplo de uso
file_path = '/home/froh/results_final_summary_modelos.csv'
data = pd.read_csv(file_path)

# Llenar valores nulos en 'embedding' con '0 sin embedding'
data['embedding'] = data['embedding'].fillna('0 sin embedding')
#df = data[((data['Patience'] == 5) | (data['Patience'].isna())) & ((data['freezed_layers'] == 0) | (data['freezed_layers'].isna()))]
df = data[data['Patience'].isna()]
id_vars = ['model', 'dataset']
pivot_col = 'embedding'
metric_columns = ['f1_positive', 'recall_positive', 'ROC_area']

analyze_data(df, id_vars, pivot_col, metric_columns)

